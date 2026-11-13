"""One transparent training loop used for both the baseline and the contrastive model (plan §12, option A)."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from .config import Config
from .evaluate import corpus_chrf, generate
from .modeling import PairCollator, PairDataset, compute_losses
from .utils import autocast, free_memory, resolve_precision, save_json, seed_all

TENSOR_KEYS = ("input_ids", "attention_mask", "pool_mask", "labels", "pcm_input_ids", "pcm_attention_mask", "eng_kid", "pcm_kid")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if k in TENSOR_KEYS else v) for k, v in batch.items()}


def make_loader(df: pd.DataFrame, collator, batch_size: int, shuffle: bool, seed: int, drop_last: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(PairDataset(df), batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
                      generator=g, drop_last=drop_last, num_workers=0)


def make_optimizer(cfg: Config, model):
    if cfg.optimizer == "adafactor":
        return Adafactor(model.parameters(), lr=cfg.learning_rate, scale_parameter=False, relative_step=False,
                         warmup_init=False, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        no_decay = ("bias", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=cfg.learning_rate)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


@torch.no_grad()
def validation_loss(model, loader, device, precision) -> float:
    """Token-weighted mean translation cross-entropy."""
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device, precision):
            loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss
        n = int((batch["labels"] != -100).sum())
        total, tokens = total + float(loss) * n, tokens + n
    model.train(was_training)
    return total / max(tokens, 1)


def validate(model, tokenizer, cfg: Config, val: pd.DataFrame, collator, device, precision) -> dict:
    """Per-source validation CE and chrF++; selection score = unweighted mean of per-source chrF++."""
    out = {}
    for src, g in val.groupby("source"):
        loader = make_loader(g, collator, cfg.eval_batch_size, shuffle=False, seed=cfg.seed, drop_last=False)
        hyps = generate(model, tokenizer, g["english"].tolist(), cfg.source_prefix, device, precision,
                        batch_size=cfg.eval_batch_size, max_source_length=cfg.max_source_length,
                        num_beams=cfg.num_beams, max_new_tokens=cfg.max_new_tokens)
        out[src] = {"val_loss": validation_loss(model, loader, device, precision),
                    "val_chrf": corpus_chrf(hyps, g["pidgin"].tolist()),
                    "empty_outputs": sum(h == "" for h in hyps)}
    out["selection_chrf"] = sum(v["val_chrf"] for k, v in out.items() if isinstance(v, dict)) / len(out)
    return out


# Settings that do not change what a training run produces (so they don't invalidate a finished run).
_NON_TRAINING_KEYS = {"output_dir", "data_dir", "overwrite", "lambda_sweep", "bootstrap_samples",
                      "human_eval_per_bucket", "log_every"}


def run_fingerprint(cfg: Config, mode: str, lambda_cl: float, n_train: int) -> str:
    d = {k: v for k, v in cfg.to_dict().items() if k not in _NON_TRAINING_KEYS}
    d.update(mode=mode, lambda_cl=lambda_cl if mode == "contrastive" else 0.0, n_train=n_train)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def finished_run(run_dir: Path, fingerprint: str) -> dict | None:
    """The saved history of a completed run with identical settings, else None."""
    hist_path = run_dir / "train_history.json"
    if not (hist_path.exists() and (run_dir / "best" / "config.json").exists()):
        return None
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    if hist.get("config_fingerprint") != fingerprint:
        raise FileExistsError(f"{run_dir} holds a finished run with DIFFERENT settings; use another output_dir "
                              "or set overwrite=True to retrain")
    return hist


def train_run(cfg: Config, run_name: str, mode: str, lambda_cl: float, splits: dict, tokenizer, device, log=print) -> dict:
    """Train one fresh mT5 model and keep the checkpoint with the best validation selection chrF++.

    Resumable at run granularity: a finished run with identical settings is reused, not retrained
    (unless cfg.overwrite). Leftovers of an interrupted run are discarded and the run starts over.
    """
    run_dir = Path(cfg.output_dir) / run_name
    best_dir = run_dir / "best"
    fingerprint = run_fingerprint(cfg, mode, lambda_cl, len(splits["train"]))
    if not cfg.overwrite:
        done = finished_run(run_dir, fingerprint)
        if done is not None:
            log(f"[{run_name}] already finished with identical settings - reusing {best_dir}")
            return done
    if run_dir.exists():  # overwrite requested, or an interrupted run without a final history
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    precision = resolve_precision(cfg.precision, device)

    seed_all(cfg.seed)  # identical init + identical batch order for every run
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    collator = PairCollator(tokenizer, cfg.source_prefix, cfg.max_source_length, cfg.max_target_length, cfg.pool_exclude_prefix)
    # drop_last for BOTH runs so they see exactly the same batches (InfoNCE also needs B >= 2).
    loader = make_loader(splits["train"], collator, cfg.batch_size, shuffle=True, seed=cfg.seed, drop_last=True)
    accum = cfg.gradient_accumulation_steps
    updates_per_epoch = math.ceil(len(loader) / accum)
    total_updates = updates_per_epoch * cfg.epochs
    if cfg.max_train_steps:
        total_updates = min(total_updates, cfg.max_train_steps)
    opt = make_optimizer(cfg, model)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_updates), total_updates)

    history = {"run": run_name, "mode": mode, "config_fingerprint": fingerprint,
               "lambda_cl": lambda_cl, "temperature": cfg.temperature,
               "precision": precision, "batch_size": cfg.batch_size, "grad_accum": accum,
               "effective_batch": cfg.batch_size * accum, "total_updates_planned": total_updates,
               "steps": [], "epochs": []}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0, step, best, val_seconds = time.time(), 0, -math.inf, 0.0
    log(f"[{run_name}] training: {len(splits['train'])} rows, {len(loader)} batches/epoch, "
        f"{total_updates} updates planned, precision {precision}")
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        sums = {"total": 0.0, "mt": 0.0, "cl": 0.0}
        n_seen, n_batches = 0, len(loader)
        group = {"total": 0.0, "mt": 0.0, "cl": 0.0}
        for bi, batch in enumerate(loader):
            group_start = (bi // accum) * accum
            group_size = min(accum, n_batches - group_start)  # short final group is not under-weighted
            batch = to_device(batch, device)
            with autocast(device, precision):
                total, mt, cl = compute_losses(model, batch, mode, lambda_cl, cfg.temperature, cfg.mask_duplicate_negatives)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {bi}: {float(total)}")
            (total / group_size).backward()
            vals = {"total": float(total), "mt": float(mt), "cl": float(cl) if cl is not None else 0.0}
            for k in vals:
                group[k] += vals[k] / group_size
                sums[k] += vals[k]
            n_seen += 1
            if bi + 1 - group_start == group_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                history["steps"].append({"step": step, "epoch": epoch, "lr": sched.get_last_lr()[0], **group})
                if step % cfg.log_every == 0 or step == 1:
                    recent = history["steps"][-cfg.log_every:]
                    elapsed = time.time() - t0 - val_seconds
                    eta = elapsed / step * (total_updates - step)
                    log(f"[{run_name}] update {step}/{total_updates} | mt {sum(s['mt'] for s in recent) / len(recent):.3f}"
                        + (f" cl {sum(s['cl'] for s in recent) / len(recent):.3f}" if mode == "contrastive" else "")
                        + f" | lr {sched.get_last_lr()[0]:.2e} | {elapsed / 60:.1f} min, ~{eta / 60:.1f} min training left")
                group = {"total": 0.0, "mt": 0.0, "cl": 0.0}
                if step >= total_updates:
                    break
        log(f"[{run_name}] epoch {epoch}: validating on {len(splits['validation'])} sentences (beam {cfg.num_beams})...")
        tv = time.time()
        val = validate(model, tokenizer, cfg, splits["validation"], collator, device, precision)
        val_seconds += time.time() - tv
        log(f"[{run_name}] validation took {(time.time() - tv) / 60:.1f} min")
        rec = {"epoch": epoch, "updates": step, **{f"train_{k}": v / max(n_seen, 1) for k, v in sums.items()}, "val": val}
        history["epochs"].append(rec)
        improved = val["selection_chrf"] > best
        if improved:
            best = val["selection_chrf"]
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            history["best_epoch"], history["best_selection_chrf"] = epoch, best
        log(f"[{run_name}] epoch {epoch} updates {step} train_mt {rec['train_mt']:.3f}"
            + (f" train_cl {rec['train_cl']:.3f}" if mode == "contrastive" else "")
            + f" val_chrf {val['selection_chrf']:.2f}" + (" *saved*" if improved else ""))
        if step >= total_updates:
            break
    history.update(
        train_seconds=time.time() - t0, optimizer_updates=step,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
        best_dir=str(best_dir),
    )
    save_json(history, run_dir / "train_history.json")
    del model, opt, sched
    free_memory()
    return history


def load_best(run_dir: str | Path, device):
    from transformers import AutoTokenizer
    path = Path(run_dir) / "best"
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()
    return model, AutoTokenizer.from_pretrained(path)
