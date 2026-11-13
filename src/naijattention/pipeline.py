"""End-to-end steps shared by the notebook and scripts/run_experiment.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import (build_splits, check_invariants, check_raw_counts, load_raw, save_splits, smoke_subset)
from .evaluate import (encode_sentences, generate, human_eval_template, paired_bootstrap_chrf,
                       paired_bootstrap_mean, public, retrieval_metrics, sentence_chrf)
from .train import load_best, train_run
from .utils import free_memory, resolve_precision, save_json


# --------------------------------------------------------------------------- data

def prepare_data(cfg: Config, log=print) -> tuple[dict[str, pd.DataFrame], dict]:
    raw, provenance = load_raw(cfg.data_dir)
    raw_counts = check_raw_counts(raw)
    splits, audit, removal_log = build_splits(raw, cfg.train_subset_size, cfg.seed)
    audit["invariants"] = check_invariants(splits, cfg.train_subset_size)
    audit.update(raw_counts=raw_counts, provenance=provenance)
    if cfg.smoke_test:
        splits = smoke_subset(splits, cfg.smoke_train_rows, cfg.smoke_eval_rows_per_source, cfg.seed)
        audit["SMOKE_TEST"] = "splits below are a tiny subset; results are NOT research results"
        audit["smoke_split_counts"] = {k: v.groupby("source").size().to_dict() for k, v in splits.items()}
    out = Path(cfg.output_dir)
    save_splits(splits, out, removal_log)
    save_json(audit, out / "data_audit.json")
    log("final split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return splits, audit


# --------------------------------------------------------------------------- training

def lambda_run_name(lam: float) -> str:
    return "contrastive_lam_" + f"{lam:g}".replace(".", "p")


def planned_runs(cfg: Config) -> list[tuple[str, str, float]]:
    lams = [cfg.lambda_cl] + [x for x in cfg.lambda_sweep if x != cfg.lambda_cl]
    return [("baseline", "baseline", 0.0)] + [(lambda_run_name(x), "contrastive", x) for x in lams]


def train_all(cfg: Config, splits, tokenizer, device, log=print, runs=None) -> dict[str, dict]:
    return {name: train_run(cfg, name, mode, lam, splits, tokenizer, device, log=log)
            for name, mode, lam in (runs or planned_runs(cfg))}


def pick_contrastive(histories: dict[str, dict]) -> str:
    """Choose lambda on VALIDATION selection chrF++ only."""
    cands = {k: v for k, v in histories.items() if v["mode"] == "contrastive"}
    return max(cands, key=lambda k: cands[k].get("best_selection_chrf", -np.inf))


# --------------------------------------------------------------------------- test evaluation

def evaluate_test(cfg: Config, splits, run_names: dict[str, str], device, log=print) -> dict:
    """run_names = {"baseline": <dir name>, "contrastive": <dir name>}. Test set is touched only here."""
    out = Path(cfg.output_dir)
    precision = resolve_precision(cfg.precision, device)
    test = splits["test"].reset_index(drop=True)
    hyps, emb, determinism = {}, {}, {}
    for role, run in run_names.items():
        model, tok = load_best(out / run, device)
        hyps[role] = generate(model, tok, test["english"].tolist(), cfg.source_prefix, device, precision,
                              batch_size=cfg.eval_batch_size, max_source_length=cfg.max_source_length,
                              num_beams=cfg.num_beams, max_new_tokens=cfg.max_new_tokens)
        again = generate(model, tok, test["english"].tolist()[:3], cfg.source_prefix, device, precision,
                         batch_size=cfg.eval_batch_size, max_source_length=cfg.max_source_length,
                         num_beams=cfg.num_beams, max_new_tokens=cfg.max_new_tokens)
        determinism[role] = again == hyps[role][:3]
        kw = dict(batch_size=cfg.eval_batch_size, max_length=cfg.max_source_length)
        emb[role] = {
            "E": encode_sentences(model, tok, test["english"].tolist(), device, precision, prefix=cfg.source_prefix,
                                  exclude_prefix=cfg.pool_exclude_prefix, **kw),
            "P": encode_sentences(model, tok, test["pidgin"].tolist(), device, precision, **kw),
        }
        del model
        free_memory()
        log(f"evaluated {role} ({run})")

    # Translations table (row-aligned for both systems).
    tr = test[["source", "source_id", "english", "pidgin"]].rename(columns={"pidgin": "reference"})
    for role in run_names:
        tr[f"hyp_{role}"] = hyps[role]
        tr[f"chrf_{role}"] = sentence_chrf(hyps[role], tr["reference"].tolist())
        tr[f"empty_{role}"] = [h == "" for h in hyps[role]]
    (out / "results").mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / "results" / "translations.csv", index=False)

    metrics = {"chrf": {}, "retrieval": {}, "runs": run_names, "determinism_spot_check": determinism,
               "run_label": cfg.run_label, "is_research_run": cfg.is_research_run}
    raw_ret = {}
    groups = [(src, test.index[test["source"] == src].to_numpy()) for src in sorted(test["source"].unique())]
    groups.append(("pooled", test.index.to_numpy()))
    for name, idx in groups:
        refs = test.loc[idx, "pidgin"].tolist()
        metrics["chrf"][name] = paired_bootstrap_chrf([hyps["baseline"][i] for i in idx], [hyps["contrastive"][i] for i in idx],
                                                      refs, n=cfg.bootstrap_samples, seed=cfg.seed)
        metrics["chrf"][name]["note"] = "a = baseline, b = contrastive; corpus chrF++ (word_order=2)"
        rr = {}
        for role in run_names:
            rr[role] = retrieval_metrics(emb[role]["E"][idx], emb[role]["P"][idx], pcm_keys=test.loc[idx, "pcm_norm"].tolist())
        raw_ret[name] = rr
        metrics["retrieval"][name] = {role: public(m) for role, m in rr.items()}
        metrics["retrieval"][name]["r1_delta_bootstrap"] = paired_bootstrap_mean(
            rr["baseline"]["_per_query_correct"], rr["contrastive"]["_per_query_correct"], n=cfg.bootstrap_samples, seed=cfg.seed)
    save_json(metrics, out / "results" / "metrics.json")

    sheet, key = human_eval_template(test, hyps["baseline"], hyps["contrastive"], cfg.human_eval_per_bucket, cfg.seed)
    sheet.to_csv(out / "results" / "human_eval_template.csv", index=False)
    key.to_csv(out / "results" / "human_eval_KEY_do_not_open_before_scoring.csv", index=False)
    return {"metrics": metrics, "translations": tr, "retrieval_raw": raw_ret, "embeddings": emb}


# --------------------------------------------------------------------------- reporting

def results_table(metrics: dict, histories: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for role, run in metrics["runs"].items():
        h = histories.get(run, {})
        row = {"model": role, "run": run}
        for name, m in metrics["chrf"].items():
            row[f"chrF++ {name}"] = m["a"] if role == "baseline" else m["b"]
        for name, r in metrics["retrieval"].items():
            row[f"R@1 {name}"] = r[role]["en2pcm_r@1"]
            row[f"R@1 pcm->en {name}"] = r[role]["pcm2en_r@1"]
        row["train_min"] = h.get("train_seconds", float("nan")) / 60
        row["updates"] = h.get("optimizer_updates")
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def delta_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, m in metrics["chrf"].items():
        rows.append({"metric": f"chrF++ {name}", "baseline": m["a"], "contrastive": m["b"], "delta": m["delta"],
                     "ci95_lo": m["ci95"][0], "ci95_hi": m["ci95"][1], "ci_excludes_0": m["ci_excludes_zero"]})
    for name, r in metrics["retrieval"].items():
        d = r["r1_delta_bootstrap"]
        rows.append({"metric": f"R@1 en->pcm {name} (chance {r['baseline']['chance_r1']:.4f})", "baseline": d["a"],
                     "contrastive": d["b"], "delta": d["delta"], "ci95_lo": d["ci95"][0], "ci95_hi": d["ci95"][1],
                     "ci_excludes_0": d["ci_excludes_zero"]})
    return pd.DataFrame(rows).set_index("metric")


def training_health(histories: dict[str, dict]) -> pd.DataFrame:
    """Is the MT loss falling, are validation outputs non-empty, and how long did it take?
    Untrained mT5 starts around MT loss 15-25 (it was pretrained only on span corruption); a healthy
    fine-tune should be well under ~5 within a few hundred updates."""
    rows = []
    for name, h in histories.items():
        mt = [s["mt"] for s in h["steps"]]
        k = max(1, len(mt) // 10)
        val = h["epochs"][-1]["val"]
        rows.append({
            "run": name, "updates": h["optimizer_updates"],
            "mt_loss_first10pct": float(np.mean(mt[:k])) if mt else np.nan,
            "mt_loss_last10pct": float(np.mean(mt[-k:])) if mt else np.nan,
            "val_chrf_last_epoch": val["selection_chrf"],
            "empty_val_outputs": sum(v["empty_outputs"] for v in val.values() if isinstance(v, dict)),
            "minutes_incl_validation": h["train_seconds"] / 60,
            "peak_gpu_gb": h.get("peak_gpu_mem_gb"),
        })
    return pd.DataFrame(rows).set_index("run")


def plot_learning_curves(histories: dict[str, dict], path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, h in histories.items():
        steps = pd.DataFrame(h["steps"])
        ep = pd.DataFrame([{"epoch": e["epoch"], "val_chrf": e["val"]["selection_chrf"],
                            "val_loss": np.mean([v["val_loss"] for v in e["val"].values() if isinstance(v, dict)])}
                           for e in h["epochs"]])
        axes[0].plot(steps["step"], steps["mt"], label=f"{name} MT", alpha=0.8)
        if h["mode"] == "contrastive":
            axes[0].plot(steps["step"], steps["cl"], "--", label=f"{name} CL", alpha=0.8)
        axes[1].plot(ep["epoch"], ep["val_loss"], marker="o", label=name)
        axes[2].plot(ep["epoch"], ep["val_chrf"], marker="o", label=name)
    for ax, t in zip(axes, ["train loss per update", "validation MT loss (mean over sources)", "validation chrF++ (selection)"]):
        ax.set_title(t)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("optimizer update")
    axes[1].set_xlabel("epoch")
    axes[2].set_xlabel("epoch")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return fig


def plot_alignment(raw_ret: dict, path: Path, pool: str = "pooled"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (role, m) in zip(axes, raw_ret[pool].items()):
        ax.hist(m["_pos"], bins=40, alpha=0.6, label="gold pair cosine")
        ax.hist(m["_hardest_neg"], bins=40, alpha=0.6, label="hardest negative cosine")
        ax.set_title(f"{role}: R@1 = {m['en2pcm_r@1']:.3f} ({pool}, N={m['n']})")
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return fig


def _saved_tokenizers_match(cfg: Config, histories: list[dict], test: pd.DataFrame) -> bool:
    """Saved checkpoints must tokenize exactly like the original checkpoint (guards against
    tokenizer save/reload drift between transformers versions)."""
    from transformers import AutoTokenizer
    texts = [cfg.source_prefix + e for e in test["english"]] + test["pidgin"].tolist()
    ref = AutoTokenizer.from_pretrained(cfg.checkpoint)(texts)["input_ids"]
    return all(AutoTokenizer.from_pretrained(h["best_dir"])(texts)["input_ids"] == ref for h in histories)


def integrity_checks(cfg: Config, splits, histories: dict[str, dict], result: dict) -> dict:
    tr, te = result["translations"], splits["test"]
    b, c = histories[result["metrics"]["runs"]["baseline"]], histories[result["metrics"]["runs"]["contrastive"]]
    checks = {
        "same_test_rows_for_both_systems": len(tr) == len(te) and (tr["source_id"].values == te["source_id"].values).all(),
        "matched_optimizer_updates": b["optimizer_updates"] == c["optimizer_updates"],
        "matched_effective_batch": b["effective_batch"] == c["effective_batch"],
        "no_test_rows_in_train": not splits["train"]["source_id"].isin(set(te["source_id"])).any(),
        "deterministic_generation": all(result["metrics"]["determinism_spot_check"].values()),
        "checkpoints_exist": all((Path(h["best_dir"]) / "config.json").exists() for h in (b, c)),
        "all_metrics_finite": all(np.isfinite(m["delta"]) for m in result["metrics"]["chrf"].values()),
        "saved_tokenizers_match_original": _saved_tokenizers_match(cfg, [b, c], te),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"integrity checks failed: {failed}")
    return {k: bool(v) for k, v in checks.items()}
