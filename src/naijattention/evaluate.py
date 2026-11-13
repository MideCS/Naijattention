"""Generation, chrF++ with paired bootstrap, encoder retrieval, human-eval template, inference demo."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import CHRF

from .modeling import masked_mean_pool, prefix_token_ids
from .utils import autocast

_CHRF = CHRF(word_order=2)  # chrF++


# --------------------------------------------------------------------------- generation

@torch.no_grad()
def generate(model, tokenizer, english: list[str], prefix: str, device, precision: str, *, batch_size: int,
             max_source_length: int, num_beams: int, max_new_tokens: int) -> list[str]:
    """Deterministic beam search. Inputs are length-sorted for speed, outputs returned in input order."""
    was_training = model.training
    model.eval()
    order = sorted(range(len(english)), key=lambda i: len(english[i]))
    out = [None] * len(english)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([prefix + english[i] for i in idx], max_length=max_source_length, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        with autocast(device, precision):
            gen = model.generate(**enc, num_beams=num_beams, do_sample=False, max_new_tokens=max_new_tokens)
        for i, text in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip()
    model.train(was_training)
    return out


# --------------------------------------------------------------------------- chrF++

def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    return _CHRF.corpus_score(hyps, [refs]).score


def sentence_chrf(hyps: list[str], refs: list[str]) -> list[float]:
    return [_CHRF.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def paired_bootstrap_chrf(hyp_a: list[str], hyp_b: list[str], refs: list[str], n: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over test rows for corpus chrF++(B) - chrF++(A). Recomputes corpus scores
    from summed per-sentence statistics, which is exactly how sacrebleu computes corpus chrF."""
    sa = np.asarray(_CHRF._extract_corpus_statistics(hyp_a, [refs]), dtype=np.float64)
    sb = np.asarray(_CHRF._extract_corpus_statistics(hyp_b, [refs]), dtype=np.float64)
    score = lambda stats: _CHRF._compute_score_from_stats(stats.tolist()).score  # noqa: E731
    base_a, base_b = score(sa.sum(0)), score(sb.sum(0))
    assert abs(base_a - corpus_chrf(hyp_a, refs)) < 1e-6, "chrF statistics path disagrees with corpus_score"
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, len(refs), len(refs))
        deltas[k] = score(sb[idx].sum(0)) - score(sa[idx].sum(0))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"a": base_a, "b": base_b, "delta": base_b - base_a, "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_rows": len(refs), "n_bootstrap": n}


# --------------------------------------------------------------------------- retrieval

@torch.no_grad()
def encode_sentences(model, tokenizer, texts: list[str], device, precision: str, *, prefix: str = "",
                     exclude_prefix: bool = False, batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    """L2-normalised masked-mean-pooled encoder embeddings, [N, d] float32."""
    was_training = model.training
    model.eval()
    k = len(prefix_token_ids(tokenizer, prefix)) if (prefix and exclude_prefix) else 0
    encoder, chunks = model.get_encoder(), []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer([prefix + t for t in texts[start:start + batch_size]], max_length=max_length,
                        truncation=True, padding=True, return_tensors="pt").to(device)
        with autocast(device, precision):
            h = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        mask = enc["attention_mask"].clone()
        mask[:, :k] = 0
        chunks.append(torch.nn.functional.normalize(masked_mean_pool(h, mask), dim=-1).cpu())
    model.train(was_training)
    return torch.cat(chunks).numpy()


def retrieval_metrics(E: np.ndarray, P: np.ndarray, pcm_keys=None, eng_keys=None, ks=(1, 5, 10)) -> dict:
    """Row i of E and P is a gold pair. Every query is ranked against ALL candidates."""
    assert E.shape == P.shape and np.isfinite(E).all() and np.isfinite(P).all()
    n = len(E)
    S = E @ P.T
    gold = np.diag(S)
    # Strict, index-based rank consistent with argmax: ties are broken toward the lower index,
    # so an identical duplicate candidate earlier in the pool beats the gold row.
    earlier = np.tril(np.ones((n, n), dtype=bool), k=-1)
    rank_fwd = (S > gold[:, None]).sum(1) + ((S == gold[:, None]) & earlier).sum(1)
    rank_rev = (S.T > gold[:, None]).sum(1) + ((S.T == gold[:, None]) & earlier).sum(1)
    res = {"n": n, "chance_r1": 1.0 / n}
    for k in ks:
        if k <= n:
            res[f"en2pcm_r@{k}"] = float((rank_fwd < k).mean())
    res["pcm2en_r@1"] = float((rank_rev < 1).mean())
    top = S.argmax(1)
    if pcm_keys is not None:  # relaxed: any candidate with identical normalised Pidgin counts as correct
        keys = np.asarray(pcm_keys)
        res["en2pcm_r@1_relaxed"] = float((keys[top] == keys).mean())
    off = S.copy()
    np.fill_diagonal(off, -np.inf)
    hardest = off.max(1)
    res.update(mean_pos_cos=float(gold.mean()), median_pos_cos=float(np.median(gold)),
               mean_hardest_neg_cos=float(hardest.mean()), mean_neg_cos=float((S.sum() - gold.sum()) / max(n * n - n, 1)))
    res["_per_query_correct"] = rank_fwd < 1
    res["_pos"], res["_hardest_neg"], res["_top"] = gold, hardest, top
    return res


def paired_bootstrap_mean(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    deltas = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"a": float(a.mean()), "b": float(b.mean()), "delta": float(b.mean() - a.mean()),
            "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def public(metrics: dict) -> dict:
    """Drop the per-query arrays (keys starting with '_') for JSON output."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- human evaluation

def difficulty_bucket(english: str) -> str:
    """Predeclared on the English source only, before looking at any model output."""
    words = len(english.split())
    clauses = len(re.findall(r"[,;:]| and | but | because | which | that | when | if ", f" {english.lower()} "))
    if words <= 8 and clauses == 0:
        return "simple"
    if words >= 20 or clauses >= 2:
        return "difficult"
    return "medium"


def human_eval_template(test: pd.DataFrame, hyp_a: list[str], hyp_b: list[str], per_bucket: int, seed: int):
    """Blinded A/B sheet (system order randomised per row) and a separate unblinding key."""
    d = test.assign(hyp_baseline=hyp_a, hyp_contrastive=hyp_b)
    d["difficulty"] = d["english"].map(difficulty_bucket)
    rng = np.random.default_rng(seed)
    picked = []
    for bucket in ("simple", "difficult"):
        pool = d[d["difficulty"] == bucket]
        for _, g in pool.groupby("source"):  # balance sources within each bucket
            picked.append(g.sample(n=min(len(g), -(-per_bucket // pool["source"].nunique())), random_state=seed))
    sel = pd.concat(picked).reset_index(drop=True) if picked else d.head(0)
    flip = rng.random(len(sel)) < 0.5
    sheet = pd.DataFrame({
        "item_id": [f"h{i:03d}" for i in range(len(sel))],
        "source": sel["source"], "difficulty": sel["difficulty"],
        "english": sel["english"], "reference_pidgin": sel["pidgin"],
        "system_A": np.where(flip, sel["hyp_contrastive"], sel["hyp_baseline"]),
        "system_B": np.where(flip, sel["hyp_baseline"], sel["hyp_contrastive"]),
    })
    for sysname in ("A", "B"):
        for crit in ("semantic_1to5", "grammar_1to5", "naturalness_1to5"):
            sheet[f"{sysname}_{crit}"] = ""
    sheet["error_notes"], sheet["reference_problem"] = "", ""
    key = pd.DataFrame({"item_id": sheet["item_id"], "source_id": sel["source_id"],
                        "system_A": np.where(flip, "contrastive", "baseline"),
                        "system_B": np.where(flip, "baseline", "contrastive")})
    return sheet, key


# --------------------------------------------------------------------------- inference demo

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def translate_to_pidgin(text: str, model, tokenizer, prefix: str, device, precision: str, *, num_beams: int = 4,
                        max_new_tokens: int = 128, max_source_length: int = 128) -> str:
    """Sentence-by-sentence translation of arbitrary English (e.g. LLM output). Cross-sentence context is lost."""
    sents = [s for s in _SENT_SPLIT.split(text.strip()) if s]
    outs = generate(model, tokenizer, sents, prefix, device, precision, batch_size=16,
                    max_source_length=max_source_length, num_beams=num_beams, max_new_tokens=max_new_tokens)
    return " ".join(outs)
