"""Batching, masked mean pooling, symmetric InfoNCE and the combined MT + contrastive loss."""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Rows of {english, pidgin} plus integer ids of their normalised text (for false-negative masking)."""

    def __init__(self, df: pd.DataFrame):
        self.english = df["english"].tolist()
        self.pidgin = df["pidgin"].tolist()
        self.eng_kid = pd.factorize(df["eng_norm"])[0].tolist()
        self.pcm_kid = pd.factorize(df["pcm_norm"])[0].tolist()

    def __len__(self) -> int:
        return len(self.english)

    def __getitem__(self, i: int) -> dict:
        return {"english": self.english[i], "pidgin": self.pidgin[i], "eng_kid": self.eng_kid[i], "pcm_kid": self.pcm_kid[i]}


def prefix_token_ids(tokenizer, prefix: str) -> list[int]:
    """Leading token ids shared by every `prefix + sentence`.

    Measured empirically (SentencePiece may attach the prefix's trailing space to the next word).
    """
    if not prefix:
        return []
    probes = ["Hello there.", "the man", "1990 was a year", "'Quote' me", "Ọmọ dey", "A"]
    ids = [tokenizer(prefix + p, add_special_tokens=False)["input_ids"] for p in probes]
    k = 0
    while all(len(x) > k for x in ids) and len({x[k] for x in ids}) == 1:
        k += 1
    return ids[0][:k]


class PairCollator:
    """Builds translation inputs/labels plus a separate Pidgin encoder view for the contrastive loss."""

    def __init__(self, tokenizer, prefix: str, max_source_length: int, max_target_length: int, pool_exclude_prefix: bool = True):
        self.tok = tokenizer
        self.prefix = prefix
        self.max_src = max_source_length
        self.max_tgt = max_target_length
        self.prefix_ids = prefix_token_ids(tokenizer, prefix) if pool_exclude_prefix else []
        self.prefix_len = len(self.prefix_ids)

    def __call__(self, items: list[dict]) -> dict:
        src = self.tok([self.prefix + x["english"] for x in items], max_length=self.max_src,
                       truncation=True, padding=True, return_tensors="pt")
        labels = self.tok(text_target=[x["pidgin"] for x in items], max_length=self.max_tgt,
                          truncation=True, padding=True, return_tensors="pt")["input_ids"]
        labels[labels == self.tok.pad_token_id] = -100
        # The Pidgin encoder view gets NO task prefix.
        pcm = self.tok([x["pidgin"] for x in items], max_length=self.max_tgt, truncation=True,
                       padding=True, return_tensors="pt")
        pool_mask = src["attention_mask"].clone()
        if self.prefix_len:
            expected = torch.tensor(self.prefix_ids)
            if not (src["input_ids"][:, : self.prefix_len] == expected).all():
                raise RuntimeError("task prefix did not tokenize identically across the batch")
            pool_mask[:, : self.prefix_len] = 0
        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "pool_mask": pool_mask,
            "labels": labels,
            "pcm_input_ids": pcm["input_ids"],
            "pcm_attention_mask": pcm["attention_mask"],
            "eng_kid": torch.tensor([x["eng_kid"] for x in items]),
            "pcm_kid": torch.tensor([x["pcm_kid"] for x in items]),
        }


def truncation_report(tokenizer, df: pd.DataFrame, prefix: str, max_src: int, max_tgt: int) -> dict:
    src = [len(x) for x in tokenizer([prefix + e for e in df["english"]])["input_ids"]]
    tgt = [len(x) for x in tokenizer(df["pidgin"].tolist())["input_ids"]]
    s, t = pd.Series(src), pd.Series(tgt)
    return {
        "source_tokens": {"mean": float(s.mean()), "p95": float(s.quantile(0.95)), "max": int(s.max())},
        "target_tokens": {"mean": float(t.mean()), "p95": float(t.quantile(0.95)), "max": int(t.max())},
        "source_truncated_frac": float((s > max_src).mean()),
        "target_truncated_frac": float((t > max_tgt).mean()),
    }


# --------------------------------------------------------------------------- pooling & loss

def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of hidden states over positions where mask == 1. hidden [B,T,d], mask [B,T] -> [B,d] (fp32)."""
    m = mask.unsqueeze(-1).to(torch.float32)
    return (hidden.float() * m).sum(1) / m.sum(1).clamp_min(1.0)


def duplicate_mask(eng_kid: torch.Tensor, pcm_kid: torch.Tensor) -> torch.Tensor:
    """True at off-diagonal (i,j) where rows i and j share English or Pidgin text: not real negatives."""
    same = (eng_kid[:, None] == eng_kid[None, :]) | (pcm_kid[:, None] == pcm_kid[None, :])
    same.fill_diagonal_(False)
    return same


def info_nce(eng_emb: torch.Tensor, pcm_emb: torch.Tensor, temperature: float, neg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives; positives on the diagonal. Always computed in fp32."""
    if eng_emb.size(0) < 2:
        raise ValueError("InfoNCE needs batch size >= 2 (batch size 1 has no negatives)")
    with torch.autocast(device_type=eng_emb.device.type, enabled=False):
        e = F.normalize(eng_emb.float(), dim=-1)
        p = F.normalize(pcm_emb.float(), dim=-1)
        scores = e @ p.T / temperature
        if neg_mask is not None:
            scores = scores.masked_fill(neg_mask, float("-inf"))
        target = torch.arange(scores.size(0), device=scores.device)
        return 0.5 * (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target))


def compute_losses(model, batch: dict, mode: str, lambda_cl: float = 0.0, temperature: float = 0.07,
                   mask_duplicates: bool = True):
    """Returns (total_loss, mt_loss, cl_loss_or_None).

    baseline: ordinary seq2seq cross-entropy.
    contrastive: the English encoder pass is shared between translation and the contrastive loss;
    Pidgin goes through the same encoder only for the contrastive loss.
    """
    if mode == "baseline":
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        return out.loss, out.loss, None
    if mode != "contrastive":
        raise ValueError(f"unknown mode {mode!r}")
    encoder = model.get_encoder()
    eng = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True)
    mt_loss = model(attention_mask=batch["attention_mask"], encoder_outputs=eng, labels=batch["labels"]).loss
    pcm = encoder(input_ids=batch["pcm_input_ids"], attention_mask=batch["pcm_attention_mask"], return_dict=True)
    e = masked_mean_pool(eng.last_hidden_state, batch["pool_mask"])
    p = masked_mean_pool(pcm.last_hidden_state, batch["pcm_attention_mask"])
    neg_mask = duplicate_mask(batch["eng_kid"], batch["pcm_kid"]) if mask_duplicates else None
    cl_loss = info_nce(e, p, temperature, neg_mask)
    return mt_loss + lambda_cl * cl_loss, mt_loss, cl_loss
