"""Sanity checks that must pass before any training run. Each raises AssertionError on failure."""
from __future__ import annotations

import numpy as np
import torch

from .evaluate import retrieval_metrics
from .modeling import compute_losses, duplicate_mask, info_nce, masked_mean_pool


def check_pooling() -> None:
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(masked_mean_pool(h, m), torch.tensor([[2.0, 3.0]])), "pooling ignores padding incorrectly"
    # Appending more pad positions must not change the embedding.
    h2 = torch.cat([h, torch.full((1, 4, 2), -50.0)], dim=1)
    m2 = torch.cat([m, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    assert torch.allclose(masked_mean_pool(h2, m2), masked_mean_pool(h, m))
    # All-masked rows give a zero vector instead of NaN.
    assert torch.isfinite(masked_mean_pool(h, torch.zeros_like(m))).all()


def check_info_nce() -> None:
    B = 6
    eye = torch.eye(B)
    aligned = info_nce(eye, eye, temperature=0.05)
    shuffled = info_nce(eye, eye[torch.roll(torch.arange(B), 1)], temperature=0.05)
    assert torch.isfinite(aligned) and aligned >= 0
    assert aligned < 0.01, f"aligned identity embeddings should give ~0 loss, got {aligned}"
    assert shuffled > aligned + 1.0, "misaligned positives should increase the loss"
    try:
        info_nce(eye[:1], eye[:1], 0.07)
        raise AssertionError("batch size 1 must be rejected")
    except ValueError:
        pass
    # Duplicate masking removes false negatives but never the positive.
    keys = torch.tensor([0, 0, 1, 2, 3, 4])
    mask = duplicate_mask(keys, torch.arange(B))
    assert mask[0, 1] and mask[1, 0] and not mask.diagonal().any() and mask.sum() == 2
    dup = torch.stack([eye[0], eye[0], eye[2], eye[3], eye[4], eye[5]])
    assert info_nce(dup, dup, 0.05, mask) < info_nce(dup, dup, 0.05), "masking duplicates should lower the loss"


def check_retrieval_gate() -> None:
    E = np.eye(8, dtype=np.float32)
    assert retrieval_metrics(E, E)["en2pcm_r@1"] == 1.0
    assert retrieval_metrics(E, np.roll(E, 1, axis=0))["en2pcm_r@1"] == 0.0
    # Identical duplicate candidates: strict R@1 follows argmax (lower index wins), relaxed accepts both.
    P = E.copy()
    P[1] = P[0]
    r = retrieval_metrics(E, P, pcm_keys=["a", "a"] + list("cdefgh"))
    assert r["en2pcm_r@1_relaxed"] == 1.0 and r["en2pcm_r@1"] < 1.0


def check_model_losses(model, batch: dict, lambda_cl: float = 0.1, temperature: float = 0.07) -> dict:
    """On a real batch: lambda=0 contrastive path == baseline loss; combined loss reaches encoder and decoder.

    Pass a throwaway model: this runs backward passes on it (gradients are cleared afterwards, weights
    are not changed). Not deep-copied, to avoid holding two models' worth of memory.
    """
    model.eval()  # no dropout, so the two forward paths must agree
    with torch.no_grad():
        base, _, _ = compute_losses(model, batch, "baseline")
        zero, mt0, cl0 = compute_losses(model, batch, "contrastive", 0.0, temperature)
    assert torch.allclose(base, zero, atol=1e-4), f"lambda=0 contrastive loss {zero} != baseline {base}"
    assert torch.isfinite(cl0) and cl0 >= 0
    model.train()
    model.zero_grad(set_to_none=True)
    total, mt, cl = compute_losses(model, batch, "contrastive", lambda_cl, temperature)
    assert cl.requires_grad, "contrastive loss is detached"
    total.backward()
    enc_grad = model.get_encoder().block[0].layer[0].SelfAttention.q.weight.grad
    dec_grad = model.get_decoder().block[0].layer[0].SelfAttention.q.weight.grad
    for name, g in (("encoder", enc_grad), ("decoder", dec_grad)):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"no gradient reached the {name}"
    # The contrastive term alone must also produce encoder gradients.
    model.zero_grad(set_to_none=True)
    _, _, cl_only = compute_losses(model, batch, "contrastive", lambda_cl, temperature)
    cl_only.backward()
    g = model.get_encoder().block[0].layer[0].SelfAttention.q.weight.grad
    assert g is not None and g.abs().sum() > 0, "contrastive loss does not reach the encoder"
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"baseline_loss": float(base), "contrastive_lambda0_loss": float(zero), "cl_loss": float(cl), "total": float(total)}


def check_batch(batch: dict, tokenizer) -> dict:
    B = batch["input_ids"].size(0)
    for k in ("attention_mask", "pool_mask", "labels", "pcm_input_ids", "pcm_attention_mask"):
        assert batch[k].size(0) == B, f"{k} has wrong batch dim"
    assert not (batch["labels"] == tokenizer.pad_token_id).any(), "label padding must be -100, not pad id"
    assert (batch["pool_mask"] <= batch["attention_mask"]).all()
    assert (batch["pool_mask"].sum(1) > 0).all(), "a row has no English tokens left to pool"
    lab = batch["labels"].clone()
    lab[lab == -100] = tokenizer.pad_token_id
    return {"batch_size": B, "decoded_target_0": tokenizer.decode(lab[0], skip_special_tokens=True),
            "decoded_source_0": tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True),
            "decoded_pcm_view_0": tokenizer.decode(batch["pcm_input_ids"][0], skip_special_tokens=True)}


def run_static_checks() -> None:
    check_pooling()
    check_info_nce()
    check_retrieval_gate()
