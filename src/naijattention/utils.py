"""Seeding, device/precision selection, environment reporting, small IO helpers."""
from __future__ import annotations

import contextlib
import gc
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(pref: str, device: torch.device) -> str:
    if pref == "fp16":
        raise ValueError("fp16 is not supported: mT5 activations overflow in fp16. Use bf16 or fp32.")
    if pref == "auto":
        # Require native bf16 (Ampere+). is_bf16_supported() also returns True for emulated bf16
        # on e.g. Colab T4s, which is very slow.
        native = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        return "bf16" if native else "fp32"
    if pref not in {"bf16", "fp32"}:
        raise ValueError(f"unknown precision {pref!r}")
    return pref


def autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def environment_report() -> dict:
    import sacrebleu
    import transformers

    rep = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sacrebleu": sacrebleu.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rep.update(
            gpu=props.name,
            gpu_memory_gb=round(props.total_memory / 2**30, 2),
            bf16_supported=torch.cuda.is_bf16_supported(),
            cuda=torch.version.cuda,
        )
    return rep


def free_memory() -> None:
    """Release Python garbage and PyTorch's cached GPU blocks between models. Without this, Windows
    machines with small GPUs can exhaust commit memory (spilled GPU memory counts against it) and the
    next model load crashes inside safetensors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
