"""Experiment configuration. One dataclass, saved to config.json with every run."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class Config:
    seed: int = 42
    checkpoint: str = "google/mt5-small"
    output_dir: str = "outputs/pidgin_mt5"
    data_dir: str = "data"
    overwrite: bool = False  # refuse to clobber an existing run directory unless True

    # Data
    train_subset_size: int | None = None  # None = every cleaned, leakage-filtered training row
    source_prefix: str = "translate English to Nigerian Pidgin: "
    max_source_length: int = 128
    max_target_length: int = 128

    # Optimisation. mT5 needs a much higher LR than BERT-style models; Adafactor at 1e-3
    # is the standard T5/mT5 fine-tuning recipe and is far lighter on memory than AdamW.
    optimizer: str = "adafactor"  # "adafactor" | "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32  # per step; also the number of in-batch InfoNCE candidates
    gradient_accumulation_steps: int = 1
    eval_batch_size: int = 64
    epochs: int = 10
    max_train_steps: int | None = None  # cap on optimizer updates (smoke tests)
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    precision: str = "auto"  # "auto" -> bf16 if supported else fp32. fp16 is refused (mT5 overflows).
    gradient_checkpointing: bool = False
    log_every: int = 50  # print a progress line every N optimizer updates

    # Contrastive objective
    lambda_cl: float = 0.1
    lambda_sweep: list[float] = field(default_factory=list)  # e.g. [0.05, 0.5]; empty = only lambda_cl
    temperature: float = 0.07
    pool_exclude_prefix: bool = True  # pool English only over sentence tokens, not the task prefix
    mask_duplicate_negatives: bool = True  # don't treat same-text rows in a batch as negatives

    # Decoding / evaluation
    num_beams: int = 4
    max_new_tokens: int = 128
    bootstrap_samples: int = 1000
    human_eval_per_bucket: int = 20

    # Smoke-test sizes (only used when smoke_test=True)
    smoke_test: bool = False
    smoke_train_rows: int = 64
    smoke_eval_rows_per_source: int = 16

    @classmethod
    def smoke(cls, **overrides) -> "Config":
        """A config that exercises every code path in a few minutes on a small GPU or CPU."""
        base = cls(
            smoke_test=True,
            output_dir="outputs/smoke",
            overwrite=True,
            batch_size=8,
            eval_batch_size=16,
            max_source_length=64,
            max_target_length=64,
            epochs=2,
            max_train_steps=6,
            num_beams=2,
            max_new_tokens=24,
            bootstrap_samples=100,
            human_eval_per_bucket=4,
        )
        return replace(base, **overrides)

    @classmethod
    def pilot(cls, **overrides) -> "Config":
        """Full data and the full pipeline, capped at 300 updates per model: checks learning rate,
        timing and every full-size evaluation path before the real run (~10-15 min on an A100)."""
        return replace(cls(output_dir="outputs/pilot", epochs=1, max_train_steps=300), **overrides)

    @property
    def is_research_run(self) -> bool:
        return not self.smoke_test and self.max_train_steps is None

    @property
    def run_label(self) -> str:
        return "smoke" if self.smoke_test else ("pilot" if self.max_train_steps else "full")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
