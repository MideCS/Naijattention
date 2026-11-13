"""Run the whole experiment from the command line.

    python scripts/run_experiment.py --smoke            # a few minutes, NOT research results
    python scripts/run_experiment.py                    # full run (GPU required)
    python scripts/run_experiment.py --set batch_size=16 --set lambda_sweep=[0.05,0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from naijattention import Config  # noqa: E402
from naijattention.checks import check_batch, check_model_losses, run_static_checks  # noqa: E402
from naijattention.modeling import PairCollator, truncation_report  # noqa: E402
from naijattention.pipeline import (delta_table, evaluate_test, integrity_checks, pick_contrastive,  # noqa: E402
                                    plot_alignment, plot_learning_curves, prepare_data, results_table, train_all,
                                    training_health)
from naijattention.train import make_loader, to_device  # noqa: E402
from naijattention.utils import (environment_report, free_memory, get_device, resolve_precision,  # noqa: E402
                                 save_json, seed_all)


def parse_overrides(pairs: list[str]) -> dict:
    types = {f.name for f in fields(Config)}
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown config key {k}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="tiny subset, a few minutes, any device")
    mode.add_argument("--pilot", action="store_true", help="full data, 300 updates per model")
    ap.add_argument("--output-root", help="parent directory for outputs, e.g. a mounted Google Drive folder")
    ap.add_argument("--set", action="append", default=[], help="config override key=value (JSON values)")
    args = ap.parse_args()
    overrides = parse_overrides(args.set)
    preset = Config.smoke if args.smoke else Config.pilot if args.pilot else Config
    cfg = preset()
    if args.output_root and "output_dir" not in overrides:
        overrides["output_dir"] = str(Path(args.output_root) / Path(cfg.output_dir).name)
    cfg = replace(cfg, **overrides)
    print(f"RUN MODE: {cfg.run_label} -> {cfg.output_dir}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    env = environment_report()
    save_json(env, out / "environment.json")
    print(json.dumps(env, indent=2))
    seed_all(cfg.seed)
    device = get_device()
    print("device:", device, "precision:", resolve_precision(cfg.precision, device))

    run_static_checks()
    splits, audit = prepare_data(cfg)
    print(json.dumps({k: audit[k] for k in ("leakage_removed", "duplicate_pairs_removed", "split_counts",
                                              "published_dev_test_overlap")}, indent=2))

    tok = AutoTokenizer.from_pretrained(cfg.checkpoint)
    print("truncation:", truncation_report(tok, splits["train"], cfg.source_prefix, cfg.max_source_length, cfg.max_target_length))
    collator = PairCollator(tok, cfg.source_prefix, cfg.max_source_length, cfg.max_target_length, cfg.pool_exclude_prefix)
    batch = next(iter(make_loader(splits["train"], collator, min(cfg.batch_size, 4), False, cfg.seed, True)))
    print("batch check:", check_batch(batch, tok))
    from transformers import AutoModelForSeq2SeqLM
    probe = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint).to(device)
    print("loss checks:", check_model_losses(probe, to_device(batch, device), cfg.lambda_cl, cfg.temperature))
    del probe
    free_memory()

    histories = train_all(cfg, splits, tok, device)
    best_cl = pick_contrastive(histories)
    print("selected contrastive run (by validation chrF++):", best_cl)
    result = evaluate_test(cfg, splits, {"baseline": "baseline", "contrastive": best_cl}, device)
    plot_learning_curves(histories, out / "figures" / "learning_curves.png")
    plot_alignment(result["retrieval_raw"], out / "figures" / "alignment_histogram.png")
    checks = integrity_checks(cfg, splits, histories, result)
    save_json(checks, out / "results" / "integrity_checks.json")

    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(results_table(result["metrics"], histories).round(3).T)
        print(delta_table(result["metrics"]).round(3))
    print("integrity:", checks)
    if not cfg.is_research_run:
        print(f"\n{cfg.run_label.upper()} RUN COMPLETE - these numbers are not research results.")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(training_health(histories).round(3))


if __name__ == "__main__":
    main()
