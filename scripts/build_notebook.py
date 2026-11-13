"""Generate notebooks/naijattention.ipynb. Edit cells here, then re-run this script."""
from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/MideCS/Naijattention.git"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Evaluating Contrastive Learning for English → Nigerian Pidgin Translation

**Research question.** With training data, architecture, budget and decoding held fixed, does adding a
cross-lingual sentence-level contrastive objective to translation cross-entropy improve (a) English→Pidgin
translation quality (chrF++) and/or (b) English–Pidgin encoder embedding alignment (top-1 retrieval)?

| Experiment | Initialisation | Objective | Inference |
|---|---|---|---|
| Baseline | `google/mt5-small` | translation CE | encoder–decoder EN→PCM |
| Contrastive | fresh `google/mt5-small`, same seed | CE + λ · symmetric InfoNCE | identical |

```
English (+ task prefix) ──► mT5 encoder ──► decoder ──► Pidgin      (L_MT, both models)
        │                        │
        └── mean-pool (sentence tokens) ──┐
Pidgin reference ──► same encoder ──► mean-pool ──► InfoNCE over the batch (L_CL, contrastive model only)
```

Data: MAFAND-MT en-pcm (news), UD Naija-NSC (spoken), MTurk Pidgin CSV (training only), pinned to exact
upstream commits. Validation/test are the **published** MAFAND and Naija dev/test splits, reported per source.

**How to run on Colab (A100):** Runtime → Run all with `RUN_MODE = "pilot"` (the default), check the
training-health table, then set `RUN_MODE = "full"` and Run all again. Outputs are saved to
`MyDrive/naijattention/outputs/`. If Colab disconnects, Run all again: finished training runs are reused.
""")

md("## 1. Environment")
code(f"""
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.chdir("/content")
    if Path("Naijattention").exists():
        subprocess.run(["git", "-C", "Naijattention", "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "{REPO_URL}"], check=True)
    os.chdir("Naijattention")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
    # Outputs (checkpoints, results) go to Google Drive so they survive a disconnect. Re-running
    # the notebook after a disconnect reuses every finished training run and redoes only the rest.
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/naijattention/outputs")
else:  # local: run from the repo root regardless of where Jupyter was started
    root = Path.cwd()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    os.chdir(root)
    OUTPUT_ROOT = Path("outputs")
sys.path.insert(0, str(Path.cwd() / "src"))
print("working directory:", Path.cwd(), "| outputs under:", OUTPUT_ROOT)
""")
code("""
import json
from dataclasses import replace
import numpy as np, pandas as pd, torch
import matplotlib
from IPython.display import display, Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from naijattention import Config
from naijattention import checks
from naijattention.data import SOURCES
from naijattention.modeling import PairCollator, truncation_report
from naijattention.train import make_loader, to_device, load_best
from naijattention.evaluate import translate_to_pidgin
from naijattention.pipeline import (prepare_data, planned_runs, train_all, pick_contrastive, evaluate_test,
                                    results_table, delta_table, plot_learning_curves, plot_alignment, integrity_checks)
from naijattention.utils import environment_report, free_memory, get_device, resolve_precision, save_json, seed_all

pd.set_option("display.max_colwidth", 120)
env = environment_report()
env
""")

md("""
## 2. Configuration

Defaults follow the implementation plan with these deliberate changes: Adafactor at lr 1e-3 (the standard
mT5 recipe; 3e-5 AdamW barely moves mT5), batch 32 per step (more in-batch negatives), 10 epochs, BF16 or FP32
only (mT5 overflows in FP16), English pooled over sentence tokens only, and same-text rows in a batch
masked out as negatives. Change anything here; the full config is saved with the outputs.
""")
code("""
#   "smoke": 64 training rows, a few minutes on any GPU/CPU - checks the code runs
#   "pilot": full data, 300 updates per model (~10-15 min on an A100) - checks LR, timing, full-size eval
#   "full" : the real experiment
RUN_MODE = os.environ.get("NAIJA_MODE", "pilot")   # <- set to "full" once the pilot looks healthy

overrides = dict(
    # train_subset_size=25000,      # None = all ~35k cleaned rows
    # lambda_sweep=[0.05, 0.5],     # extra contrastive runs; lambda chosen on validation chrF++ only
    # batch_size=16, gradient_accumulation_steps=2,   # if memory is tight (fewer negatives!)
)
preset = {"smoke": Config.smoke, "pilot": Config.pilot, "full": Config}[RUN_MODE]
cfg = preset()
cfg = replace(cfg, output_dir=str(OUTPUT_ROOT / Path(cfg.output_dir).name), **overrides)
device = get_device()
precision = resolve_precision(cfg.precision, device)
out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
cfg.save(out / "config.json"); save_json(env, out / "environment.json")
seed_all(cfg.seed)
print(f"RUN MODE: {cfg.run_label} | outputs: {out} | device: {device} | precision: {precision}")
if not cfg.is_research_run:
    print("NOTE: this is a", cfg.run_label, "run - its numbers are not research results.")
cfg
""")

md("## 3. Data: load, audit, clean, leakage-filter, split")
code("""
pd.DataFrame(SOURCES).T[["repo", "sha", "license", "domain"]]
""")
code("""
splits, audit = prepare_data(cfg)
print(json.dumps({k: audit[k] for k in ["raw_counts", "leakage_removed", "duplicate_pairs_removed", "train_english_with_multiple_targets",
                                         "published_dev_test_overlap", "naija_cross_split_overlap",
                                         "eval_repeated_pidgin_rows", "split_counts"]}, indent=1))
print("provenance dropped rows:", audit["provenance"]["dropped_missing_fields"])
print("invariants:", audit["invariants"])
""")
md("""
Published dev/test overlap is reported, not repaired: the benchmarks stay comparable with other work.
Train is disjoint from both held-out sets on normalised English **and** Pidgin.
""")
code("""
train = splits["train"]
sample = pd.concat([train.sample(min(15, len(train)), random_state=cfg.seed),
                    train.loc[[train["english"].str.len().idxmin(), train["english"].str.len().idxmax()]]])
display(sample[["source", "english", "pidgin"]])
lengths = pd.DataFrame({"source": train["source"], "en_words": train["english"].str.split().str.len(),
                        "pcm_words": train["pidgin"].str.split().str.len()})
display(lengths.groupby("source").describe().round(1))
""")

md("""
### Qualitative alignment audit (manual)

Fill in `flag` for ~50 random training pairs (`accurate_translation`, `paraphrase`, `potential_misalignment`,
`other`). Save the scored sheet next to the outputs. This is descriptive only; nothing is filtered from it.
""")
code("""
audit_sheet = pd.concat([g.sample(min(len(g), 17), random_state=cfg.seed) for _, g in train.groupby("source")])
audit_sheet = audit_sheet[["source", "source_id", "english", "pidgin"]].assign(flag="")
audit_sheet.to_csv(out / "alignment_audit_template.csv", index=False)
print(len(audit_sheet), "rows written to", out / "alignment_audit_template.csv")
""")

md("## 4. Tokenisation, batches and pre-training checks")
code("""
tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
print(json.dumps(truncation_report(tokenizer, train, cfg.source_prefix, cfg.max_source_length, cfg.max_target_length), indent=1))
collator = PairCollator(tokenizer, cfg.source_prefix, cfg.max_source_length, cfg.max_target_length, cfg.pool_exclude_prefix)
print("task-prefix tokens excluded from English pooling:", collator.prefix_len)
batch = next(iter(make_loader(train, collator, min(cfg.batch_size, 4), shuffle=False, seed=cfg.seed, drop_last=True)))
print({k: tuple(v.shape) for k, v in batch.items()})
checks.check_batch(batch, tokenizer)
""")
md("Unit checks for pooling, InfoNCE and retrieval, then on the real model: λ=0 matches the baseline loss, and gradients reach the encoder and decoder.")
code("""
checks.run_static_checks()
probe = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint).to(device)
print(checks.check_model_losses(probe, to_device(batch, device), cfg.lambda_cl, cfg.temperature))
with torch.no_grad():
    enc = tokenizer([cfg.source_prefix + "Good morning, how are you?"] * 2, return_tensors="pt").to(device)
    print("pre-training generate works:", tokenizer.batch_decode(probe.generate(**enc, max_new_tokens=8), skip_special_tokens=True))
del probe; free_memory()
""")

md("""
## 5. Training: baseline and contrastive

Both runs start from the same pretrained weights and seed, see identical batches in identical order, use the
same optimizer, schedule and number of updates, and keep the epoch with the best **validation** chrF++
(mean of MAFAND-dev and Naija-dev). The test set is not touched here.
""")
code("""
print(planned_runs(cfg))
histories = train_all(cfg, splits, tokenizer, device)
best_cl = pick_contrastive(histories)
print("contrastive run selected on validation chrF++:", best_cl)
pd.DataFrame([{"run": k, "best_epoch": h.get("best_epoch"), "best_val_chrf": h.get("best_selection_chrf"),
               "updates": h["optimizer_updates"], "minutes": h["train_seconds"] / 60,
               "peak_gpu_gb": h["peak_gpu_mem_gb"]} for k, h in histories.items()])
""")
md("""
**Training health.** Untrained mT5 starts at an MT loss of about 15–25. A healthy run should be well under 5
within a few hundred updates and give non-empty validation outputs. In a pilot, check this table before
switching to `RUN_MODE = "full"`. If the loss is flat, revisit `learning_rate`. The pilot's minutes ×
(full updates / 300) gives a rough time for the full run.
""")
code("""
from naijattention.pipeline import training_health
training_health(histories).round(3)
""")
code("""
plot_learning_curves(histories, out / "figures" / "learning_curves.png");
""")
md("How large is λ·L_CL compared with L_MT? If it is tiny, the contrastive term barely affects training.")
code("""
h = histories[best_cl]
pd.DataFrame([{"epoch": e["epoch"], "train_mt": e["train_mt"], "train_cl": e["train_cl"],
               "lambda*cl": h["lambda_cl"] * e["train_cl"]} for e in h["epochs"]]).round(4)
""")

md("## 6. Test evaluation: chrF++ and encoder retrieval")
code("""
result = evaluate_test(cfg, splits, {"baseline": "baseline", "contrastive": best_cl}, device)
display(results_table(result["metrics"], histories).round(3).T)
display(delta_table(result["metrics"]).round(3))
""")
md("""
chrF++ is SacreBLEU corpus chrF with `word_order=2`, computed separately per test source. The 95% CIs come
from a paired bootstrap over test rows: they cover test-sample noise but **not** seed-to-seed training variance.
Retrieval ranks every English test sentence against all Pidgin references in the same pool. Strict R@1 is
`argmax == gold`; the relaxed variant accepts identical-text duplicates, which matters for Naija's repeated short utterances.
""")
code("""
display(pd.DataFrame({pool: {f"{role}:{k}": v for role in ("baseline", "contrastive")
                             for k, v in result["metrics"]["retrieval"][pool][role].items()}
                      for pool in result["metrics"]["retrieval"]}).round(4))
plot_alignment(result["retrieval_raw"], out / "figures" / "alignment_histogram.png");
""")

md("## 7. Paired examples")
code("""
tr = result["translations"].assign(diff=lambda d: d["chrf_contrastive"] - d["chrf_baseline"])
cols = ["source", "english", "reference", "hyp_baseline", "hyp_contrastive", "diff"]
print("contrastive better:"); display(tr.nlargest(3, "diff")[cols])
print("baseline better:");    display(tr.nsmallest(3, "diff")[cols])
both_bad = tr[(tr["chrf_baseline"] < 20) & (tr["chrf_contrastive"] < 20)]
print("both poor:");           display(both_bad.head(3)[cols])
""")

md("""
## 8. Human evaluation (blinded)

`results/human_eval_template.csv` contains predeclared *simple* and *difficult* test items (chosen from the
English side only), with the two systems shuffled into A/B per row. Fluent Nigerian Pidgin speakers score
semantic preservation, grammar and naturalness (1 = severe failure, 3 = understandable with significant
issues, 5 = strong) before opening the key file. Scores are never generated here.
""")
code("""
pd.read_csv(out / "results" / "human_eval_template.csv").head()
""")
code("""
# After scoring: set the path to the completed sheet and run this cell.
SCORED = out / "results" / "human_eval_scored.csv"
if SCORED.exists():
    s = pd.read_csv(SCORED); key = pd.read_csv(out / "results" / "human_eval_KEY_do_not_open_before_scoring.csv")
    s = s.merge(key, on="item_id", suffixes=("", "_sys"))
    long = []
    for slot in ("A", "B"):
        for crit in ("semantic_1to5", "grammar_1to5", "naturalness_1to5"):
            long.append(pd.DataFrame({"system": s[f"system_{slot}_sys"], "difficulty": s["difficulty"],
                                      "criterion": crit, "score": s[f"{slot}_{crit}"]}))
    display(pd.concat(long).groupby(["criterion", "difficulty", "system"])["score"].agg(["mean", "count"]).unstack())
else:
    print("no scored sheet yet:", SCORED)
""")

md("""
## 9. Inference demo: English (e.g. LLM output) → Pidgin

A demonstration only. Text outside the news and spoken training domains is **unverified generalisation**, and
sentence-by-sentence translation loses cross-sentence context.
""")
code("""
example_llm_output = ("Drink plenty of water when the weather is hot. "
                      "If you feel dizzy, sit down in a cool place and rest. "
                      "The clinic opens at eight o'clock tomorrow morning.")
for role, run in {"baseline": "baseline", "contrastive": best_cl}.items():
    m, t = load_best(out / run, device)
    print(f"{role:>12}:", translate_to_pidgin(example_llm_output, m, t, cfg.source_prefix, device, precision,
                                              num_beams=cfg.num_beams, max_new_tokens=cfg.max_new_tokens))
    del m; free_memory()
""")

md("## 10. Integrity checks and reproducibility")
code("""
ic = integrity_checks(cfg, splits, histories, result)
save_json(ic, out / "results" / "integrity_checks.json")
print(json.dumps(ic, indent=1))
print("outputs in", out.resolve())
if not cfg.is_research_run:
    print(f"\\n{cfg.run_label.upper()} RUN COMPLETE - numbers above are not research results.")
""")
md("""
## 11. Limitations (fill in after the full run)

- Single seed: the bootstrap CIs do not include training variance. Re-run with 2–3 seeds before claiming an effect.
- Mismatched train/eval: about two thirds of training is MTurk data, which appears in neither validation nor test.
- In-batch negatives: the contrastive signal depends on batch size; duplicate texts are masked but paraphrases are not.
- Reference quality: Naija English glosses of spoken utterances, and freer MTurk translations, penalise valid outputs.
- Retrieval is directly optimised by the contrastive model, so better R@1 alone is expected; the open question is whether chrF++ follows.
- Licences: MAFAND is CC BY-NC 4.0, Naija-NSC is CC BY-SA 4.0, and the MTurk licence is unconfirmed. Don't redistribute the merged corpus.

**Reproduce:** `python scripts/run_experiment.py` (or this notebook) with the saved `config.json`. Data files are
pinned by commit SHA and checked against expected row counts.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
})
path = Path(__file__).resolve().parents[1] / "notebooks" / "naijattention.ipynb"
path.parent.mkdir(exist_ok=True)
nbf.write(nb, path)
print("wrote", path)
