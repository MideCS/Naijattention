# Implementation plan: Contrastive learning for English-to-Nigerian-Pidgin translation

**Project:** Evaluating Contrastive Learning for English to Nigerian Pidgin Translation  
**Deliverable:** One reproducible Jupyter notebook that loads and audits three parallel corpora (MAFAND-MT, Naija Treebank, and the specified MTurk CSV), fine-tunes two comparable mT5-small models, evaluates translations and sentence alignment, and demonstrates English-to-Pidgin inference.  
**Core research question:** Holding the training data, architecture, budget, and decoding settings fixed, does adding a cross-lingual sentence-level contrastive objective to translation cross-entropy improve translation quality and/or encoder embedding alignment?

## Implementation status (2026-11-16)

Code: https://github.com/MideCS/Naijattention. It lives in `src/naijattention/` (package), `notebooks/naijattention.ipynb`
(Colab-ready) and `scripts/run_experiment.py` (CLI, `--smoke` / `--pilot` / full, `--set key=value` overrides).
Every data count in §3A was reproduced exactly at the pinned commits (35,450 raw train -> 310 leakage removals ->
79+2+1 duplicate removals -> 35,058 train: MAFAND 4,787, MTurk 23,381, Naija 6,890; validation 2,474; test 2,536;
published dev/test overlap 17/20/14).

**Verification done so far**

| Stage | Where | Outcome |
|---|---|---|
| Unit tests (`tests/test_core.py`, 8 tests) | local | pass |
| Static checks (pooling, InfoNCE, retrieval gate, loss sanity, batch check) | runs before every training | pass |
| Tokenizer/collation of all 40,068 rows | local | pass; saved tokenizer encodes identically to the original |
| Smoke run (64 rows, 6 updates) | local CLI + notebook | pass, all integrity checks true |
| Pilot, batch 8, greedy decoding | local RTX 3060 6 GB | pass; 8/8 integrity checks; 0 empty outputs |
| Pilot, batch 32, beam 4 (the real settings) | Colab A100 | pass; val chrF++ 51.93 baseline vs 52.02 contrastive; InfoNCE falls from ln(32) ~ 3.47 to ~0.3 |

Pilot numbers (1 epoch, 300 updates, one seed) only show the pipeline works and the learning rate is sane;
they are **not** research results. The 0.09 chrF++ validation gap is well within seed noise.

**Running it**

- Colab: open `notebooks/naijattention.ipynb` from GitHub, set `RUN_MODE` (`"smoke"`, `"pilot"` default, `"full"`),
  run all. Outputs go to Google Drive (`MyDrive/naijattention/outputs/<mode>`), checkpoints ~1 GB per model.
- Run-level resume: a run whose `train_history.json` exists with identical settings is reused, not retrained;
  an interrupted run restarts from scratch; a finished run with *different* settings raises an error unless
  `overwrite=True`. Mid-epoch resume is not implemented, so a Colab disconnect costs at most the run in progress.
- Full run: ~1,095 updates/epoch x 10 epochs per model (~36x the pilot's 300 updates) plus 10 validation passes
  per model. Estimate wall-clock as pilot training minutes x 36 + validation time.
- Before claiming an effect, repeat both models with 2-3 seeds (`--set seed=...`, separate `output_dir`).
- GPU memory (mT5-small, bf16, max length 128): batch 8 ~5.2 GB, batch 16 ~8.4 GB, batch 32 ~15 GB. A T4 has
  no fast bf16, so it runs FP32 and needs `gradient_checkpointing=True` or batch 16 x accumulation 2 (fewer negatives).

**Engineering additions not in the original plan:** progress logging with ETA every `log_every` updates and
validation timing; explicit GPU memory release between runs (a Windows access-violation crash when loading the
second model was traced to memory retained by the pre-training loss check and fixed); strict retrieval rank
consistent with argmax (ties broken toward the lower index) alongside relaxed R@1 and the reverse direction;
paired bootstrap for chrF++ and retrieval deltas; blinded A/B human-evaluation sheet with a separate key file;
a training-health table (initial/final losses, best epoch, peak memory) and 8 end-of-run integrity checks.

**Deviations from the original plan, and why:**

| Item | Original | Now | Reason |
|---|---|---|---|
| Optimizer / LR | AdamW 3e-5, 3 epochs | Adafactor 1e-3, 10 epochs | 3e-5 barely moves mT5 in ~2k updates; Adafactor 1e-3 is the standard mT5 recipe and uses far less memory |
| Batch | 8 x accum 4 | 32 per step | InfoNCE negatives come only from the per-step batch; accumulation adds none |
| Precision | BF16, else FP16 | BF16 on Ampere+, else FP32 | mT5 overflows in FP16; T4 "bf16 support" is emulated and slow |
| English pooling | prefixed sentence | prefix tokens excluded from pooling | the identical task prefix otherwise adds a shared component to every English embedding |
| False negatives | optional ablation | same-text rows in a batch masked by default | Naija has many repeated short utterances |
| Subset | 25K default | all ~35K by default (`train_subset_size`) | more data with a fixed test set; the 25K option is still available |
| Selection | best val chrF++ | mean of MAFAND-dev and Naija-dev chrF++ | predeclared per §8.1 |
| Domain tag for MTurk | `bbc_news` | `unknown` | MTurk sentence provenance is unverified |
| Nested 5K/10K splits | listed | dropped | no experiment used them |

## 0. Required design decisions

Follow the proposal's two-model comparison exactly:

| Experiment | Initialization | Training objective | Inference |
| --- | --- | --- | --- |
| Baseline | `google/mt5-small` | Translation cross-entropy only | Encoder-decoder English -> Pidgin |
| Contrastive | Fresh `google/mt5-small` with identical seed | Translation cross-entropy + lambda * symmetric InfoNCE | Same encoder-decoder English -> Pidgin |

Both models use **the same pretrained encoder weights shared for English and Pidgin encoding** in the contrastive branch. Do **not** introduce separate language-specific encoders or decoders: mT5 already has one encoder and one decoder. Pidgin text is passed through the encoder *only* for the contrastive loss; at translation inference, the decoder generates Pidgin from an encoded English input. No contrastive branch is needed at inference. Do not inadvertently train the baseline from the checkpoint of the contrastive model, or vice versa.

Make these explicit implementation defaults, adjustable in one config cell:

```python
CFG = {
    "seed": 42,
    "datasets": ["mafand_en_pcm", "naija_ud_nsc", "mturk_pidgin"],
    "train_subset_size": None,  # None = all ~35K cleaned rows; 25000 for the 25K experiment
    "checkpoint": "google/mt5-small",
    "source_prefix": "translate English to Nigerian Pidgin: ",
    "max_source_length": 128,
    "max_target_length": 128,
    "optimizer": "adafactor",
    "batch_size": 32,           # per step = in-batch InfoNCE candidates
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "epochs": 10,
    "warmup_ratio": 0.05,
    "lambda_cl": 0.1,
    "temperature": 0.07,
    "num_beams": 4,
    "max_new_tokens": 128,
    "retrieval_chunk_size": 256,
    "smoke_test": False,
}
```

These are **starting hyperparameters**, not previously measured optima. Use one consistent set of generation parameters throughout both experiments. If memory is constrained, reduce microbatch size and increase accumulation; document that a smaller microbatch reduces the number of in-batch contrastive negatives and is *not* perfectly equivalent to the original contrastive experiment. Prefer a CUDA GPU with ~16-24 GB or more for full-model tuning; probe actual memory rather than assuming a particular machine suffices. BF16 on Ampere or newer GPUs, otherwise FP32. **Never FP16**: mT5 activations overflow. CPU is for smoke tests only.

## 1. Notebook structure and expected artifacts

Use these ordered notebook sections/cells. Every numbered item should become one or more markdown and code cells, with an explanatory markdown cell preceding nontrivial code.

1. Title, research question, methodological diagram, experiment table.
2. Environment install/import/version report and random seed.
3. Configuration and device inspection.
4. Load and verify three datasets in their native JSONL / CoNLL-U / CSV formats (see Section 3A).
5. Data cleaning, benchmark-preserving leakage exclusions, merged training corpus, and saved manifests.
6. Tokenizer, translation examples, data collators, and dataloaders.
7. Masked mean pooling and symmetric contrastive loss unit tests.
8. Translation baseline training.
9. Combined-loss model training and lambda experiments.
10. Learning curves, checkpoint selection and run summary.
11. Final test-set translation and chrF++ comparison.
12. Encoder retrieval evaluation and similarity diagnostics.
13. Paired examples and manually scored simple/difficult cases.
14. Optional inference demonstration on external English text.
15. Results table, limitations, and automatic integrity checks.

Save outputs beneath a configurable relative directory such as `outputs/pidgin_mt5/`:

```text
outputs/pidgin_mt5/
  config.json
  environment.json
  data_audit.json
  splits/{train,val_mafand,val_naija,test_mafand,test_naija}.jsonl
    splits/{overlap_audit,source_counts}.csv
  baseline/{checkpoint files,train_history.json}
  contrastive_lam_0p1/{checkpoint files,train_history.json}
  optional_lambdas/{...}
  results/{translations.csv,metrics.json,retrieval.json,human_eval_template.csv}
  figures/{learning_curves.png,alignment_histogram.png,examples.png}
```

Write checkpoint directories using `save_pretrained` (including tokenizer); do not save only a raw PyTorch state dict. If models are too large for notebook attachment, keep code and metadata in the notebook and instruct the runner where to save large checkpoints. The notebook must still run top-to-bottom when given sufficient compute.

## 2. Environment and reproducibility

**Cell tasks**

1. Install/import `torch`, `transformers`, `datasets`, `accelerate`, `sentencepiece`, `sacrebleu`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, and `tqdm`. `evaluate` is optional; use `sacrebleu.CHRF` directly to make chrF++ parameters explicit.
2. Print Python, PyTorch, Transformers, Datasets and SacreBLEU versions; GPU name, memory, CUDA availability, BF16 support; record them to `environment.json`. Pin the tested package versions in a final requirements cell once the notebook works; Hugging Face Trainer APIs change between versions.
3. Set Python `random`, NumPy, and PyTorch seeds (including CUDA). Where practical enable deterministic algorithms and document any non-deterministic GPU behavior.
4. Initialize output folders and a checkpoint-resume flag; never silently overwrite model checkpoints when resuming.
5. Provide `SMOKE_TEST` configuration: e.g. 32 train / 16 validation / 16 test examples, one training step or at most one abbreviated epoch, shorter sequences; clearly label smoke results as **not valid research results**.

**Gate:** Test imports; save/reload the config; verify that GPU selection and seed are printed.

## 3. Dataset loading, validation, and bias audit

**Primary datasets (updated and verified):** MAFAND-MT `en-pcm`, UD Naija-NSC Treebank, and the requested `MTurk_Pidgin.csv`. Section 3A is the **authoritative source loading and split policy**; it supersedes the original OPUS-MT560-only 80/10/10 setup. The original OPUS dataset is now an *optional separate ablation*; do not silently append it to the three-source corpus. Do not create a random 80/10/10 split over the merged full data, since this would destroy MAFAND/Naija published benchmarks.

**Cell tasks**

1. Run the three source loaders in Section 3A; verify their column mappings, source split labels, and raw row counts.
2. Display 15-25 reproducibly sampled source/target pairs, including shortest/longest examples; report word and mT5-token length distributions, blank strings, duplicate exact pairs, repeated English sentences, repeated Pidgin sentences, and source-target identical strings.
3. **Qualitative audit:** MTurk English is a human translation of Pidgin and may be freer than a literal gloss; Naija English glosses spoken utterances. Randomly review ~50 pairs per source mix and assign rough flags: `accurate_translation`, `paraphrase`, `potential_misalignment`, `other`. Tabulate counts and save a few anonymized/safe examples. This quality review is descriptive, not a new labeled training dataset.
4. Strip leading/trailing whitespace, standardize Unicode (NFC), collapse multiple internal whitespace runs while preserving punctuation/case, remove empty/null rows. **Do not** lowercase, delete punctuation, remove local spellings, or aggressively normalize Pidgin; chrF++ should assess real written variation.
5. Normalize pair keys for duplicate detection (Unicode normalization, whitespace normalization; optionally casefold only for duplication checks). Remove exact duplicate pairs. Inspect conflicts where one identical English input has multiple Pidgin targets; retain valid alternatives if appropriate but keep all occurrences of that English text in the same split.
6. **Preserve** the source-provided dev/test assignments; do not independently shuffle MAFAND/Naija. Deduplicate and exclude training rows overlapping *either* side of *either* held-out source. Optionally group near-duplicate texts for additional exclusion, but record how many are removed. The previously proposed union-find connected components approach remains useful for any *additional random split of training data only*, and for deduplication, not for reassigning published benchmark test rows.
7. Assert **train-vs-holdout** normalized English overlap and Pidgin overlap are zero. Published **validation-vs-test** splits contain some overlap (measured in Section 3A); report it, keep their published identities intact, and optionally give a separate decontaminated aggregate metric. Save original row IDs, cleaned texts, source, split, duplicate-removal reason, and source commit hash. Keep test sealed until final evaluation.

**Gate:** No empty pairs in the final corpus, no exact train-to-holdout source or target overlap, reproducible row counts, realistic samples displayed. Preserve and separately report published dev/test overlaps rather than claiming zero overlap everywhere. If the dataset is materially misaligned, annotate the limitation and optionally prepare a *separate* clean subset analysis; do not silently change the primary test set after seeing model outcomes.

## 3A. Tested source formats, exact loaders, and merged-dataset contract (2026-11-13)

These instructions are based on small **real file loading tests**, not inferred schemas. The tests fetched the named GitHub files, parsed the complete MTurk CSV, the three MAFAND JSONL split files, and the three UD Naija CoNLL-U split files. Recheck source revisions on future runs; upstream files can change.

### Source A: MAFAND-MT (`en-pcm`)

- Repository: `https://github.com/masakhane-io/lafand-mt`
- Raw file pattern: `https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/en-pcm/{train,dev,test}.json`
- **Actual format: JSON Lines despite the `.json` extension**. Each line is a separate JSON object; `json.load(open(file))` raises `JSONDecodeError: Extra data`. Use `pd.read_json(path, lines=True)` (nested dict column) or `[json.loads(line) for line in file if line.strip()]`.
- Verified line structure: `{"translation":{"en":"Ekiti police nab 2 serial burglars","pcm":"Ekiti Police don catch 2 big tif."}}`.
- Field mapping: `english = row['translation']['en']`, `pidgin = row['translation']['pcm']`; if loading via a `Dataset` wrapper, inspect the nested `translation` field before indexing.
- Observed counts: **4,790 train / 1,484 dev / 1,564 test = 7,838**. A table in the project dataset card sometimes reports 1,574 test, but the actual raw `test.json` fetched for this test contains **1,564** lines. Record the source SHA/revision and trust the actual file count for your run.
- Observed exact duplicate pairs: 2 in train, 2 in dev, 0 in test (before normalizing). Both languages non-null in tested files.
- **Important compatibility note:** The Hugging Face dataset card currently reports that its old `mafand.py` loading script is unsupported by modern `datasets`; prefer loading the repository's three direct JSONL files using `load_dataset('json', data_files={...})` or Python/Pandas. Do not build a notebook that relies solely on `load_dataset('masakhane/mafand', 'en-pcm')` succeeding.
- License declared in dataset card: **CC BY-NC 4.0**. Attribute and preserve restrictions.

### Source B: Naija Treebank / DiscoNaija

- Use the original **UD Naija-NSC** parallel utterances for translation; DiscoNaija adds discourse annotations to this same underlying material, so **do not append DiscoNaija as a second 9K dataset**. That would double-count the same data.
- Repository: `https://github.com/UniversalDependencies/UD_Naija-NSC`
- Raw file pattern: `https://raw.githubusercontent.com/UniversalDependencies/UD_Naija-NSC/master/pcm_nsc-ud-{train,dev,test}.conllu`
- **Actual format: CoNLL-U**. Each example is a blank-line-separated block of `# key = value` metadata followed by token annotation rows. Only metadata is needed for translation; do not build input text by joining the tabular token rows.
- Observed metadata keys include `sent_id`, `text`, `text_en`, `text_orthographic`, `speaker_id`, `sound_url` and speaker demographics. **The actual file uses `text_orthographic`, not `text_ortho`** (some documentation calls this `text_ortho`).
- Field mapping: `english = meta['text_en']`, `pidgin = meta['text_orthographic']`, `row_id = meta['sent_id']`, optional grouping key `speaker_id` / recording prefix in `sent_id`.
- **Do not use the raw `text` field as the Pidgin target**: it contains transcription markers such as `//`, `<`, `#` and `{ ... |c ... }`. The orthographic field is already the readable normalized text.
- Tested counts: **7,279 train / 990 dev / 972 test = 9,241** utterance blocks. The public project description says 9,242, a minor version/count discrepancy. Exactly **one** train block lacked `text_orthographic`, leaving **7,278 usable train rows** when dropped (ID `IBA_34_News-Report-By-Samuel_MG__13`). Optionally derive a clean target for that one row only after manual inspection; default to dropping it. Dev/test had no missing `text_orthographic` or `text_en` in the test.
- Use `conllu.parse_incr(file)` if installed, accessing `TokenList.metadata`. Alternatively use the tested minimal metadata parser below. Handle final file block even without a terminal blank line.
- License: **CC BY-SA 4.0**. Preserve source attribution and evaluate whether share-alike terms affect redistribution of a merged corpus; do not automatically publish a merged dataset.

### Source C: Requested MTurk CSV (not the 116K synthetic dataset)

- Exact user-selected file: `https://github.com/Tommy0201/LowResource-Paper-Dataset/blob/main/Nigerian-Pidgin/MTurk_Pidgin.csv`
- Direct download URL: `https://raw.githubusercontent.com/Tommy0201/LowResource-Paper-Dataset/main/Nigerian-Pidgin/MTurk_Pidgin.csv`
- **Actual format: UTF-8 CSV**. Load with `pd.read_csv(url, dtype='string')` (or download locally then read). Escape commas/quotes are handled by Pandas' CSV parser. Do not split lines on commas manually.
- **Actual columns:** `pidgin_source`, `eng_translation` (NOT `english`, `pidgin`, or `eng`). Reverse these for English-to-Pidgin: `english = row['eng_translation']`; `pidgin = row['pidgin_source']`.
- Observed counts: **23,382 rows**, **0 null values**, **0 blank strings** in either column, **1 exact duplicate pair**, **2 repeated Pidgin source strings**, **4 repeated English translations**, before text normalization. This is the specific MTurk file; do not confuse it with the much larger `Tommy0201/pidgin-to-english` Hugging Face dataset.
- Original translation direction was **Pidgin -> English**; reverse supervision to English -> Pidgin. A human-translated English sentence may be freer than a literal gloss, so perform a sample alignment audit before contrastive training. The repository provenance should be cited and its data license confirmed before redistribution.
- There are no pre-existing validation/test flags in this CSV; put it **in the training pool only**, not the held-out tests. The model's validation and test sets come from MAFAND and Naija.

### Minimal, directly tested extraction pattern

The following standard-library/Pandas workflow matches the file formats observed. A production notebook may wrap this in helper functions and use `datasets.Dataset.from_pandas` afterward.

```python
from pathlib import Path
import json
import pandas as pd

MAFAND_BASE = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/en-pcm"
NAIJA_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_Naija-NSC/master"
MTURK_URL = "https://raw.githubusercontent.com/Tommy0201/LowResource-Paper-Dataset/main/Nigerian-Pidgin/MTurk_Pidgin.csv"

def parse_conllu_metadata(text):
    # Reads comments only; token rows are not needed for translation.
    records, block = [], []
    for line in text.splitlines() + [""]:
        if line.strip():
            block.append(line)
        elif block:
            meta = {}
            for item in block:
                if item.startswith("# ") and " = " in item:
                    k, v = item[2:].split(" = ", 1)
                    meta[k] = v
            if "sent_id" in meta:
                records.append(meta)
            block = []
    return records

def load_mafand(split):
    # 'dev' is renamed to 'validation' in our harmonized schema.
    url = f"{MAFAND_BASE}/{split}.json"
    raw = pd.read_json(url, lines=True)
    records = raw['translation'].tolist()
    return pd.DataFrame({
        'english': [r['en'] for r in records],
        'pidgin': [r['pcm'] for r in records],
        'source': 'mafand',
        'source_split': split,
        'source_id': [f'mafand-{split}-{i}' for i in range(len(raw))],
    })

def load_naija(split):
    from urllib.request import urlopen
    url = f"{NAIJA_BASE}/pcm_nsc-ud-{split}.conllu"
    with urlopen(url) as response:
        blocks = parse_conllu_metadata(response.read().decode('utf-8'))
    rows = []
    for r in blocks:
        if not r.get('text_en') or not r.get('text_orthographic'):
            continue
        rows.append({
            'english': r['text_en'], 'pidgin': r['text_orthographic'],
            'source': 'naija', 'source_split': split,
            'source_id': r['sent_id'], 'speaker_id': r.get('speaker_id'),
            'recording_id': r['sent_id'].rsplit('__', 1)[0],
        })
    return pd.DataFrame(rows)

def load_mturk():
    d = pd.read_csv(MTURK_URL, dtype='string')
    assert {'pidgin_source', 'eng_translation'} <= set(d.columns)
    return pd.DataFrame({
        'english': d.eng_translation,
        'pidgin': d.pidgin_source,
        'source': 'mturk',
        'source_split': 'train',
        'source_id': [f'mturk-{i}' for i in range(len(d))],
    })

raw_train = pd.concat([
    load_mafand('train'), load_naija('train'), load_mturk()
], ignore_index=True)
raw_val = pd.concat([
    load_mafand('dev'), load_naija('dev')
], ignore_index=True)
raw_test = pd.concat([
    load_mafand('test'), load_naija('test')
], ignore_index=True)

assert len(raw_train) == 35450  # 4790 + 7278 + 23382, at checked revisions
assert len(raw_val) == 2474    # 1484 + 990
assert len(raw_test) == 2536   # 1564 + 972
```

**Production hardening:** Network fetches can fail; cache files under `data/raw/`, pin SHA commits instead of `main`/`master`, include retries/timeouts, hash raw bytes, and log the exact version. The snippet above prioritizes verified schema clarity; adjust counts only after explicitly checking the pinned versions.

### Harmonized row schema

Every row should eventually contain exactly these primary columns (additional metadata is fine):

| Column | Type | Meaning |
| --- | --- | --- |
| `english` | string | Raw English source sentence, cleaned conservatively |
| `pidgin` | string | Gold Pidgin translation; training label and contrastive positive |
| `source` | enum | `mafand`, `naija`, `mturk` |
| `source_split` | enum | Original `train`, `dev`, `test`; do not erase |
| `source_id` | string | Stable unique provenance row ID |
| `split` | enum | Harmonized `train`, `validation`, or `test` |
| `domain` | string | `news`, `spoken`, or `bbc_news` |
| `eng_norm`, `pcm_norm` | string | Dedup-only normalized text, not tokenizer inputs |
| `speaker_id`, `recording_id` | optional string | Naija provenance; useful for grouping and leakage analysis |

Map `dev -> validation`; preserve `source_split` unchanged. Avoid using positional row indices alone after filtering; IDs must remain stable. A model sample is always `{'english': '...', 'pidgin': '...'}` irrespective of native format, with the tokenizer consuming `english` and cross-entropy labels coming from `pidgin`.

### Full combined dataset construction: exact ordering

1. **Load and tag separately.** Materialize all **seven** source files (3 MAFAND + 3 Naija + 1 MTurk CSV); assign source/split/IDs before concatenation. Keep a per-source provenance table and immutable original-file hash. Do not duplicate DiscoNaija as if it were independent of Naija Treebank.
2. **Validate + minimally clean.** Strip whitespace, Unicode NFC-normalize, replace repeated whitespace; drop null/empty pairs and the one Naija row without `text_orthographic`. Preserve case, punctuation, and Pidgin orthography. Flag odd length ratios and potential semantic mismatch for manual review, but do not mechanically discard large numbers of valid code-switched examples.
3. **Reserve benchmarks before constructing training.** `validation = mafand/dev + naija/dev`, `test = mafand/test + naija/test`, with source labels retained. Do not reserve random MTurk rows as test unless you clearly define a separate *additional* benchmark (the primary evaluation is source-defined).
4. **Make normalized comparison keys.** Define `norm(s)` as Unicode NFKC + casefold + trim + collapse whitespace for duplicate detection. Preserve original NFC strings for training and chrF++. Record `pair_key=(eng_norm,pcm_norm)`.
5. **Block train-to-evaluation leakage.** Build combined held-out sets of `eng_norm` and `pcm_norm` across **both validation and test**, and remove *any training row whose normalized English OR normalized Pidgin string appears in either*. Also remove exact normalized pair duplicates within training, retaining the highest-priority source row (`naija` human-spoken > `mafand` curated > `mturk` when conflicts are truly identical); always preserve a duplicate-removal audit. Optional near-duplicate MinHash/fuzzy filtering is a separate documented pass, not a silent change.
6. **Important observed overlap results.** On the tested raw versions: 35,450 initial training rows; 5,010 combined validation/test rows. Exactly **310 training rows** matched at least one English or Pidgin normalized key in the held-out pool. After excluding these and deduplicating remaining *pair keys*, **35,058 training rows** remained. These are measured outputs of the simple normalization logic, not guaranteed counts after other quality filters or version changes.
7. **Do not silently reassign held-out examples.** Our probe also found overlaps between the *published* dev and test sets themselves: 17 dev rows matched a test English key, 20 matched a test Pidgin key, and 14 matched an identical pair. Keep original source benchmarks intact for comparability; document these cases. For a strictly disjoint **secondary** aggregate analysis, remove conflicting validation rows according to a rule established before model evaluation, or report per-source metrics separately. Never train on either held-out partition.
8. **Training-set size.** The primary experiment uses **all 35,058 cleaned training rows** (`train_subset_size=None`). The optional 25K experiment (`train_subset_size=25000`) deterministically samples from the cleaned training pool only, keeping all `naija` and `mafand` rows and drawing the remainder from `mturk` (seed 42; result: MAFAND 4,787 / MTurk 13,323 / Naija 6,890). Validation and test are never subsampled. The nested 5K/10K splits were dropped.
9. **Check recording leakage within Naija.** `speaker_id` can recur across splits; evaluate whether `recording_id` or speaker content crosses. Do not automatically move established Naija test rows, but report source/speaker split overlap, and provide an optional stricter speaker-disjoint evaluation if the original test permits it.
10. **Convert and save.** Save harmonized source-specific data as `data/processed/{source}_{split}.parquet` and the selected merged `train_25k.parquet`, `validation.parquet`, `test.parquet`, plus two per-source holdout variants. Save a metadata JSON with raw/source counts, exclusions, duplicate counts, hashes, seed and code revision. `datasets.Dataset.from_pandas(df, preserve_index=False)` or `DatasetDict` can be used downstream; the custom collator must consume the renamed `english` and `pidgin` columns rather than old `eng`/`pcm` names.
11. **Assert invariants.** Training length equals the chosen size; train English keys are disjoint from held-out English keys; train Pidgin keys disjoint from held-out Pidgin keys; no duplicate normalized pair keys remain in training; all required strings are nonempty; all provenance IDs unique within source; all labels are legitimate Pidgin strings. Check no evaluation pair was placed into training, and print the precise source composition of each split.

**Caveat for contrastive retrieval:** The pooled MAFAND+Naija test set contains repeated generic utterances (e.g. short greetings); strict index-based Recall@1 treats an equivalent alternative sentence as a negative. Report per-source Recall@1, inspect duplicates, and optionally compute an equivalence-aware metric that accepts any same-normalized-target candidate. Keep index-based R@1 as the comparable primary metric and **never count the duplicate pairs as extra positive instances in training**.

## 4. Tokenization and batches

1. Load `AutoTokenizer.from_pretrained(CFG['checkpoint'])` and `AutoModelForSeq2SeqLM.from_pretrained(CFG['checkpoint'])`. Prefer current documented API, e.g. `tokenizer(text_target=...)` and `DataCollatorForSeq2Seq`.
2. For each example construct the encoder source `CFG['source_prefix'] + eng`, and the decoder target `pcm`. Use exactly this prefix during both training and inference. Keep the **raw** English and Pidgin strings separately for the contrastive branch.
3. Tokenize English translation source to `max_source_length` and Pidgin decoder labels to `max_target_length`, `truncation=True`. For comparison, report what fraction of examples is truncated and show original lengths; optionally raise bounds if >5% truncated and compute allows.
4. Create Pidgin **encoder** tokenization separately for contrastive alignment. **Do not add the English-to-Pidgin translation prefix to the Pidgin encoder view.** Choose whether the English encoder contrastive view uses the same prefixed source as translation or the plain English sentence; default to *prefixed* English so alignment concerns the exact English features consumed by translation, but document this choice and use it identically across experiments and retrieval.
5. Implement a custom batch collation function combining `DataCollatorForSeq2Seq` for English translation inputs/labels with dynamically padded Pidgin encoder `pcm_input_ids`/`pcm_attention_mask`. Keep the two sets of `input_ids` distinct; ensure label padding is `-100`, attention mask padding is 0, and all tensors are moved to device before forward passes.
6. Print one real minibatch's tensor shapes and decoded source/target texts. Verify labels with `-100` replaced by pad ID decode back to target text.

**Gate:** For any batch size B, `input_ids`, `attention_mask`, `labels`, `pcm_input_ids`, `pcm_attention_mask` all have first dimension B; padding ignored in CE and pooling. `model.generate` runs on a 2-example batch before training.

## 5. Sentence embeddings and contrastive objective

We need one sentence vector per English input and one per Pidgin reference, using **the same shared mT5 encoder**. The translation model's encoder output should be reused for English to avoid computing that encoder twice.

### 5.1 Masked mean pooling

Given encoder hidden states `H` of shape `[B, T, d]` and attention mask `M` of shape `[B, T]`, compute:

`sentence = sum_t(H[:,t,:] * M[:,t,None]) / clamp_min(sum_t M[:,t], 1)`.

Normalize each sentence vector to unit L2 length using `F.normalize(..., p=2, dim=-1)`. No random projection head by default: embeddings used in training and retrieval must represent the **same space**. If an experimental projection head is added, include it consistently at evaluation and report it as a distinct architecture.

### 5.2 Symmetric InfoNCE

For a batch of B parallel examples, English embeddings `E` and Pidgin embeddings `P` are `[B,d]`. Create `S = E @ P.T / temperature`, an `[B,B]` logits matrix. Pair `(i,i)` is a positive pair, other columns and rows are in-batch negatives:

`L_en_to_pcm = cross_entropy(S, arange(B))`  
`L_pcm_to_en = cross_entropy(S.T, arange(B))`  
`L_CL = 0.5 * (L_en_to_pcm + L_pcm_to_en)`.

`L_total = L_MT + lambda_cl * L_CL`.

Implement with PyTorch `torch.nn.functional.cross_entropy`; **do not** detach embeddings or compute them under `torch.no_grad()` during contrastive training. The logits matrix must be computed on the same device and in a numerically stable dtype (cast to float32 for similarity/cross-entropy if training mixed precision). Batch size 1 is uninformative for InfoNCE: use `drop_last=True` for contrastive training if needed. In-batch duplicate or semantically equivalent sentences are potential *false negatives*; the split/data audit should quantify duplicates, and any masking variant should be marked as an optional ablation.

### 5.3 Combined forward pass (pseudocode)

```python
def compute_losses(model, batch, lambda_cl, tau):
    # Encoder output used for English-to-Pidgin translation.
    eng_outputs = model.get_encoder()(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        return_dict=True,
    )
    mt_outputs = model(
        attention_mask=batch['attention_mask'],
        encoder_outputs=eng_outputs,
        labels=batch['labels'],
        return_dict=True,
    )
    mt_loss = mt_outputs.loss

    pcm_outputs = model.get_encoder()(
        input_ids=batch['pcm_input_ids'],
        attention_mask=batch['pcm_attention_mask'],
        return_dict=True,
    )
    eng_emb = masked_mean_pool(eng_outputs.last_hidden_state,
                               batch['attention_mask'])
    pcm_emb = masked_mean_pool(pcm_outputs.last_hidden_state,
                               batch['pcm_attention_mask'])
    eng_emb = torch.nn.functional.normalize(eng_emb.float(), dim=-1)
    pcm_emb = torch.nn.functional.normalize(pcm_emb.float(), dim=-1)
    scores = eng_emb @ pcm_emb.T / tau
    targets = torch.arange(scores.size(0), device=scores.device)
    cl_loss = 0.5 * (
        F.cross_entropy(scores, targets) +
        F.cross_entropy(scores.T, targets)
    )
    return mt_loss + lambda_cl * cl_loss, mt_loss, cl_loss
```

If using `encoder_outputs` fails with a particular installed Transformers version, wrap English encoder hidden states in the correct documented `BaseModelOutput(last_hidden_state=...)` structure, and verify equivalence to an ordinary translation forward pass with contrastive weight 0. Do not turn off gradients for the precomputed encoder output.

**Unit tests (must execute before training):**

- Pooling of a manually constructed padded tensor equals hand-computed means; adding pad tokens changes neither embedding nor cosine similarity.
- `S` is `[B,B]`, `L_CL` finite and nonnegative; `B=1` is explicitly blocked for contrastive training.
- Construct `E=P=identity(B)` with small tau: contrastive loss is low; shuffle P without changing positive labels: loss increases.
- Backpropagating combined loss produces nonzero finite gradients on encoder weights **and** decoder weights; no detach/no_grad mistakes.
- With `lambda_cl=0`, full combined forward MT loss matches baseline translation loss on the same model and batch (up to float tolerance).

## 6. Baseline fine-tuning

1. Create a *fresh* `google/mt5-small` with the fixed seed. Use the normal translation forward pass (`model(..., labels=...)`); loss is standard token-level cross-entropy, with pad targets masked by `-100`.
2. Use AdamW at proposed defaults, linear decay with warmup, gradient clipping 1.0, gradient accumulation, mixed precision where supported, and training for three epochs initially. Correctly step scheduler only on optimizer updates, not raw minibatches.
3. At each epoch compute mean training loss, validation translation loss, and validation chrF++ using deterministic beam search. Evaluation must use `model.eval()` and `torch.no_grad()`; return to `model.train()` afterward.
4. Save the best checkpoint by validation chrF++ (not test), plus epoch logs and configuration. If validation metric is noisy, report the checkpoint-selection policy before running. For strict fairness, the contrastive model must use the same checkpoint-selection policy.
5. Record training time, number of optimizer steps, GPU peak memory and total trainable parameters.

**Gate:** Training loss decreases on a tiny subset, validation generation returns nonempty Pidgin outputs, saved tokenizer/model can be reloaded and generate the same output under deterministic settings.

## 7. Contrastive fine-tuning

1. Initialize **another** fresh mT5-small using the same seed/pretrained weights, train/val splits, optimizer, learning-rate schedule, updates/epochs, accumulation, and translation decoding settings as baseline.
2. Use the custom dual-view collator and `compute_losses`. Train with `lambda_cl=0.1`, `temperature=0.07` initially. Track *each of* `L_total`, `L_MT`, and `L_CL` by step and epoch; avoid comparing unnormalized sums if they are averaged differently.
3. Validate MT loss and chrF++ on the same validation subset as baseline. Also optionally monitor validation retrieval, **but do not tune on test**. Save best model by validation chrF++.
4. Run at least two lambda values if compute allows: `lambda_cl in {0.05, 0.1, 0.5}`; select lambda by validation chrF++ only. It is acceptable to run only 0.1 if the compute budget is constrained, but state that lambda is not tuned. Temperature sensitivity `{0.05,0.1}` is optional. Each sweep starts from the same original pretrained model, never from another sweep's trained weights.
5. For apples-to-apples comparisons log total optimizer updates and *effective* batch sizes. Keep number of training examples equal across experiments. The contrastive model costs more per step due to the second encoder pass; equalize training **updates**, not wall-clock runtime.

**Gate:** Contrastive gradients are nonzero, model trains without NaN/Inf, both losses are recorded, lambda and tau are saved with checkpoint, and baseline/contrastive runs have matching training budgets.

## 8. Automatic translation evaluation (held-out test only)

1. Reload the selected baseline and contrastive checkpoints (best validation chrF++); freeze both and run `eval()`. Use a predeclared selection metric such as the equally weighted mean of MAFAND-validation chrF++ and Naija-validation chrF++ (state the policy in advance). Report **MAFAND-test** and **Naija-test** independently as primary domain-separated scores; optionally calculate pooled results as secondary results, without hiding cross-domain differences.
2. Generate predictions for every test English sentence with identical tokenizer, prefix and `generate(num_beams=4, do_sample=False, max_new_tokens=128, ...)`. Set min length only if predeclared; beware truncating long targets.
3. Preserve input, reference, prediction for both models, original IDs, **dataset source**, input/output lengths, and whether output is blank. Save a single row-aligned `translations.csv`.
4. Compute **corpus chrF++** with SacreBLEU's `CHRF(word_order=2)` and otherwise default documented parameters; compute once per **source-specific test corpus**, not by averaging sentence-level chrF++ scores. Pass references in the shape accepted by the installed SacreBLEU API. Also report sentence chrF++ for pairwise error analysis, but keep it distinct from corpus score.
5. Report `baseline chrF++`, `contrastive chrF++`, and delta. Compute a **paired bootstrap** 95% confidence interval for the delta by resampling aligned test rows with replacement (e.g. 1,000 replicates), recomputing corpus chrF++ for both on each resample. State if interval includes zero; do not claim significance based only on the raw delta.
6. Optionally report BLEU, mean generated tokens, empty output rate, and sample inference throughput, but chrF++ stays the preregistered primary translation metric.

**Gate:** Same number and ordering of examples for both systems; no test rows fed to training or validation; deterministic repeated generation matches for a few spot-check rows.

## 9. Encoder embedding-alignment evaluation

The proposed metric is **top-1 cross-lingual sentence retrieval**: for each English test sentence, rank the Pidgin **reference sentences** in the held-out test set by cosine similarity and ask whether the highest-ranked Pidgin sentence is the gold matching pair. **Do not** use generated predictions as the retrieval target, and **do not** compare English output of one model with Pidgin encoder output of another model.

1. For each *trained model independently*, tokenize all English test inputs using the same English encoder-view policy as training, and all Pidgin reference texts using the raw Pidgin encoder-view policy. Compute retrieval separately within MAFAND and Naija test pools; a pooled candidate pool can be an additional harder task, but must be labeled accordingly.
2. Pass each language through `model.get_encoder()` with `torch.no_grad()` in deterministic batches; use masked mean pooling and L2 normalization. Ensure no `-100` label IDs get passed to encoder.
3. Produce `E, P` arrays `[N,d]`, ensure row i English and row i Pidgin are gold pair. Compute cosine similarity `E @ P.T`, in GPU chunks or CPU matrix blocks if N is large. Retrieve `argmax` for each English query (do **not** compute only diagonal similarities; the metric requires comparison against every candidate).
4. Compute `Recall@1 = mean(argmax(S[i,:]) == i)`, plus optional Recall@5/10 and reverse Pidgin->English Recall@1. Report chance baseline `1/N` and total N. Because semantically identical candidate translations may exist, explicitly quantify repeated/near-duplicate candidates and optionally compute a relaxed correctness score that accepts alternate exact duplicates; keep strict index-based Recall@1 as primary.
5. For both models, report mean/median matched-pair cosine similarity, mean highest incorrect similarity, and histograms of positive-vs-negative similarities. Example nearest-neighbor tables should include English query, gold Pidgin, top retrieved Pidgin, rank and similarity.
6. Compare per-query retrieval success between models; use paired bootstrap confidence intervals for difference in Recall@1. Plot a small heatmap (e.g. 15x15) with correct match on diagonal, but preserve test ordering.

**Gate:** Identity synthetic embeddings yield 100% Recall@1, shuffled identity yields expected changed recall; all embeddings unit-normalized and finite; identical retrieval candidate set across models; no separate trained retrieval head used only by one model.

## 10. Human evaluation (simple vs difficult)

1. **Before looking at predictions**, define selection criteria on English references. Example simple: short, single-clause, common words; difficult: longer/multiclause, idioms, domain-specific terminology or ambiguous phrasing. Construct a reproducible stratified subset of ~20 simple and ~20 difficult test examples. Do not use the training set for this analysis. Report the domain mix (news vs spoken) of the selected items.
2. Create a blinded `human_eval_template.csv`: example ID, English sentence, reference Pidgin, anonymous `system_A` output, anonymous `system_B` output, difficulty, plus empty scoring columns; randomly shuffle A/B per example and store the decoding key separately.
3. Rubric (each dimension 1-5): **semantic preservation** (meaning of English faithfully conveyed), **grammatical adequacy** (coherent Pidgin syntax), **naturalness** (sounds like contemporary Nigerian Pidgin rather than literal English). Define anchors: 1 = severe failure, 3 = understandable but significant issues, 5 = strong. Include `error_notes` and optional `reference_problem` flag.
4. Prefer two fluent Nigerian Pidgin evaluators, independently scoring outputs without model identities; if only one evaluator is available, report that limitation. Assess inter-rater agreement (e.g. weighted Cohen kappa for ordinal criteria if two raters). Do not invent completed human scores.
5. Reveal system labels only after scoring; report average scores by category and difficulty with example-level paired differences. Show at least three concrete examples: contrastive wins, baseline wins, and both fail. Discuss limitations of references when they paraphrase or add content.

**Gate:** No automatically generated placeholder numbers are represented as human judgments; every numerical human-evaluation claim must be traceable to scored rows.

## 11. Optional LLM-to-Pidgin inference demonstration

This is a **demo**, not part of test metrics. To avoid reliance on paid APIs or credentials, start with a short fixed English paragraph drafted directly in the notebook and label it `example LLM-style output`. Optionally let the user paste real English text generated by an external LLM; do not require an LLM API for notebook completion.

1. Write `translate_to_pidgin(text, model, tokenizer, device, max_new_tokens=128)` that prepends the exact training source prefix, tokenizes, calls deterministic `generate` and decodes with `skip_special_tokens=True`.
2. Print English text, baseline output, contrastive output side by side. Provide a small optional text input widget (`ipywidgets`) if installed; otherwise a plain Python function is sufficient.
3. For paragraphs longer than training examples, split into sentences/chunks, translate one at a time, and recombine. State that chunking may lose cross-sentence context.
4. Label performance outside the news/spoken training domains as **unverified generalization**; do not imply the chrF++ score transfers automatically to LLM outputs.

## 12. Training-loop implementation guidance

Use either (A) a transparent custom PyTorch training loop for **both** models or (B) `Seq2SeqTrainer` for the baseline and a carefully overridden `compute_loss` for the contrastive model. **Prefer A** for clarity: identical optimizer/checkpoint logic and explicit losses are easier to audit.

Recommended custom loop:

```text
for experiment in [baseline, contrastive(s)]:
    seed_all(seed); load fresh mT5 weights
    optimizer = AdamW(model.parameters(), lr=..., weight_decay=...)
    scheduler = linear_schedule(num_training_optimizer_updates, warmup=...)
    for epoch:
        model.train()
        for batch in train_loader:
            move tensors to device
            with autocast_if_supported():
                if baseline: total_loss = model(..., labels=...).loss
                else: total_loss, mt_loss, cl_loss = compute_losses(...)
            backward(total_loss / gradient_accumulation_steps)
            if accumulation boundary OR last batch:
                unscale gradients if using GradScaler
                clip gradients, optimizer.step(), scheduler.step(), zero_grad()
            accumulate *unscaled* logged losses weighted by batch size
        eval: calculate validation CE and generated chrF++
        save if best validation chrF++
    reload best model, record all metadata
```

When the last accumulation group is shorter than the nominal size, adjust loss scaling to avoid underweighting it; make sure no optimizer step is silently skipped. When using AMP, use current supported autocast/GradScaler API for installed PyTorch. Never call `model.generate` under the gradient-enabled training forward pass.

## 13. Suggested final notebook outputs

**Table A — experimental configuration:** checkpoint, data revision, exact split counts, epochs, optimizer steps, train microbatch/accumulation, lambda, tau, model parameter count, selected checkpoint epoch, seed.

**Table B — headline comparisons:**

| Model | Test chrF++ | Delta vs baseline | Test EN->PCM R@1 | Delta vs baseline | Test PCM->EN R@1 | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline MT | computed | — | computed | — | computed | computed |
| MT + contrastive | computed | computed | computed | computed | computed | computed |

**Figures:** (1) train/val MT loss by epoch, plus CL loss for contrastive model; (2) grouped model comparison of chrF++ and Recall@1, including CIs; (3) positive-pair vs negative-pair cosine distribution; (4) at most one retrieval heatmap and short qualitative table.

**Discussion prompts:**

- Did contrastive learning improve embedding alignment? Did the gain translate into improved chrF++? If not, what could explain the disconnect?
- Was the alignment loss much larger or smaller than MT loss? What was the actual scale of `lambda * L_CL`?
- Are failures due to wrong meaning, unnatural Pidgin, or reference inconsistency/domain skew?
- How much does the MTurk-heavy training mix (absent from validation/test) shift the model away from news and spoken Pidgin?
- What limitations arise from in-batch negatives, ~35k pairs, one seed, fixed training budget, and noise in evaluation references?

Do **not** make claims about improvement unless both final metrics and their uncertainties have been computed.

## 14. Agent acceptance checklist

The implementing agent should treat the notebook as **done** only when:

- [ ] Notebook executes top-to-bottom in smoke mode without manual variable fixes.
- [ ] Full-data run loads actual columns from each source (`translation.en/pcm`, `text_en/text_orthographic`, `eng_translation/pidgin_source`) and saves exact, leakage-audited splits.
- [ ] Both models initialize separately from the same `google/mt5-small` checkpoint.
- [ ] Translation labels mask padding; contrastive encoder masks attention-pad positions in pooling.
- [ ] Symmetric InfoNCE positive pairs are batch diagonal; gradients reach English/Pidgin encoder and translation decoder.
- [ ] Baseline and contrastive training budgets, seed, checkpoint selection and deterministic decoding are matched.
- [ ] Best model/tokenizer checkpoints can be restored and used standalone.
- [ ] chrF++ uses `word_order=2`, full test corpus, same paired references, plus paired CI.
- [ ] Retrieval compares every English embedding with **all** gold Pidgin test embeddings, using each trained model's own encoder, plus paired CI.
- [ ] Human evaluation is blinded, scored against a predeclared rubric, and never fabricated.
- [ ] All metrics/figures are actually generated from runs; no hardcoded results or example scores.
- [ ] Notebook concludes with a candid limitations section and reproducibility instructions.

## 15. References / implementation anchors

- Project proposal: *Evaluating Contrastive Learning for English to Nigerian Pidgin Translation*, uploaded PDF.
- Datasets: MAFAND-MT en-pcm (masakhane-io/lafand-mt), UD Naija-NSC, MTurk_Pidgin.csv (Tommy0201/LowResource-Paper-Dataset); commits pinned in `src/naijattention/data.py`
- Official Hugging Face seq2seq translation guide: https://huggingface.co/docs/transformers/tasks/translation
- mT5 model: https://huggingface.co/google/mt5-small
- SacreBLEU CHRF documentation/source: https://github.com/mjpost/sacrebleu
- Contrastive framework background: Pan et al. (2021), *Contrastive Learning for Many-to-Many Multilingual Neural Machine Translation* (cited in proposal).

**Research-integrity note:** The proposal's original OPUS-MT560 English–Pidgin subset is **not** the three-source combined dataset specified in Section 3A, nor necessarily the BBC Pidgin corpus mentioned in the motivation. Do not conflate these datasets. The original OPUS preview is predominantly religious; the updated combined dataset mixes news and spoken material. Report domain-specific results, source license obligations, and any residual alignment problems.
