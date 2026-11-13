"""Load, audit, clean, leakage-filter and split the three English->Pidgin sources.

Sources are pinned to exact upstream commits. Validation/test come only from the
*published* MAFAND and UD Naija-NSC dev/test splits; MTurk is training-only.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

SOURCES = {
    "mafand": {
        "repo": "masakhane-io/lafand-mt",
        "sha": "c1a423500753a28b2fae8f5d5e1634de768cd89b",
        "files": {s: f"data/json_files/en-pcm/{s}.json" for s in ("train", "dev", "test")},
        "license": "CC BY-NC 4.0",
        "domain": "news",
    },
    "naija": {
        "repo": "UniversalDependencies/UD_Naija-NSC",
        "sha": "8877d801f1ad9e5e7eb52f509d83f25d6695c2e1",
        "files": {s: f"pcm_nsc-ud-{s}.conllu" for s in ("train", "dev", "test")},
        "license": "CC BY-SA 4.0",
        "domain": "spoken",
    },
    "mturk": {
        "repo": "Tommy0201/LowResource-Paper-Dataset",
        "sha": "bf88f5e020232a2116059dddd486deb0edbd4124",
        "files": {"train": "Nigerian-Pidgin/MTurk_Pidgin.csv"},
        "license": "unconfirmed - check before redistributing",
        "domain": "unknown",  # provenance of the MTurk sentences has not been verified
    },
}
# When identical normalised pairs appear in several sources, keep the highest-priority copy.
SOURCE_PRIORITY = {"naija": 0, "mafand": 1, "mturk": 2}
EVAL_SOURCES = ("mafand", "naija")
SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}

# Usable row counts after parsing, at the pinned SHAs. A mismatch means the parser or
# the pinned files changed and should be investigated, not silently accepted.
EXPECTED_RAW_COUNTS = {
    ("mafand", "train"): 4790, ("mafand", "dev"): 1484, ("mafand", "test"): 1564,
    ("naija", "train"): 7278, ("naija", "dev"): 990, ("naija", "test"): 972,
    ("mturk", "train"): 23382,
}


# --------------------------------------------------------------------------- download

def raw_url(source: str, split: str) -> str:
    spec = SOURCES[source]
    return f"https://raw.githubusercontent.com/{spec['repo']}/{spec['sha']}/{spec['files'][split]}"


def fetch_raw(source: str, split: str, raw_dir: str | Path, retries: int = 3, timeout: int = 60) -> Path:
    """Download one source file into a commit-specific cache directory (idempotent)."""
    spec = SOURCES[source]
    dest = Path(raw_dir) / source / spec["sha"][:12] / Path(spec["files"][split]).name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = raw_url(source, split)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "naijattention-data-loader"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except Exception as err:  # network errors: retry with backoff
            last_err = err
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- parsers

def parse_mafand(path: Path, split: str) -> pd.DataFrame:
    # JSON Lines despite the .json extension.
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    for i, line in enumerate(lines):
        tr = json.loads(line)["translation"]
        rows.append({"english": tr["en"], "pidgin": tr["pcm"], "source_id": f"mafand-{split}-{i}"})
    return pd.DataFrame(rows)


def parse_conllu_metadata(text: str) -> list[dict]:
    """Return the `# key = value` metadata of every sentence block; token rows are ignored."""
    records, meta = [], {}
    for line in text.splitlines() + [""]:
        if line.strip():
            if line.startswith("# ") and " = " in line:
                k, v = line[2:].split(" = ", 1)
                meta[k.strip()] = v
        elif meta:
            if "sent_id" in meta:
                records.append(meta)
            meta = {}
    return records


def parse_naija(path: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    blocks = parse_conllu_metadata(Path(path).read_text(encoding="utf-8"))
    rows, dropped = [], []
    for m in blocks:
        # text_orthographic is the readable Pidgin; the raw `text` field has transcription markup.
        if not m.get("text_en", "").strip() or not m.get("text_orthographic", "").strip():
            dropped.append(m["sent_id"])
            continue
        rows.append({
            "english": m["text_en"],
            "pidgin": m["text_orthographic"],
            "source_id": m["sent_id"],
            "speaker_id": m.get("speaker_id"),
            "recording_id": m["sent_id"].rsplit("__", 1)[0],
        })
    return pd.DataFrame(rows), dropped


def parse_mturk(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = {"pidgin_source", "eng_translation"} - set(d.columns)
    if missing:
        raise ValueError(f"MTurk CSV is missing columns {missing}; got {list(d.columns)}")
    # Collected as Pidgin -> English; we reverse the direction.
    return pd.DataFrame({
        "english": d["eng_translation"].astype(str),
        "pidgin": d["pidgin_source"].astype(str),
        "source_id": [f"mturk-{i}" for i in range(len(d))],
    })


def load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Fetch (or reuse cached) source files and return one tagged DataFrame plus provenance."""
    raw_dir = Path(data_dir) / "raw"
    frames, provenance, dropped = [], {}, {}
    for source, spec in SOURCES.items():
        for split in spec["files"]:
            path = fetch_raw(source, split, raw_dir)
            if source == "mafand":
                df = parse_mafand(path, split)
            elif source == "naija":
                df, dropped[f"naija/{split}"] = parse_naija(path, split)
            else:
                df = parse_mturk(path)
            df["source"], df["source_split"] = source, split
            frames.append(df)
            provenance[f"{source}/{split}"] = {
                "url": raw_url(source, split), "commit": spec["sha"], "sha256": sha256_file(path),
                "rows": len(df), "license": spec["license"],
            }
    raw = pd.concat(frames, ignore_index=True)
    for col in ("speaker_id", "recording_id"):
        raw[col] = raw[col].astype("string")
    raw["_order"] = range(len(raw))
    return raw, {"files": provenance, "dropped_missing_fields": dropped}


def check_raw_counts(raw: pd.DataFrame) -> dict:
    got = raw.groupby(["source", "source_split"]).size().to_dict()
    mismatches = {f"{k[0]}/{k[1]}": {"expected": v, "got": got.get(k)} for k, v in EXPECTED_RAW_COUNTS.items() if got.get(k) != v}
    if mismatches:
        raise AssertionError(f"raw counts differ from the pinned-SHA expectations: {mismatches}")
    return {f"{k[0]}/{k[1]}": v for k, v in got.items()}


# --------------------------------------------------------------------------- cleaning

_WS = re.compile(r"\s+")


def clean_text(s) -> str:
    """Conservative cleaning for model text: NFC, trim, collapse whitespace. Case/punctuation kept."""
    if s is None or (isinstance(s, float) and s != s) or s is pd.NA:
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFC", str(s))).strip()


def norm_key(s: str) -> str:
    """Aggressive key used only for duplicate / leakage detection, never as model input."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", s).casefold()).strip()


def build_splits(raw: pd.DataFrame, train_subset_size: int | None, seed: int) -> tuple[dict[str, pd.DataFrame], dict, pd.DataFrame]:
    """Returns (splits, audit, removal_log). splits has keys train / validation / test."""
    audit: dict = {}
    df = raw.copy()
    df["english"] = df["english"].map(clean_text)
    df["pidgin"] = df["pidgin"].map(clean_text)
    empty = (df["english"] == "") | (df["pidgin"] == "")
    audit["dropped_empty_after_cleaning"] = df[empty].groupby("source").size().to_dict()
    df = df[~empty].copy()
    df["eng_norm"] = df["english"].map(norm_key)
    df["pcm_norm"] = df["pidgin"].map(norm_key)
    df["split"] = df["source_split"].map(SPLIT_MAP)
    df["domain"] = df["source"].map(lambda s: SOURCES[s]["domain"])

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    removal_log = []

    # 1. Leakage: drop any training row whose English OR Pidgin appears in validation or test.
    held_eng = set(val["eng_norm"]) | set(test["eng_norm"])
    held_pcm = set(val["pcm_norm"]) | set(test["pcm_norm"])
    eng_hit, pcm_hit = train["eng_norm"].isin(held_eng), train["pcm_norm"].isin(held_pcm)
    leak = eng_hit | pcm_hit
    reason = (eng_hit.map({True: "eng", False: ""}) + pcm_hit.map({True: "+pcm", False: ""})).str.strip("+")
    removal_log.append(train[leak].assign(removal_reason="heldout_overlap:" + reason[leak]))
    audit["train_rows_before_filtering"] = len(train)
    audit["leakage_removed"] = train[leak].groupby("source").size().to_dict()
    train = train[~leak]

    # 2. Exact normalised-pair duplicates within train: keep highest-priority source, then first seen.
    train = train.assign(_prio=train["source"].map(SOURCE_PRIORITY)).sort_values(["_prio", "_order"])
    dup = train.duplicated(["eng_norm", "pcm_norm"], keep="first")
    removal_log.append(train[dup].assign(removal_reason="duplicate_pair"))
    audit["duplicate_pairs_removed"] = train[dup].groupby("source").size().to_dict()
    train = train[~dup].drop(columns="_prio").sort_values("_order")

    # English inputs with several distinct valid Pidgin targets are kept (all in train).
    n_targets = train.groupby("eng_norm")["pcm_norm"].nunique()
    audit["train_english_with_multiple_targets"] = int((n_targets > 1).sum())
    audit["train_rows_after_filtering"] = len(train)

    # 3. Optional deterministic subsample: keep all naija + mafand, fill the rest from mturk.
    train = subsample_train(train, train_subset_size, seed)
    audit["train_subset_size"] = train_subset_size

    # 4. Published dev/test overlap is reported, never "fixed" by moving rows.
    audit["published_dev_test_overlap"] = {
        "dev_rows_with_test_english": int(val["eng_norm"].isin(set(test["eng_norm"])).sum()),
        "dev_rows_with_test_pidgin": int(val["pcm_norm"].isin(set(test["pcm_norm"])).sum()),
        "dev_rows_with_test_pair": int(pd.MultiIndex.from_frame(val[["eng_norm", "pcm_norm"]]).isin(
            pd.MultiIndex.from_frame(test[["eng_norm", "pcm_norm"]])).sum()),
    }
    # 5. Naija speaker / recording overlap across published splits (reported only).
    nj = df[df["source"] == "naija"]
    by = {s: nj[nj["split"] == s] for s in ("train", "validation", "test")}
    audit["naija_cross_split_overlap"] = {
        f"{a}_vs_{b}": {
            "shared_speakers": len(set(by[a]["speaker_id"].dropna()) & set(by[b]["speaker_id"].dropna())),
            "shared_recordings": len(set(by[a]["recording_id"].dropna()) & set(by[b]["recording_id"].dropna())),
        }
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    # 6. Repeated targets inside evaluation pools (affects strict retrieval Recall@1).
    audit["eval_repeated_pidgin_rows"] = {
        f"{split}/{src}": int(g["pcm_norm"].duplicated(keep=False).sum())
        for split, d in (("validation", val), ("test", test)) for src, g in d.groupby("source")
    }

    splits = {"train": train.reset_index(drop=True), "validation": val.reset_index(drop=True), "test": test.reset_index(drop=True)}
    audit["split_counts"] = {k: v.groupby("source").size().to_dict() for k, v in splits.items()}
    return splits, audit, pd.concat(removal_log, ignore_index=True)


def subsample_train(train: pd.DataFrame, size: int | None, seed: int) -> pd.DataFrame:
    if size is None or size >= len(train):
        return train
    core = train[train["source"] != "mturk"]
    if len(core) <= size:
        extra = train[train["source"] == "mturk"].sample(n=size - len(core), random_state=seed)
        out = pd.concat([core, extra])
    else:
        out = train.sample(n=size, random_state=seed)
    return out.sort_values("_order")


def smoke_subset(splits: dict[str, pd.DataFrame], n_train: int, n_eval_per_source: int, seed: int) -> dict[str, pd.DataFrame]:
    """Tiny deterministic subset for smoke tests. Results on it are NOT research results."""
    def per_source(d, n):
        return pd.concat([g.sample(n=min(n, len(g)), random_state=seed) for _, g in d.groupby("source")]).sort_values("_order")
    train = splits["train"]
    n_src = train["source"].nunique()
    return {
        "train": per_source(train, -(-n_train // n_src)).head(n_train).reset_index(drop=True),
        "validation": per_source(splits["validation"], n_eval_per_source).reset_index(drop=True),
        "test": per_source(splits["test"], n_eval_per_source).reset_index(drop=True),
    }


# --------------------------------------------------------------------------- invariants & IO

def check_invariants(splits: dict[str, pd.DataFrame], expected_train_size: int | None = None) -> dict:
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    held_eng = set(va["eng_norm"]) | set(te["eng_norm"])
    held_pcm = set(va["pcm_norm"]) | set(te["pcm_norm"])
    checks = {
        "no_empty_strings": all((d[c].str.len() > 0).all() for d in splits.values() for c in ("english", "pidgin")),
        "train_english_disjoint_from_heldout": not tr["eng_norm"].isin(held_eng).any(),
        "train_pidgin_disjoint_from_heldout": not tr["pcm_norm"].isin(held_pcm).any(),
        "no_duplicate_pairs_in_train": not tr.duplicated(["eng_norm", "pcm_norm"]).any(),
        "source_ids_unique": all(not d["source_id"].duplicated().any() for d in splits.values()),
        "heldout_sources_are_mafand_naija": set(va["source"]) | set(te["source"]) <= set(EVAL_SOURCES),
        "heldout_keeps_published_split": (va["source_split"] == "dev").all() and (te["source_split"] == "test").all(),
        "train_only_from_train_split": (tr["source_split"] == "train").all(),
    }
    if expected_train_size is not None:
        checks["train_size_matches"] = len(tr) == expected_train_size
    checks = {k: bool(v) for k, v in checks.items()}
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"data invariants failed: {failed}")
    return checks


SAVE_COLS = ["english", "pidgin", "source", "source_split", "source_id", "split", "domain",
             "eng_norm", "pcm_norm", "speaker_id", "recording_id"]


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path, removal_log: pd.DataFrame | None = None) -> None:
    out = Path(out_dir) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name, d in splits.items():
        d[SAVE_COLS].to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        if name != "train":
            for src, g in d.groupby("source"):
                g[SAVE_COLS].to_json(out / f"{name}_{src}.jsonl", orient="records", lines=True, force_ascii=False)
    if removal_log is not None:
        removal_log[["source", "source_split", "source_id", "english", "pidgin", "removal_reason"]].to_csv(
            out / "removal_audit.csv", index=False)
