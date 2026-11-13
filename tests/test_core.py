"""Fast tests: no downloads. Model-level checks use a tiny randomly initialised mT5."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from naijattention import checks  # noqa: E402
from naijattention.data import build_splits, check_invariants, clean_text, norm_key, parse_conllu_metadata  # noqa: E402
from naijattention.evaluate import corpus_chrf, paired_bootstrap_chrf, retrieval_metrics  # noqa: E402


def test_static_checks():
    checks.run_static_checks()


def test_clean_and_norm():
    assert clean_text("  Wetin  dey happen?\n") == "Wetin dey happen?"
    assert clean_text(None) == "" and clean_text(float("nan")) == ""
    assert norm_key("How  You DEY") == norm_key("how you dey")


def test_conllu_metadata_parser_handles_missing_final_blank_line():
    text = "# sent_id = A__1\n# text_en = Hi\n# text_orthographic = How far\n1\tHow\n\n# sent_id = A__2\n# text_en = Bye\n1\tx"
    recs = parse_conllu_metadata(text)
    assert [r["sent_id"] for r in recs] == ["A__1", "A__2"] and "text_orthographic" not in recs[1]


def _toy_raw():
    rows = [
        # source, split, english, pidgin
        ("mafand", "train", "Hello friend", "How far padi"),
        ("mafand", "train", "Leaked english", "unique pcm 1"),      # english overlaps test -> removed
        ("mturk", "train", "Some text", "Leaked pidgin"),           # pidgin overlaps dev -> removed
        ("mturk", "train", "hello  FRIEND", "how far padi"),        # duplicate pair of a higher-priority row
        ("naija", "train", "Good morning", "Mornin o"),
        ("naija", "train", "   ", "empty english"),                 # dropped as empty
        ("mafand", "dev", "Dev sentence", "Leaked pidgin"),
        ("naija", "dev", "Another dev", "Dev pcm"),
        ("mafand", "test", "leaked ENGLISH", "Test pcm"),
        ("naija", "test", "Test two", "Test pcm two"),
    ]
    df = pd.DataFrame(rows, columns=["source", "source_split", "english", "pidgin"])
    df["source_id"] = [f"id{i}" for i in range(len(df))]
    df["speaker_id"] = pd.Series([None] * len(df), dtype="string")
    df["recording_id"] = pd.Series([None] * len(df), dtype="string")
    df["_order"] = range(len(df))
    return df


def test_build_splits_leakage_and_dedup():
    splits, audit, log = build_splits(_toy_raw(), None, 0)
    assert set(splits["train"]["english"]) == {"Hello friend", "Good morning"}
    assert audit["leakage_removed"] == {"mafand": 1, "mturk": 1}
    assert audit["duplicate_pairs_removed"] == {"mturk": 1}
    assert set(log["removal_reason"]) == {"heldout_overlap:eng", "heldout_overlap:pcm", "duplicate_pair"}
    check_invariants(splits)


def test_bootstrap_matches_corpus_chrf():
    refs = ["how far padi", "I dey come", "wetin dey happen"]
    a, b = ["how far", "I dey", "wetin happen"], ["how far padi", "I dey come", "wetin dey"]
    r = paired_bootstrap_chrf(a, b, refs, n=50, seed=0)
    assert r["a"] == pytest.approx(corpus_chrf(a, refs)) and r["b"] == pytest.approx(corpus_chrf(b, refs))
    assert r["delta"] == pytest.approx(r["b"] - r["a"]) and r["ci95"][0] <= r["ci95"][1]


def test_retrieval_reverse_direction():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(20, 8)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    m = retrieval_metrics(E, E)
    assert m["en2pcm_r@1"] == 1.0 and m["pcm2en_r@1"] == 1.0 and m["en2pcm_r@5"] == 1.0


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import MT5Config, MT5ForConditionalGeneration
    torch.manual_seed(0)
    cfg = MT5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=2, num_decoder_layers=2, num_heads=2,
                    decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    return MT5ForConditionalGeneration(cfg)


def test_model_loss_checks_on_tiny_mt5(tiny_model):
    g = torch.Generator().manual_seed(0)
    B = 4
    ids = torch.randint(2, 64, (B, 7), generator=g)
    mask = torch.ones_like(ids)
    mask[0, 5:] = 0
    labels = torch.randint(2, 64, (B, 5), generator=g)
    labels[1, 3:] = -100
    batch = {"input_ids": ids, "attention_mask": mask, "pool_mask": mask.clone(), "labels": labels,
             "pcm_input_ids": torch.randint(2, 64, (B, 6), generator=g), "pcm_attention_mask": torch.ones(B, 6, dtype=torch.long),
             "eng_kid": torch.arange(B), "pcm_kid": torch.arange(B)}
    res = checks.check_model_losses(tiny_model, batch)
    assert res["baseline_loss"] == pytest.approx(res["contrastive_lambda0_loss"], abs=1e-4)


def test_finished_run_reuse_and_mismatch(tmp_path):
    import json
    from dataclasses import replace
    from naijattention import Config
    from naijattention.train import finished_run, run_fingerprint
    cfg = Config.smoke()
    fp = run_fingerprint(cfg, "contrastive", 0.1, 64)
    assert fp == run_fingerprint(replace(cfg, output_dir="elsewhere", bootstrap_samples=5), "contrastive", 0.1, 64)
    assert fp != run_fingerprint(replace(cfg, temperature=0.05), "contrastive", 0.1, 64)
    assert fp != run_fingerprint(cfg, "contrastive", 0.5, 64)
    run = tmp_path / "run"
    assert finished_run(run, fp) is None                      # nothing there
    (run / "best").mkdir(parents=True)
    assert finished_run(run, fp) is None                      # interrupted: no history yet
    (run / "best" / "config.json").write_text("{}")
    (run / "train_history.json").write_text(json.dumps({"config_fingerprint": fp, "x": 1}))
    assert finished_run(run, fp)["x"] == 1                    # finished, same settings -> reuse
    with pytest.raises(FileExistsError):
        finished_run(run, "different")                        # finished, other settings -> refuse
