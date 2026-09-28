"""Checks that the saved splits are the right size, balanced, disjoint and repeatable.

Run after `python scripts/download_data.py`:  pytest tests/test_splits.py
"""
import pandas as pd
import pytest

from src.data import (AG_LABELS, BBC_LABELS, build_splits, fix_glued_sentences, load,
                      parse_bbc_article, repair_ag_text)
from src.utils import SPLITS

pytestmark = pytest.mark.skipif(not (SPLITS / "ag_train.parquet").exists(),
                                reason="run scripts/download_data.py first")


@pytest.fixture(scope="module")
def s():
    return {p.stem: pd.read_parquet(p) for p in SPLITS.glob("*.parquet")}


# ---- AG News ---------------------------------------------------------------
def test_ag_sizes(s):
    assert len(s["ag_train"]) == 30_000
    assert len(s["ag_train_20k"]) == 20_000
    assert len(s["ag_val"]) == 2_000
    assert len(s["ag_test"]) == 7_600


def test_ag_balanced(s):
    assert s["ag_train"]["label"].value_counts().tolist() == [7_500] * 4
    assert s["ag_train_20k"]["label"].value_counts().tolist() == [5_000] * 4
    assert s["ag_test"]["label"].value_counts().tolist() == [1_900] * 4
    assert sorted(s["ag_train"]["label_name"].unique()) == sorted(AG_LABELS)


def test_ag_disjoint_and_nested(s):
    train, val = set(s["ag_train"]["orig_idx"]), set(s["ag_val"]["orig_idx"])
    assert not train & val                                  # val never seen in training
    assert set(s["ag_train_20k"]["orig_idx"]) <= train      # 20k is a subset of the 30k
    assert not set(s["ag_test"]["text"]) & set(s["ag_train"]["text"])


def test_ag_text_repaired(s):
    for bad in ["#39;", "quot;", "&lt;", "\\"]:
        assert not s["ag_train"]["text"].str.contains(bad, regex=False).any(), bad


# ---- BBC -------------------------------------------------------------------
def test_bbc(s):
    bbc = s["bbc"]
    assert bbc["body"].is_unique                           # duplicates removed
    assert sorted(bbc["category"].unique()) == BBC_LABELS
    assert (bbc["title"].str.len() > 0).all() and (bbc["summary"].str.len() > 0).all()
    assert not bbc["body"].str.contains(r"\.\.[A-Z]").any()  # no glued paragraphs left
    assert bbc["summ_sample"].sum() == 100
    assert abs((bbc["lm_split"] == "test").mean() - 0.2) < 0.01


# ---- TweetEval -------------------------------------------------------------
def test_tweets(s):
    assert [len(s[f"tweet_{p}"]) for p in ["train", "val", "test"]] == [45_615, 2_000, 12_284]
    assert set(s["tweet_train"]["label"]) == {0, 1, 2}


# ---- Determinism and loader ------------------------------------------------
def test_rebuild_is_identical(s):
    rebuilt = build_splits()
    for name, df in rebuilt.items():
        pd.testing.assert_frame_equal(df, s[name], check_like=False)


def test_fast_mode():
    df = load("ag_train", fast=True)
    assert df["label"].value_counts().tolist() == [500] * 4
    assert df.equals(load("ag_train", fast=True))          # same rows every time


# ---- Repair functions on hand-written examples -----------------------------
def test_repair_examples():
    assert repair_ag_text("the Web #39;s No. 1") == "the Web's No. 1"
    assert repair_ag_text("AT amp;T costs #36;10") == "AT&T costs $10"
    assert repair_ag_text("&lt;b&gt;Hi&lt;/b&gt; there\\again") == "Hi there again"
    assert parse_bbc_article("Big win..It rained.Then sun...Next day") == \
        ("Big win", "It rained. Then sun. Next day")
    assert fix_glued_sentences("U.S. firm") == "U.S. firm"   # abbreviations untouched
