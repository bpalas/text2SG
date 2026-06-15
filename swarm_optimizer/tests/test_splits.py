"""Test stratified eval/test splits."""
from pathlib import Path

import pandas as pd

from swarm_optimizer.splits import generate_splits


def test_splits_counts():
    """Verify eval and test counts."""
    splits = generate_splits(seed=42)
    # Actual data has 93 articles total, proportionally stratified
    # test_ratio = 30/93 (30.11%), eval_ratio = 63/93 (67.74%)
    assert len(splits["eval"]) == 63
    assert len(splits["test"]) == 30
    assert len(set(splits["eval"]) & set(splits["test"])) == 0  # no overlap


def test_splits_reproducible():
    """Verify deterministic with seed."""
    s1 = generate_splits(seed=42)
    s2 = generate_splits(seed=42)
    assert s1["eval"] == s2["eval"]
    assert s1["test"] == s2["test"]


def test_splits_cover_all_strata():
    """Verify test set has at least 1 article from each stratum."""
    splits = generate_splits(seed=42)
    arts = pd.read_parquet(
        Path(__file__).parent.parent.parent.parent
        / "gold_standard_v5/data/pilot_gold_articles.parquet"
    )
    test_arts = arts[arts["article_id"].isin(splits["test"])]
    assert test_arts["stratum"].nunique() == 8


from swarm_optimizer.splits import subsample


def test_subsample_size_and_membership():
    pool = [f"a{i}" for i in range(20)]
    s = subsample(pool, k=5, seed=0)
    assert len(s) == 5
    assert all(x in pool for x in s)
    assert len(set(s)) == 5


def test_subsample_deterministic_with_seed():
    pool = [f"a{i}" for i in range(20)]
    assert subsample(pool, 5, seed=42) == subsample(pool, 5, seed=42)


def test_subsample_varies_with_seed():
    pool = [f"a{i}" for i in range(20)]
    assert subsample(pool, 5, seed=1) != subsample(pool, 5, seed=2)


def test_subsample_caps_at_pool_size():
    pool = ["a", "b", "c"]
    assert sorted(subsample(pool, 10, seed=0)) == ["a", "b", "c"]
