"""Stratified eval/test splits from pilot_gold_articles.parquet."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from swarm_optimizer.gold_version import gold_paths


def gold_articles_path():
    return gold_paths()["articles"]


def splits_path():
    return gold_paths()["splits"]


def generate_splits(seed: int = 42) -> dict:
    """
    Generate stratified eval/test split.

    Proportionally allocates test articles per stratum at ~30% ratio (30/93).
    Ensures each stratum has at least 1 article in test set.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict with "eval" (list of article_id), "test" (list of article_id), and "seed".
    """
    arts = pd.read_parquet(gold_articles_path())
    rng = np.random.default_rng(seed)

    eval_ids: list[str] = []
    test_ids: list[str] = []

    for _, group in arts.groupby("stratum", sort=False):
        ids = group["article_id"].tolist()
        ids_arr = np.array(ids)
        rng.shuffle(ids_arr)
        ids = ids_arr.tolist()

        # Allocate ~30% to test, at least 1
        n_test = max(1, round(len(ids) * 30 / 93))
        test_ids.extend(ids[:n_test])
        eval_ids.extend(ids[n_test:])

    return {"eval": eval_ids, "test": test_ids, "seed": seed}


def load_splits(seed: int = 42) -> dict:
    """
    Load splits from disk, or generate and persist if missing.

    Args:
        seed: Random seed (only used if generating).

    Returns:
        Dict with "eval", "test", and "seed".
    """
    path = splits_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    splits = generate_splits(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def subsample(pool: list[str], k: int, seed: int) -> list[str]:
    """Submuestreo aleatorio sin reemplazo de k artículos del pool (RoboPhD undersampling).
    Determinista dado el seed. Si k >= len(pool), devuelve el pool completo (orden barajado)."""
    rng = np.random.default_rng(seed)
    arr = np.array(pool, dtype=object)
    rng.shuffle(arr)
    n = min(k, len(pool))
    return arr[:n].tolist()
