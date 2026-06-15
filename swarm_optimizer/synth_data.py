"""Carga de datasets sintéticos (verdad plantada) y split estratificado.
Espejo de splits.py pero para results/synthetic/<dataset>/.
Los loaders aceptan `base` para tests herméticos (tmp_path)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).parent.parent / "results/synthetic"


def _dir(dataset: str, base: Path | None = None) -> Path:
    return (base or SYNTH_ROOT) / dataset


def load_synth_articles(dataset: str = "v1", base: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(_dir(dataset, base) / "articles.parquet")


def load_synth_gold(dataset: str = "v1", base: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(_dir(dataset, base) / "gold.parquet")


def load_synth_unions(dataset: str = "v1", base: Path | None = None) -> dict:
    """{article_id: {uid: {union_id, type, canonical_names, surfaces}}}."""
    return json.loads((_dir(dataset, base) / "unions.json").read_text(encoding="utf-8"))


def load_synth_split(dataset: str = "v1", seed: int = 42, base: Path | None = None) -> dict:
    """Split estratificado por dominio×registro, ~75% train / 25% test, persistido en
    <dataset>/split.json. Cada estrato aporta al menos 1 a test.

    Deduplica article_id antes de estratificar: el sintético tiene ids duplicados con
    bodies distintos (pilot 1, v1 5); sin dedup el mismo id caería en train Y test,
    contaminando el held-out. El `seed` solo aplica en la 1ª generación (después se lee
    el split.json persistido)."""
    path = _dir(dataset, base) / "split.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    arts = load_synth_articles(dataset, base).drop_duplicates("article_id", keep="first")
    rng = np.random.default_rng(seed)
    strat = arts["dominio"].astype(str) + "|" + arts["registro"].astype(str)
    train, test = [], []
    for _, group in arts.groupby(strat, sort=True):
        ids = group["article_id"].tolist()
        arr = np.array(ids, dtype=object)
        rng.shuffle(arr)
        ids = arr.tolist()
        n_test = max(1, round(len(ids) * 0.25))
        test.extend(ids[:n_test])
        train.extend(ids[n_test:])
    assert set(train).isdisjoint(test), "split contaminado: id en train y test"
    split = {"train": sorted(train), "test": sorted(test), "seed": seed}
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split
