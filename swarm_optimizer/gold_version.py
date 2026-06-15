"""Resuelve las rutas del gold standard según GOLD_VERSION.

- "current" (default): el gold real en el repo hermano gold_standard_v5/ (93 art).
- "v3": el gold unificado local en data/gold_v3/ (193 art).

Centraliza lo que antes estaba hardcodeado en loop.py / splits.py / rubric.py.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).parent.parent          # text2graph-evolve/
_SIBLING = _REPO.parent / "gold_standard_v5" / "data"


def gold_paths(version: str | None = None) -> dict[str, Path]:
    """Devuelve {articles, relations, unions_dir, splits} para la versión pedida.

    version=None -> usa la env var GOLD_VERSION, o "current" si no está seteada.
    """
    version = version or os.environ.get("GOLD_VERSION", "current")
    if version == "v3":
        base = _REPO / "data" / "gold_v3"
        return {
            "articles": base / "articles.parquet",
            "relations": base / "gold_final.parquet",
            "unions_dir": base / "entity_unions",
            "splits": base / "splits.json",
        }
    if version == "current":
        return {
            "articles": _SIBLING / "pilot_gold_articles.parquet",
            "relations": _SIBLING / "pilot_gold_final.parquet",
            "unions_dir": _SIBLING / "pilot_entity_unions",
            "splits": _REPO / "results" / "swarm" / "splits.json",
        }
    raise ValueError(f"GOLD_VERSION desconocido: {version!r}")
