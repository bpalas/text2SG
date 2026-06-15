"""Ratings ELO para selección evolutiva + muestreo de padres (DGM)."""
from __future__ import annotations

import math

K_DEFAULT = 32
ELO_BASE = 1000.0
SIGMOID_SCALE = 200.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_pairwise(rating_a: float, rating_b: float, score_a: float, k: float = K_DEFAULT):
    """score_a in {1.0 win, 0.5 tie, 0.0 loss}. Devuelve (new_a, new_b)."""
    ea = expected_score(rating_a, rating_b)
    delta = k * (score_a - ea)
    return rating_a + delta, rating_b - delta


def _weight(entry: dict) -> float:
    elo = entry.get("elo", ELO_BASE)
    children = entry.get("children", 0)
    pref = 1.0 / (1.0 + math.exp(-(elo - ELO_BASE) / SIGMOID_SCALE))  # sigmoid(ELO)
    return pref / (1.0 + children)                                    # 1/(1+#hijos)


def sample_parent(entries: list[dict], rng) -> int:
    """Muestrea índice de padre ∝ sigmoid(ELO) y ∝ 1/(1+#hijos). rng = np Generator."""
    weights = [_weight(e) for e in entries]
    total = sum(weights)
    if total <= 0:
        return int(rng.integers(len(entries)))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(entries) - 1
