import numpy as np
from swarm_optimizer.elo import expected_score, update_pairwise, sample_parent


def test_expected_score_symmetric():
    assert abs(expected_score(1000, 1000) - 0.5) < 1e-9


def test_winner_gains_loser_loses():
    new_a, new_b = update_pairwise(1000, 1000, score_a=1.0, k=32)
    assert new_a > 1000
    assert new_b < 1000
    assert abs((new_a - 1000) - (1000 - new_b)) < 1e-9   # zero-sum


def test_upset_moves_more_than_expected_win():
    # un under-rated (800) que vence a un favorito (1200) gana más que al revés
    underdog_gain = update_pairwise(800, 1200, 1.0)[0] - 800
    favorite_gain = update_pairwise(1200, 800, 1.0)[0] - 1200
    assert underdog_gain > favorite_gain


def test_sample_parent_prefers_high_elo_low_children():
    rng = np.random.default_rng(0)
    entries = [
        {"elo": 1300, "children": 0},   # fuerte, inexplorado -> favorito
        {"elo": 900, "children": 10},   # débil, sobre-explorado
    ]
    picks = [sample_parent(entries, rng) for _ in range(400)]
    assert picks.count(0) > picks.count(1)


def test_sample_parent_returns_valid_index():
    rng = np.random.default_rng(1)
    entries = [{"elo": 1000, "children": 0}]
    assert sample_parent(entries, rng) == 0
