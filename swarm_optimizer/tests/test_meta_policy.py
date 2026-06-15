import tempfile
from pathlib import Path

import numpy as np

from swarm_optimizer.meta_policy import (
    ARMS, MetaPolicy, reward_from_delta,
)


def test_reward_from_delta_maps_and_clips():
    assert reward_from_delta(0.0) == 0.5
    assert reward_from_delta(0.2) == 1.0
    assert reward_from_delta(-0.2) == 0.0
    # clip de outliers
    assert reward_from_delta(5.0) == 1.0
    assert reward_from_delta(-5.0) == 0.0


def test_choose_returns_valid_arm():
    p = MetaPolicy()
    rng = np.random.default_rng(0)
    for _ in range(50):
        arm = p.choose(rng)
        assert arm in ARMS


def test_choose_respects_available_ops():
    p = MetaPolicy()
    rng = np.random.default_rng(1)
    for _ in range(50):
        op, bias = p.choose(rng, available_ops=["diff_a", "diff_b"])
        assert op in ("diff_a", "diff_b")
        assert bias in ("exploit", "explore")


def test_update_shifts_posterior_towards_rewarded_arm():
    p = MetaPolicy()
    good, bad = ("diff_b", "exploit"), ("diff_a", "explore")
    for _ in range(30):
        p.update(good, 1.0)
        p.update(bad, 0.0)
    means = p.posterior_means()
    assert means["diff_b|exploit"] > 0.9
    assert means["diff_a|explore"] < 0.1
    # con posteriors tan separados, Thompson debe preferir el brazo bueno
    rng = np.random.default_rng(2)
    picks = [p.choose(rng, available_ops=["diff_a", "diff_b"]) for _ in range(200)]
    assert picks.count(good) > picks.count(bad)


def test_epsilon_floor_keeps_all_arms_alive():
    p = MetaPolicy()
    # castigar fuertemente todos los brazos menos uno
    for arm in ARMS:
        if arm != ("diff_a", "exploit"):
            for _ in range(50):
                p.update(arm, 0.0)
    rng = np.random.default_rng(3)
    picks = {p.choose(rng) for _ in range(2000)}
    # el ε-floor garantiza que brazos castigados sigan apareciendo a veces
    assert len(picks) > 1


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "policy.json"
        p = MetaPolicy(path)
        p.update(("cross", "explore"), 1.0)
        p2 = MetaPolicy(path)
        assert p2.posterior_means()["cross|explore"] == \
            p.posterior_means()["cross|explore"]


def test_credit_championship_rewards_window():
    p = MetaPolicy()
    window = [("diff_b", "exploit"), ("diff_b", "exploit")]
    before = p.posterior_means()["diff_b|exploit"]
    p.credit_championship(window, score_delta=0.1)
    after = p.posterior_means()["diff_b|exploit"]
    assert after > before


def test_credit_championship_penalizes_on_regression():
    p = MetaPolicy()
    window = [("fresh", "explore")]
    before = p.posterior_means()["fresh|explore"]
    p.credit_championship(window, score_delta=-0.1)
    after = p.posterior_means()["fresh|explore"]
    assert after < before


def test_credit_championship_empty_window_is_noop():
    p = MetaPolicy()
    before = dict(p.posterior_means())
    p.credit_championship([], score_delta=0.5)
    assert p.posterior_means() == before
