import tempfile
from pathlib import Path

import numpy as np

from swarm_optimizer.archive import Archive
from swarm_optimizer.genome import Genome


def _g(p="v"):
    return Genome.from_seed() if p == "seed" else Genome(prompt_text=p)


def test_add_returns_incrementing_ids_and_counts_children():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        root = arc.add(_g("root"), mutation_type="seed")
        child = arc.add(_g("child"), parent_id=root, mutation_type="diff_a",
                        artifact_touched="A")
        assert root == 0 and child == 1
        assert arc.get(root).children == 1
        assert arc.get(child).parent_id == root
        assert arc.get(child).artifact_touched == "A"


def test_record_elo_and_championship():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        i = arc.add(_g("a"))
        arc.record_elo(i, 1240.0)
        arc.record_championship(i, score=0.6, metrics={"Recall_rel": 0.4})
        assert arc.get(i).elo == 1240.0
        assert arc.get(i).championship_score == 0.6


def test_champion_picks_best_championship_score():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        a = arc.add(_g("a")); b = arc.add(_g("b"))
        arc.record_championship(a, 0.5, {})
        arc.record_championship(b, 0.7, {})
        assert arc.champion().id == b


def test_champion_none_when_no_championship_run():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        arc.add(_g("a"))
        assert arc.champion() is None


def test_top_by_elo_orders_desc():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        a = arc.add(_g("a")); b = arc.add(_g("b")); c = arc.add(_g("c"))
        arc.record_elo(a, 1100); arc.record_elo(b, 1300); arc.record_elo(c, 900)
        top = arc.top_by_elo(2)
        assert [e.id for e in top] == [b, a]


def test_select_parent_returns_existing_id():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        arc.add(_g("a")); arc.add(_g("b"))
        rng = np.random.default_rng(0)
        pid = arc.select_parent(rng)
        assert pid in (0, 1)


def test_persists_and_reloads():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "arch.jsonl"
        arc = Archive(path)
        i = arc.add(_g("persisted"))
        arc.record_elo(i, 1234.0)
        reloaded = Archive(path)
        assert reloaded.get(i).genome.prompt_text == "persisted"
        assert reloaded.get(i).elo == 1234.0


def test_total_tokens_sums_championship_metrics():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        a = arc.add(_g("a"))
        arc.record_championship(a, 0.5, {"tokens": 1000})
        assert arc.total_tokens() == 1000


def test_diagnosis_and_delta_persist():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "arch.jsonl"
        arc = Archive(path)
        i = arc.add(_g("a"), diagnosis="faltan relaciones")
        arc.record_delta(i, 0.12)
        reloaded = Archive(path)
        assert reloaded.get(i).diagnosis == "faltan relaciones"
        assert reloaded.get(i).fitness_delta == 0.12


# ── extensiones del informe 2026-06-09 ───────────────────────────── #
def test_lineage_walks_to_seed():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        root = arc.add(_g("root"), mutation_type="seed")
        c1 = arc.add(_g("c1"), parent_id=root, mutation_type="diff_a")
        c2 = arc.add(_g("c2"), parent_id=c1, mutation_type="diff_b")
        chain = arc.lineage(c2)
        assert [e.id for e in chain] == [c2, c1, root]


def test_memory_snippets_formats_best_and_worst():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        root = arc.add(_g("root"), mutation_type="seed")
        good = arc.add(_g("g"), parent_id=root, mutation_type="diff_b",
                       artifact_touched="B", diagnosis="reglas de validación")
        bad = arc.add(_g("b"), parent_id=good, mutation_type="diff_a",
                      artifact_touched="A")
        leaf = arc.add(_g("l"), parent_id=bad, mutation_type="diff_a")
        arc.record_delta(good, 0.05)
        arc.record_delta(bad, -0.08)
        mem = arc.memory_snippets(leaf)
        assert "Funcionó:" in mem and "+0.050" in mem
        assert "NO funcionó:" in mem and "-0.080" in mem
        assert "reglas de validación" in mem


def test_memory_snippets_empty_without_deltas():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        root = arc.add(_g("root"))
        assert arc.memory_snippets(root) == ""


def test_meta_review_persists_and_reloads():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "arch.jsonl"
        arc = Archive(path)
        i = arc.add(_g("a"))
        arc.record_meta_review(i, "1. patrón co_occurs espurio")
        reloaded = Archive(path)
        assert reloaded.get(i).meta_review == "1. patrón co_occurs espurio"
