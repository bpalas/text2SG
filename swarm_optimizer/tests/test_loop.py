import contextlib
import io
import tempfile
from pathlib import Path

import pandas as pd

from swarm_optimizer.genome import Genome
from swarm_optimizer.loop import skirmish_result, should_stop, run_loop
from swarm_optimizer.archive import Archive


def test_skirmish_result_maps_scores_to_winloss():
    # a mejor que b -> 1.0 ; empate -> 0.5 ; a peor -> 0.0
    assert skirmish_result(0.7, 0.5) == 1.0
    assert skirmish_result(0.5, 0.5) == 0.5
    assert skirmish_result(0.3, 0.5) == 0.0


def test_should_stop_on_max_iter():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "a.jsonl")
        assert should_stop(arc, iteration=10, max_iter=10, budget_tokens=10**9) is True


def test_should_stop_on_budget():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "a.jsonl")
        i = arc.add(Genome.from_seed())
        arc.record_championship(i, 0.5, {"tokens": 2_000_000})
        assert should_stop(arc, iteration=1, max_iter=80, budget_tokens=1_000_000) is True


def _fake_extract_fn(article_ids, articles_df, gold_df, union_map, genome):
    """Extractor determinista falso: una relación correcta por artículo si el prompt
    contiene 'BUENO', ninguna si no. Devuelve (predictions, tokens)."""
    preds = []
    good = "BUENO" in genome.prompt_text
    for aid in article_ids:
        rels = [{"from_entity": "A", "to_entity": "B", "act_type": "attacks",
                 "polarity": "negative", "issue": "x", "evidence_quote": "q"}] if good else []
        preds.append({"article_id": aid, "entities": [], "relations": rels, "tokens": 100})
    return preds, 100 * len(article_ids)


def _fake_metrics_fn(preds, article_ids, gold_df, union_map):
    """Métricas falsas: precision/recall altos si hay relaciones, 0 si no."""
    has = any(p["relations"] for p in preds)
    if has:
        return {"Precision_rel": 0.9, "Recall_rel": 0.6, "Precision_ent": 0.8,
                "Recall_ent": 0.8, "Polarity_acc": 0.9, "Act_acc": 0.8}
    return {"Precision_rel": 0.0, "Recall_rel": 0.0, "Precision_ent": 0.0,
            "Recall_ent": 0.0, "Polarity_acc": 0.0, "Act_acc": 0.0}


class FakeResp:
    def __init__(self, text): self.text = text


class FakeModels:
    # diagnose -> texto; propose -> diff que inserta 'BUENO' en el prompt
    def generate_content(self, model, contents, system=None):
        if "diagnosticador" in (system or ""):
            return FakeResp("1. faltan relaciones")
        return FakeResp(
            '{"artifact": "A", "diff": '
            '"<<<<<<< SEARCH\\nSEED\\n=======\\nSEED BUENO\\n>>>>>>> REPLACE"}'
        )


class FakeClient:
    def __init__(self): self.models = FakeModels()


def test_run_loop_improves_and_crowns_champion():
    with tempfile.TemporaryDirectory() as d:
        articles_df = pd.DataFrame({"article_id": ["x1", "x2", "x3", "x4"],
                                    "body": ["b"] * 4})
        gold_df = pd.DataFrame({"article_id": ["x1"]})
        splits = {"eval": ["x1", "x2", "x3"], "test": ["x4"]}
        seed = Genome(prompt_text="SEED")    # malo (sin 'BUENO')

        champ = run_loop(
            max_iter=4,
            budget_usd=999.0,
            subsample_k=2,
            championship_every=2,
            cross_every=99,
            archive_path=Path(d) / "arch.jsonl",
            best_path=Path(d) / "best.json",
            seed_genome=seed,
            articles_df=articles_df,
            gold_df=gold_df,
            splits=splits,
            client=FakeClient(),
            extract_fn=_fake_extract_fn,
            metrics_fn=_fake_metrics_fn,
            verbose=False,
        )
        # tras mutar, el hijo lleva 'BUENO' y debería ganar el championship
        assert "BUENO" in champ.prompt_text
        assert (Path(d) / "best.json").exists()


# ── extensiones del informe 2026-06-09 ───────────────────────────── #
class BadChildModels:
    """propose siempre empeora el prompt (quita 'BUENO')."""
    def generate_content(self, model, contents, system=None):
        if "diagnosticador" in (system or ""):
            return FakeResp("1. causa")
        return FakeResp(
            '{"artifact": "A", "diff": '
            '"<<<<<<< SEARCH\\nBUENO\\n=======\\nMALO\\n>>>>>>> REPLACE"}'
        )


class BadChildClient:
    def __init__(self): self.models = BadChildModels()


def test_cascade_gate_discards_bad_mutants_before_full_skirmish():
    with tempfile.TemporaryDirectory() as d:
        articles_df = pd.DataFrame({"article_id": ["x1", "x2", "x3", "x4"],
                                    "body": ["b"] * 4})
        gold_df = pd.DataFrame({"article_id": ["x1"]})
        splits = {"eval": ["x1", "x2", "x3"], "test": ["x4"]}
        seed = Genome(prompt_text="SEED BUENO")    # campeón fuerte

        run_loop(
            max_iter=2,
            budget_usd=999.0,
            subsample_k=2,
            gate_k=1,                  # gate activo (1 < 2)
            gate_epsilon=0.05,
            championship_every=99,
            cross_every=99,
            archive_path=Path(d) / "arch.jsonl",
            best_path=Path(d) / "best.json",
            seed_genome=seed,
            articles_df=articles_df,
            gold_df=gold_df,
            splits=splits,
            client=BadChildClient(),
            extract_fn=_fake_extract_fn,
            metrics_fn=_fake_metrics_fn,
            verbose=False,
        )
        arc = Archive(Path(d) / "arch.jsonl")
        children = [e for e in arc.all() if e.mutation_type == "diff_a"]
        assert children, "debería haber mutaciones registradas"
        for c in children:
            # descartados en gate 1: delta negativo registrado y ELO penalizado
            assert c.fitness_delta is not None and c.fitness_delta < 0
            assert c.elo < 1000.0


def test_multi_seed_adds_all_seeds_with_championship():
    with tempfile.TemporaryDirectory() as d:
        articles_df = pd.DataFrame({"article_id": ["x1", "x2"], "body": ["b"] * 2})
        gold_df = pd.DataFrame({"article_id": ["x1"]})
        splits = {"eval": ["x1"], "test": ["x2"]}
        seeds = [Genome(prompt_text="SEED BUENO"),
                 Genome(prompt_text="SEED BUENO", verify=True),
                 Genome(prompt_text="SEED BUENO", architecture="debate")]

        run_loop(
            max_iter=0,
            budget_usd=999.0,
            subsample_k=1,
            archive_path=Path(d) / "arch.jsonl",
            best_path=Path(d) / "best.json",
            seed_genomes=seeds,
            articles_df=articles_df,
            gold_df=gold_df,
            splits=splits,
            client=FakeClient(),
            extract_fn=_fake_extract_fn,
            metrics_fn=_fake_metrics_fn,
            verbose=False,
        )
        arc = Archive(Path(d) / "arch.jsonl")
        seeded = [e for e in arc.all() if e.mutation_type == "seed"]
        assert len(seeded) == 3
        assert all(e.championship_score is not None for e in seeded)
        archs = {e.genome.architecture for e in seeded}
        assert "debate" in archs


def test_run_loop_with_meta_policy_persists_posterior():
    with tempfile.TemporaryDirectory() as d:
        articles_df = pd.DataFrame({"article_id": ["x1", "x2", "x3", "x4"],
                                    "body": ["b"] * 4})
        gold_df = pd.DataFrame({"article_id": ["x1"]})
        splits = {"eval": ["x1", "x2", "x3"], "test": ["x4"]}
        seed = Genome(prompt_text="SEED")
        policy_path = Path(d) / "policy.json"

        champ = run_loop(
            max_iter=4,
            budget_usd=999.0,
            subsample_k=2,
            championship_every=2,
            use_meta_policy=True,
            meta_policy_path=policy_path,
            archive_path=Path(d) / "arch.jsonl",
            best_path=Path(d) / "best.json",
            seed_genome=seed,
            articles_df=articles_df,
            gold_df=gold_df,
            splits=splits,
            client=FakeClient(),
            extract_fn=_fake_extract_fn,
            metrics_fn=_fake_metrics_fn,
            verbose=False,
        )
        assert champ is not None
        assert policy_path.exists()     # el posterior del bandit se persiste


# ── resguardos de seguridad (informe 2026-06-09 §safeguards) ─────── #
class NoopModels:
    """propose siempre devuelve JSON inválido → noop."""
    def generate_content(self, model, contents, system=None):
        if "diagnosticador" in (system or ""):
            return FakeResp("1. causa X")
        return FakeResp("esto no es json")


class NoopClient:
    def __init__(self): self.models = NoopModels()


def test_consecutive_noop_warning_printed_to_stderr():
    with tempfile.TemporaryDirectory() as d:
        articles_df = pd.DataFrame({"article_id": ["x1", "x2", "x3"], "body": ["b"] * 3})
        gold_df = pd.DataFrame({"article_id": ["x1"]})
        splits = {"eval": ["x1", "x2"], "test": ["x3"]}
        seed = Genome(prompt_text="SEED BUENO")

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            run_loop(
                max_iter=6,
                budget_usd=999.0,
                subsample_k=1,
                championship_every=99,
                archive_path=Path(d) / "arch.jsonl",
                best_path=Path(d) / "best.json",
                seed_genome=seed,
                articles_df=articles_df,
                gold_df=gold_df,
                splits=splits,
                client=NoopClient(),
                extract_fn=_fake_extract_fn,
                metrics_fn=_fake_metrics_fn,
                verbose=True,
            )
        assert "noops consecutivos" in buf.getvalue()


def _zero_metrics_fn(preds, article_ids, gold_df, union_map):
    return {"Precision_rel": 0.0, "Recall_rel": 0.0, "Precision_ent": 0.0,
            "Recall_ent": 0.0, "Polarity_acc": 0.0, "Act_acc": 0.0}


def test_zero_extraction_seed_warns_to_stderr():
    with tempfile.TemporaryDirectory() as d:
        articles_df = pd.DataFrame({"article_id": ["x1", "x2"], "body": ["b"] * 2})
        gold_df = pd.DataFrame({"article_id": ["x1"]})
        splits = {"eval": ["x1"], "test": ["x2"]}
        seed = Genome(prompt_text="SEED BUENO")

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            run_loop(
                max_iter=0,
                budget_usd=999.0,
                subsample_k=1,
                archive_path=Path(d) / "arch.jsonl",
                best_path=Path(d) / "best.json",
                seed_genome=seed,
                articles_df=articles_df,
                gold_df=gold_df,
                splits=splits,
                client=FakeClient(),
                extract_fn=_fake_extract_fn,
                metrics_fn=_zero_metrics_fn,
                verbose=True,
            )
        assert "0 relaciones" in buf.getvalue()
