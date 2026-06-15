import json
import pandas as pd
from pathlib import Path

from swarm_optimizer.synth_data import (
    load_synth_unions, load_synth_split, load_synth_articles, load_synth_gold,
)


def _make_fixture(base: Path) -> Path:
    d = base / "mini"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"article_id": "A1", "title": "t", "body": "Boric criticó a Matthei.",
         "dominio": "politica", "registro": "formal", "es_distractor": False},
        {"article_id": "A2", "title": "t", "body": "Matthei respondió a Boric.",
         "dominio": "politica", "registro": "coloquial", "es_distractor": False},
        {"article_id": "A3", "title": "t", "body": "El club ganó la copa.",
         "dominio": "futbol", "registro": "formal", "es_distractor": True},
        {"article_id": "A4", "title": "t", "body": "Kast habló en el congreso.",
         "dominio": "politica", "registro": "formal", "es_distractor": False},
    ]).to_parquet(d / "articles.parquet", index=False)
    pd.DataFrame([
        {"article_id": "A1", "u_from": "U1", "u_to": "U2", "act_type": "accuses",
         "polarity": "negative", "issue": None, "evidence_quote": "Boric criticó a Matthei"},
    ]).to_parquet(d / "gold.parquet", index=False)
    unions = {"A1": {
        "U1": {"union_id": "U1", "type": "roster_actor",
               "canonical_names": ["Gabriel Boric"], "surfaces": ["Boric"]},
        "U2": {"union_id": "U2", "type": "roster_actor",
               "canonical_names": ["Evelyn Matthei"], "surfaces": ["Matthei"]}}}
    (d / "unions.json").write_text(json.dumps(unions), encoding="utf-8")
    return base


def test_load_synth_unions(tmp_path):
    _make_fixture(tmp_path)
    u = load_synth_unions("mini", base=tmp_path)
    assert u["A1"]["U1"]["canonical_names"] == ["Gabriel Boric"]


def test_load_synth_articles_and_gold(tmp_path):
    _make_fixture(tmp_path)
    assert len(load_synth_articles("mini", base=tmp_path)) == 4
    assert len(load_synth_gold("mini", base=tmp_path)) == 1


def test_load_synth_split_deterministic_and_stratified(tmp_path):
    _make_fixture(tmp_path)
    s1 = load_synth_split("mini", seed=42, base=tmp_path)
    s2 = load_synth_split("mini", seed=42, base=tmp_path)
    assert s1 == s2
    assert set(s1["train"]).isdisjoint(s1["test"])
    assert set(s1["train"]) | set(s1["test"]) == {"A1", "A2", "A3", "A4"}
    assert len(s1["test"]) >= 3


import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from synth_agent_eval import do_dump, do_score, _load_preds, do_collect   # noqa: E402
from swarm_optimizer.genome import Genome, AnalysisConfig   # noqa: E402


def test_do_dump_includes_analysis_block(tmp_path):
    _make_fixture(tmp_path)
    arts = load_synth_articles("mini", base=tmp_path)
    unions = load_synth_unions("mini", base=tmp_path)
    g = Genome(prompt_text="INSTRUCCIONES", architecture="given_entities",
               analysis=AnalysisConfig())
    rows = do_dump(g, ["A1"], arts, unions)
    assert len(rows) == 1 and rows[0]["article_id"] == "A1"
    assert "=== ANÁLISIS DE ACTORES ===" in rows[0]["prompt"]
    assert "INSTRUCCIONES" in rows[0]["prompt"]


def test_do_dump_appends_verify_when_flagged(tmp_path):
    _make_fixture(tmp_path)
    arts = load_synth_articles("mini", base=tmp_path)
    unions = load_synth_unions("mini", base=tmp_path)
    g = Genome(prompt_text="X", architecture="given_entities", verify=True)
    rows = do_dump(g, ["A1"], arts, unions)
    assert "VERIFICACIÓN" in rows[0]["prompt"]


def test_do_score_perfect_extraction(tmp_path):
    _make_fixture(tmp_path)
    arts = load_synth_articles("mini", base=tmp_path)
    gold = load_synth_gold("mini", base=tmp_path)
    unions = load_synth_unions("mini", base=tmp_path)
    g = Genome(prompt_text="X", architecture="given_entities")
    preds = [{"article_id": "A1", "entities": [], "relations": [
        {"from_entity": "Boric", "to_entity": "Matthei", "act_type": "accuses",
         "polarity": "negative", "issue": "x",
         "evidence_quote": "Boric criticó a Matthei"}]}]
    ids = ["A1", "A2", "A3", "A4"]
    m, predictions = do_score(g, ids, arts, gold, unions, preds)
    assert m["Precision_rel"] == 1.0
    assert m["Recall_rel"] == 1.0


def test_do_dump_handles_duplicate_article_ids():
    # El sintético real tiene article_id duplicados (pilot 1, v1 5): el lookup por
    # dict (first-wins) no debe romper, a diferencia de set_index().loc[].
    arts = pd.DataFrame([
        {"article_id": "D1", "body": "Primero.", "dominio": "politica"},
        {"article_id": "D1", "body": "Duplicado.", "dominio": "politica"},
        {"article_id": "D2", "body": "Otro.", "dominio": "futbol"},
    ])
    g = Genome(prompt_text="X", architecture="given_entities", analysis=AnalysisConfig())
    rows = do_dump(g, ["D1", "D2"], arts, {})
    assert [r["article_id"] for r in rows] == ["D1", "D2"]
    assert "Primero." in rows[0]["prompt"]   # first-wins


def test_load_synth_split_dedups_duplicate_ids(tmp_path):
    # Un article_id duplicado NO debe filtrarse a train Y test (rompería el held-out).
    d = tmp_path / "dup"
    d.mkdir()
    pd.DataFrame([
        {"article_id": "X1", "body": "a", "dominio": "politica", "registro": "formal"},
        {"article_id": "X1", "body": "b", "dominio": "politica", "registro": "formal"},
        {"article_id": "X2", "body": "c", "dominio": "futbol", "registro": "formal"},
        {"article_id": "X3", "body": "d", "dominio": "politica", "registro": "coloquial"},
    ]).to_parquet(d / "articles.parquet", index=False)
    s = load_synth_split("dup", seed=1, base=tmp_path)
    assert set(s["train"]).isdisjoint(s["test"])
    allids = s["train"] + s["test"]
    assert sorted(allids) == ["X1", "X2", "X3"]   # X1 deduplicado
    assert len(allids) == len(set(allids))


def test_load_preds_accepts_array_and_jsonl(tmp_path):
    arr = tmp_path / "a.json"
    arr.write_text('[{"article_id":"A1","entities":[],"relations":[]}]', encoding="utf-8")
    assert _load_preds(arr)[0]["article_id"] == "A1"
    jl = tmp_path / "b.jsonl"
    jl.write_text('{"article_id":"A1","relations":[]}\n{"article_id":"A2","relations":[]}\n',
                  encoding="utf-8")
    out = _load_preds(jl)
    assert len(out) == 2 and out[1]["article_id"] == "A2"


def test_do_collect_merges_redo_and_flags_failed_batches():
    expected = ["A1", "A2", "A3", "A4", "A5"]
    good = [{"article_id": "A1", "entities": [], "relations": [{"x": 1}]},
            {"article_id": "A2", "entities": [], "relations": []}]
    failed = [{"article_id": "A3", "entities": [], "relations": []},
              {"article_id": "A4", "entities": [], "relations": []}]   # lote 100% vacio
    dup = [{"article_id": "A1", "entities": [], "relations": []}]      # A1 vacio en otro archivo
    merged, redo, report = do_collect(expected, [("good", good), ("failed", failed), ("dup", dup)])
    assert any(m["article_id"] == "A1" and len(m["relations"]) == 1 for m in merged)  # gana no-vacio
    assert "A2" in {m["article_id"] for m in merged}    # vacio legitimo (lote mixto), cubierto
    assert set(redo) == {"A3", "A4", "A5"}              # lote fallido + ausente
    assert report["covered"] == 2


def test_do_collect_tolerates_malformed_entries():
    merged, redo, report = do_collect(
        ["A1"], [("f", [{"article_id": "A1", "relations": [{"x": 1}]}, "basura", {"no_id": 1}])])
    assert len(merged) == 1 and merged[0]["entities"] == []   # entities faltante -> []


def test_do_collect_does_not_redo_all_distractor_batch():
    # un lote 100% distractores (0 relaciones legítimas) NO debe marcarse como fallido.
    expected = ["D1", "D2"]
    f = [{"article_id": "D1", "entities": [], "relations": []},
         {"article_id": "D2", "entities": [], "relations": []}]
    merged, redo, report = do_collect(expected, [("dis", f)], distractor_ids={"D1", "D2"})
    assert redo == []                       # distractores cubiertos, sin redo
    assert report["covered"] == 2
