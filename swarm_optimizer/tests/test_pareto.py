from swarm_optimizer.pareto import ParetoArchive, _dominates, ParetoEntry


def _e(P, R, exp=0, _id=0):
    return ParetoEntry(id=_id, genome={}, P=P, R=R, expansions=exp)


def test_dominates_strict_and_ties():
    assert _dominates(_e(0.9, 0.9), _e(0.8, 0.8))
    assert _dominates(_e(0.9, 0.8), _e(0.8, 0.8))
    assert not _dominates(_e(0.8, 0.8), _e(0.8, 0.8))
    assert not _dominates(_e(0.9, 0.7), _e(0.8, 0.8))


def test_add_and_frontier_excludes_dominated():
    a = ParetoArchive()
    a.add({}, 0.93, 0.90)
    a.add({}, 0.94, 0.83)
    a.add({}, 0.91, 0.89)
    front = {(round(e.P, 2), round(e.R, 2)) for e in a.frontier()}
    assert (0.93, 0.90) in front and (0.94, 0.83) in front
    assert (0.91, 0.89) not in front
    assert len(a.all()) == 3


def test_pick_to_expand_least_expanded_then_lowest_id():
    a = ParetoArchive()
    e0 = a.add({}, 0.93, 0.90)
    e1 = a.add({}, 0.94, 0.83)
    a.mark_expanded(e0.id)
    assert a.pick_to_expand().id == e1.id
    a.mark_expanded(e1.id)
    assert a.pick_to_expand().id == e0.id


def test_json_roundtrip():
    a = ParetoArchive()
    a.add({"prompt_text": "x"}, 0.9, 0.8, parent_id=None)
    a.add({"prompt_text": "y"}, 0.95, 0.7, parent_id=0)
    b = ParetoArchive.from_json(a.to_json())
    assert len(b.all()) == 2
    assert b.add({}, 0.5, 0.5).id == 2


import pandas as pd
from swarm_optimizer.diagnostics import diagnostics


def test_diagnostics_fn_and_fp_distractor():
    articles = pd.DataFrame([
        {"article_id": "A1", "es_distractor": False, "dureza": "oblicua"},
        {"article_id": "D1", "es_distractor": True, "dureza": "mixta"},
    ])
    gold = pd.DataFrame([
        {"article_id": "A1", "u_from": "U1", "u_to": "U2", "act_type": "endorses",
         "evidence_quote": "valoró el compromiso de B"},
    ])
    unions = {
        "A1": {"U1": {"union_id": "U1", "type": "roster_actor",
                      "canonical_names": ["Actor A"], "surfaces": ["A"]},
               "U2": {"union_id": "U2", "type": "roster_actor",
                      "canonical_names": ["Actor B"], "surfaces": ["B"]}},
        "D1": {"U1": {"union_id": "U1", "type": "roster_actor",
                      "canonical_names": ["Actor C"], "surfaces": ["C"]},
               "U2": {"union_id": "U2", "type": "roster_actor",
                      "canonical_names": ["Actor D"], "surfaces": ["D"]}},
    }
    preds = [
        {"article_id": "A1", "entities": [], "relations": []},
        {"article_id": "D1", "entities": [], "relations": [
            {"from_entity": "Actor C", "to_entity": "Actor D", "act_type": "endorses",
             "evidence_quote": "x"}]},
    ]
    d = diagnostics(["A1", "D1"], preds, gold, unions, articles)
    assert len(d["fn"]) == 1 and d["fn"][0]["dureza"] == "oblicua"
    assert d["fn_by_dureza"].get("oblicua") == 1
    assert d["fp_distractor"] == 1 and d["fp_total"] >= 1
    assert d["fp"][0]["es_distractor"] is True


def test_diagnostics_breaks_down_fn_fp_by_registro():
    articles = pd.DataFrame([
        {"article_id": "F1", "es_distractor": False, "dureza": "directa", "medio": "Emol"},
        {"article_id": "I1", "es_distractor": False, "dureza": "directa", "medio": "La Cuarta"},
    ])
    gold = pd.DataFrame([
        {"article_id": "F1", "u_from": "U1", "u_to": "U2", "act_type": "endorses",
         "evidence_quote": "respaldó a B"},
        {"article_id": "I1", "u_from": "U1", "u_to": "U2", "act_type": "attacks",
         "evidence_quote": "criticó a B"},
    ])
    union = {"U1": {"union_id": "U1", "type": "roster_actor",
                    "canonical_names": ["Actor A"], "surfaces": ["A"]},
             "U2": {"union_id": "U2", "type": "roster_actor",
                    "canonical_names": ["Actor B"], "surfaces": ["B"]}}
    unions = {"F1": union, "I1": union}
    preds = [  # ambos artículos pierden su relación gold (FN) y agregan una inventada (FP)
        {"article_id": "F1", "entities": [], "relations": [
            {"from_entity": "Actor A", "to_entity": "Nadie", "act_type": "attacks",
             "evidence_quote": "x"}]},
        {"article_id": "I1", "entities": [], "relations": []},
    ]
    d = diagnostics(["F1", "I1"], preds, gold, unions, articles)
    assert d["fn_by_registro"] == {"formal": 1, "informal": 1}
    assert d["fp_by_registro"]["formal"] == 1 and d["fp_by_registro"]["informal"] == 0


def test_pareto_entry_stores_split_dataset_for_consistent_diag():
    a = ParetoArchive()
    a.add({}, 0.9, 0.8, split="train", dataset="v1", preds_path="p.json")
    e = ParetoArchive.from_json(a.to_json()).all()[0]
    assert e.split == "train" and e.dataset == "v1" and e.preds_path == "p.json"


# ── multi-gradiente (Quality-Diversity): ejes directed/undirected ────────────── #

def _sm(dP, dR, uP, uR):
    return {"directed": {"P": dP, "R": dR}, "undirected": {"P": uP, "R": uR}}


def test_gradient_value_parses_axis_and_metric():
    e = ParetoEntry(id=0, genome={}, P=0.9, R=0.8, subset_metrics=_sm(0.9, 0.7, 0.95, 0.85))
    assert ParetoArchive.gradient_value(e, "directed_P") == 0.9
    assert ParetoArchive.gradient_value(e, "undirected_R") == 0.85
    assert ParetoArchive.gradient_value(ParetoEntry(0, {}, 0, 0), "directed_P") is None


def test_gradient_champion_picks_max_on_axis():
    a = ParetoArchive()
    a.add({}, 0.90, 0.90, subset_metrics=_sm(0.9, 0.9, 0.95, 0.95))   # gana directed
    a.add({}, 0.70, 0.70, subset_metrics=_sm(0.5, 0.5, 0.99, 0.99))   # gana undirected, dominado global
    assert a.gradient_champion("directed_P").id == 0
    assert a.gradient_champion("undirected_R").id == 1


def test_frontier_include_champions_keeps_dominated_axis_winner():
    a = ParetoArchive()
    a.add({}, 0.90, 0.90, subset_metrics=_sm(0.9, 0.9, 0.92, 0.9))
    dom = a.add({}, 0.70, 0.70, subset_metrics=_sm(0.5, 0.5, 0.99, 0.5))  # dominado en (P,R)
    assert dom.id not in {e.id for e in a.frontier()}                      # fuera del frente global
    assert dom.id in {e.id for e in a.frontier(include_champions=True)}    # campeón undirected_P


def test_pick_prefers_explicit_gradient():
    a = ParetoArchive()
    a.add({}, 0.90, 0.90, subset_metrics=_sm(0.9, 0.9, 0.92, 0.92))
    e1 = a.add({}, 0.70, 0.70, subset_metrics=_sm(0.5, 0.5, 0.99, 0.99))
    assert a.pick_to_expand(prefer_gradient="undirected_R").id == e1.id


def test_gradient_fields_survive_json_roundtrip():
    a = ParetoArchive()
    a.add({}, 0.9, 0.8, gradient_tags=["undirected_R"], subset_metrics=_sm(0.9, 0.8, 0.95, 0.86))
    e = ParetoArchive.from_json(a.to_json()).all()[0]
    assert e.gradient_tags == ["undirected_R"]
    assert e.subset_metrics["undirected"]["R"] == 0.86


def test_legacy_json_without_gradient_fields_loads():
    legacy = '{"next_id": 1, "entries": [{"id": 0, "genome": {}, "P": 0.9, "R": 0.8}]}'
    e = ParetoArchive.from_json(legacy).all()[0]
    assert e.gradient_tags == [] and e.subset_metrics == {} and e.per_instance_scores == {}


# ── selección GEPA por win-count por instancia ───────────────────────────────── #

def test_win_counts_credits_per_instance_best():
    a = ParetoArchive()
    a.add({}, 0.9, 0.9, per_instance_scores={"x": 1.0, "y": 0.2, "z": 0.5})
    a.add({}, 0.7, 0.7, per_instance_scores={"x": 0.3, "y": 0.9, "z": 0.5})
    wc = a.win_counts()
    assert wc[0] == 2   # gana x y z (empate en z → id menor)
    assert wc[1] == 1   # gana y


def test_pick_gepa_samples_proportional_and_is_seeded():
    import random
    a = ParetoArchive()
    a.add({}, 0.9, 0.9, per_instance_scores={f"a{i}": 1.0 for i in range(9)})  # 9 wins
    a.add({}, 0.9, 0.9, per_instance_scores={"b0": 1.0})                        # 1 win (no dominado: mismo P,R)
    picks = [a.pick_to_expand_gepa(random.Random(s)).id for s in range(200)]
    n0 = picks.count(0)
    assert 150 < n0 < 200                       # ~9:1 hacia id0, pero no determinista
    assert a.pick_to_expand_gepa(random.Random(7)).id == a.pick_to_expand_gepa(random.Random(7)).id


def test_pick_gepa_uniform_without_scores():
    import random
    a = ParetoArchive()
    a.add({}, 0.9, 0.8)   # sin per_instance_scores
    a.add({}, 0.8, 0.9)
    assert a.pick_to_expand_gepa(random.Random(0)) is not None   # no crashea, cae a uniforme


# ── merge por componente: linaje + elegibilidad ──────────────────────────────── #

def test_lineage_ancestors_and_lca():
    a = ParetoArchive()
    a.add({}, 0.8, 0.8)                 # id0 (raíz)
    a.add({}, 0.85, 0.82, parent_id=0)  # id1
    a.add({}, 0.86, 0.84, parent_id=0)  # id2
    a.add({}, 0.9, 0.85, parent_id=1)   # id3 (nieto vía id1)
    assert a.ancestors(3) == [1, 0]
    assert a.lowest_common_ancestor(3, 2) == 0
    assert a.lowest_common_ancestor(1, 2) == 0


def test_merge_pairs_requires_common_ancestor_and_both_beat_it():
    a = ParetoArchive()
    a.add({}, 0.85, 0.85)                 # id0 ancestro común, f05≈0.85
    a.add({}, 0.90, 0.86, parent_id=0)    # id1 mejor que id0
    a.add({}, 0.87, 0.90, parent_id=0)    # id2 mejor que id0 (no dominado vs id1)
    pairs = a.merge_pairs()
    assert pairs, "id1 e id2 deberían ser elegibles para merge"
    pa, pb, anc = pairs[0]
    assert {pa.id, pb.id} == {1, 2} and anc.id == 0


def test_merge_pairs_excludes_ancestor_descendant_relation():
    a = ParetoArchive()
    a.add({}, 0.85, 0.85)                 # id0
    a.add({}, 0.90, 0.88, parent_id=0)    # id1
    a.add({}, 0.93, 0.90, parent_id=1)    # id2 desciende de id1 → no se fusionan entre sí
    pairs = a.merge_pairs()
    assert all(not ({p[0].id, p[1].id} == {1, 2}) for p in pairs)


# ── métrica de selección precision-first ─────────────────────────────────────── #

def test_selection_score_recall_floor_disqualifies_below():
    from swarm_optimizer.fitness import selection_score
    valid = selection_score(0.90, 0.857)          # sobre el piso 0.80
    collapsed = selection_score(0.95, 0.45)        # alta P pero recall colapsado (caso id4/id5)
    assert valid > 0 > collapsed                   # el colapso queda SIEMPRE bajo los válidos
    # entre descalificados, ordena por recall (no se cancelan en un plano)
    assert selection_score(0.95, 0.45) < selection_score(0.95, 0.79)


def test_selection_score_penalizes_distractor_hallucinations():
    from swarm_optimizer.fitness import selection_score
    clean = selection_score(0.90, 0.86, distractor_fp=0)
    halluc = selection_score(0.90, 0.86, distractor_fp=5)
    assert clean > halluc                          # inventar en distractores baja el score


def test_selection_score_beta_tilts_to_precision():
    from swarm_optimizer.fitness import selection_score
    # con β menor, un genoma más preciso supera a uno con más recall pero igual error
    precise = selection_score(0.95, 0.82, beta=0.33)
    recally = selection_score(0.86, 0.90, beta=0.33)
    assert precise > recally
