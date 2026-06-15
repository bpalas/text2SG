"""Tests de canonicalize_union y su efecto en compute_metrics.

Caso real que motivó el fix (artículo 5363c09d): el union tiene a Bachelet dos veces
(U1 'Michelle Bachelet Jeria' con surfaces, U6 'Presidenta Bachelet' sin surfaces) y la
relación gold está anclada al duplicado U6. Sin el fix, una extracción correcta
'Amnistía -> Bachelet' resolvía a U1, generando FP + FN simultáneos.
"""
import pandas as pd
import pytest

from swarm_optimizer.rubric import canonicalize_union, compute_metrics


def _union_bachelet():
    return {
        "U1": {
            "type": "roster_actor",
            "canonical_names": ["Michelle Bachelet Jeria"],
            "surfaces": ["Bachelet", "Presidenta Bachelet"],
        },
        "U2": {
            "type": "institutional_actor",
            "canonical_names": ["La Moneda"],
            "surfaces": ["La Moneda", "Palacio de Gobierno"],
        },
        "U3": {
            "type": "non_roster_actor",
            "canonical_names": ["Amnistía Internacional"],
            "surfaces": ["Amnistía Internacional", "Amnesty"],
        },
        "U4": {"type": "non_roster_actor", "canonical_names": ["Amnesty"], "surfaces": []},
        "U5": {"type": "non_roster_actor", "canonical_names": ["Palacio de Gobierno"], "surfaces": []},
        "U6": {"type": "non_roster_actor", "canonical_names": ["Presidenta Bachelet"], "surfaces": []},
    }


def test_merges_duplicates_into_surfaced_entry():
    clean, alias = canonicalize_union(_union_bachelet())
    assert alias == {"U4": "U3", "U5": "U2", "U6": "U1"}
    assert set(clean) == {"U1", "U2", "U3"}
    # el sobreviviente absorbe el canonical del duplicado
    assert "Presidenta Bachelet" in clean["U1"]["canonical_names"]


def test_fuzzy_canonical_merge_jara():
    union = {
        "U11": {"type": "roster_actor",
                "canonical_names": ["Jeannette Jara Román"],
                "surfaces": ["Jeannette Jara"]},
        "U12": {"type": "roster_actor",
                "canonical_names": ["Jeanette Jara Román"],
                "surfaces": ["Jeannette Jara", "la carta oficialista", "una comunista", "la comunista"]},
    }
    clean, alias = canonicalize_union(union)
    # sobrevive U12 (más surfaces)
    assert alias == {"U11": "U12"}
    assert set(clean) == {"U12"}


def test_no_merge_distinct_actors():
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Evelyn Matthei Fornet"],
               "surfaces": ["Matthei"]},
        "U2": {"type": "roster_actor", "canonical_names": ["José Antonio Kast Adriasola"],
               "surfaces": ["Kast"]},
    }
    clean, alias = canonicalize_union(union)
    assert alias == {}
    assert set(clean) == {"U1", "U2"}


def test_chain_resolution():
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric Font"],
               "surfaces": ["Boric", "Gabriel Boric", "el Presidente"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"],
               "surfaces": ["Gabriel Boric"]},
        "U3": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"], "surfaces": []},
    }
    clean, alias = canonicalize_union(union)
    assert set(clean) == {"U1"}
    assert all(target == "U1" for target in alias.values())


def test_compute_metrics_remaps_gold_uids():
    """La relación gold anclada al duplicado U6 debe acreditarse al extraer U3->U1."""
    union = _union_bachelet()
    gold = pd.DataFrame([
        {"article_id": "a1", "u_from": "U3", "u_to": "U6",
         "act_type": "calls_on", "polarity": "neutral"},
    ])
    preds = [{
        "article_id": "a1",
        "entities": [{"name": "Amnistía Internacional"}, {"name": "Michelle Bachelet Jeria"}],
        "relations": [{
            "from_entity": "Amnistía Internacional",
            "to_entity": "Michelle Bachelet Jeria",
            "act_type": "calls_on", "polarity": "neutral",
            "evidence_quote": "x" * 30,
        }],
    }]
    m = compute_metrics(preds, ["a1"], gold, {"a1": union})
    assert m["Precision_rel"] == 1.0, "la extracción correcta no debe ser FP"
    assert m["Recall_rel"] == 1.0, "el gold remapeado debe quedar cubierto"
    assert m["Polarity_acc"] == 1.0


def test_compute_metrics_entity_denominator_shrinks():
    """Los duplicados no deben inflar el denominador de Recall_ent."""
    union = _union_bachelet()
    preds = [{
        "article_id": "a1",
        "entities": [{"name": "Michelle Bachelet Jeria"}, {"name": "La Moneda"},
                     {"name": "Amnistía Internacional"}],
        "relations": [],
    }]
    gold = pd.DataFrame([], columns=["article_id", "u_from", "u_to", "act_type", "polarity"])
    m = compute_metrics(preds, ["a1"], gold, {"a1": union})
    assert m["Recall_ent"] == 1.0, "3 entidades reales emitidas / 3 únicas = 1.0"
