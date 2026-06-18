"""
Tests for text2sg.rubric entity matching functions.
"""
from text2sg.rubric import normalize, match_entity

# union sintético para tests — simula un YAML parseado
SAMPLE_UNION = {
    "U1": {
        "type": "roster_actor",
        "canonical_names": ["Gabriel Boric Font"],
        "surfaces": ["Gabriel Boric", "Boric", "el Presidente"],
    },
    "U2": {
        "type": "institutional_actor",
        "canonical_names": ["Gobierno de Chile"],
        "surfaces": ["el gobierno", "Gobierno"],
    },
    "U3": {
        "type": "non_roster_actor",
        "canonical_names": ["Ericka Farías"],
        "surfaces": ["delegada Farías", "Ericka Farías", "la delegada"],
    },
    "U_NIL": {
        "type": "NIL",
        "canonical_names": ["Soto"],
        "surfaces": ["Soto"],
    },
}


# --- normalize ---


def test_normalize_lowercase():
    assert normalize("Boric") == "boric"


def test_normalize_strips_accents():
    assert normalize("García") == "garcia"


def test_normalize_strips_whitespace():
    assert normalize("  Boric  ") == "boric"


# --- match_entity: exact ---


def test_match_exact_surface():
    uid, is_nil = match_entity("Gabriel Boric", SAMPLE_UNION)
    assert uid == "U1"
    assert is_nil is False


def test_match_exact_canonical():
    uid, is_nil = match_entity("Gabriel Boric Font", SAMPLE_UNION)
    assert uid == "U1"
    assert is_nil is False


def test_match_exact_surface_institutional():
    uid, is_nil = match_entity("el gobierno", SAMPLE_UNION)
    assert uid == "U2"
    assert is_nil is False


# --- match_entity: fuzzy ---


def test_match_fuzzy_partial_name():
    uid, is_nil = match_entity("Ericka Farias", SAMPLE_UNION)  # sin tilde
    assert uid == "U3"
    assert is_nil is False


def test_match_fuzzy_typo():
    uid, is_nil = match_entity("Gabriel Borick", SAMPLE_UNION)  # typo leve
    assert uid == "U1"
    assert is_nil is False


# --- match_entity: NIL y no-match ---


def test_match_nil_entity():
    uid, is_nil = match_entity("Soto", SAMPLE_UNION)
    assert uid == "U_NIL"
    assert is_nil is True


def test_match_no_match():
    uid, is_nil = match_entity("Donald Trump", SAMPLE_UNION)
    assert uid is None
    assert is_nil is False


# ── compute_metrics tests ─────────────────────────────────────────────────── #

import pandas as pd
from text2sg.rubric import compute_metrics
from text2sg.config import Config

# Predicciones perfectas
PERFECT_PREDS = [
    {
        "article_id": "art1",
        "entities": [
            {"name": "Gabriel Boric", "type": "roster_actor"},
            {"name": "el gobierno", "type": "institutional_actor"},
        ],
        "relations": [
            {
                "from_entity": "Gabriel Boric",
                "to_entity": "el gobierno",
                "act_type": "endorses",
                "polarity": "positive",
                "issue": "government_management",
                "evidence_quote": "Boric apoyó al gobierno",
            }
        ],
    }
]

GOLD_DF_PERFECT = pd.DataFrame([{
    "article_id": "art1",
    "u_from": "U1", "u_to": "U2",
    "act_type": "endorses", "polarity": "positive",
    "issue": "government_management", "evidence_quote": "x",
}])

UNION_MAP = {"art1": SAMPLE_UNION}


def test_compute_metrics_perfect():
    m = compute_metrics(PERFECT_PREDS, ["art1"], GOLD_DF_PERFECT, UNION_MAP)
    assert m["F1_rel"] == 1.0
    assert m["Polarity_acc"] == 1.0
    assert m["Act_acc"] == 1.0


def test_compute_metrics_empty_predictions():
    empty_preds = [{"article_id": "art1", "entities": [], "relations": []}]
    m = compute_metrics(empty_preds, ["art1"], GOLD_DF_PERFECT, UNION_MAP)
    assert m["F1_rel"] == 0.0
    assert m["F1_ent"] == 0.0


def test_compute_metrics_wrong_polarity():
    wrong_preds = [
        {
            "article_id": "art1",
            "entities": [
                {"name": "Gabriel Boric", "type": "roster_actor"},
                {"name": "el gobierno", "type": "institutional_actor"},
            ],
            "relations": [
                {
                    "from_entity": "Gabriel Boric",
                    "to_entity": "el gobierno",
                    "act_type": "endorses",
                    "polarity": "negative",   # EQUIVOCADO
                    "issue": "government_management",
                    "evidence_quote": "Boric criticó al gobierno",
                }
            ],
        }
    ]
    m = compute_metrics(wrong_preds, ["art1"], GOLD_DF_PERFECT, UNION_MAP)
    assert m["F1_rel"] == 1.0       # la díada es correcta
    assert m["Polarity_acc"] == 0.0  # pero el signo está mal


def test_compute_metrics_reversed_direction_is_fp_directed_tp_undirected():
    """Dirección invertida (gold U1->U2, predicho U2->U1): cuenta como FP+FN en directed
    pero TP en undirected. El gap directed→undirected aísla el costo de la dirección."""
    reversed_preds = [{
        "article_id": "art1",
        "entities": [{"name": "Gabriel Boric", "type": "roster_actor"},
                     {"name": "el gobierno", "type": "institutional_actor"}],
        "relations": [{
            "from_entity": "el gobierno",   # invertido respecto al gold U1->U2
            "to_entity": "Gabriel Boric",
            "act_type": "endorses", "polarity": "positive",
            "issue": "government_management", "evidence_quote": "x",
        }],
    }]
    m = compute_metrics(reversed_preds, ["art1"], GOLD_DF_PERFECT, UNION_MAP)
    assert m["Precision_rel"] == 0.0 and m["Recall_rel"] == 0.0          # directed: error doble
    assert m["Precision_rel_undirected"] == 1.0 and m["Recall_rel_undirected"] == 1.0  # par correcto


def test_compute_metrics_undirected_perfect_when_directed_perfect():
    m = compute_metrics(PERFECT_PREDS, ["art1"], GOLD_DF_PERFECT, UNION_MAP)
    assert m["F1_rel_undirected"] == 1.0   # sin inversiones, undirected == directed


def test_score_integrates_metrics():
    cfg = Config(prompt_text="x")
    m = {"F1_rel": 1.0, "Polarity_acc": 1.0, "F1_ent": 1.0, "Act_acc": 1.0}
    assert cfg.score(m) == 1.0
