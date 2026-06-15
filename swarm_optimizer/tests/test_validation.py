from swarm_optimizer.genome import ValidationConfig
from swarm_optimizer.validation import apply_validation, _shares_token

BODY = "Boric fue criticado por Matthei durante la sesión. Kast respaldó la moción."


def _rel(frm, to, act="attacks", pol="negative", quote="x"):
    return {"from_entity": frm, "to_entity": to, "act_type": act,
            "polarity": pol, "issue": "x", "evidence_quote": quote}


def test_substring_filter_drops_unquoted_relations():
    vc = ValidationConfig(require_evidence_substring=True, min_quote_len=0,
                          normalize_passive_direction=False)
    parsed = {"entities": [], "relations": [
        _rel("Matthei", "Boric", quote="criticado por Matthei"),   # substring real
        _rel("Kast", "Boric", quote="frase inventada que no existe"),  # no substring
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1
    assert out["relations"][0]["from_entity"] == "Matthei"


def test_min_quote_len_drops_short_quotes():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=10,
                          normalize_passive_direction=False)
    parsed = {"entities": [], "relations": [_rel("A", "B", quote="corta")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"] == []


def test_passive_direction_swaps_from_to():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=True)
    # El modelo puso la dirección al revés: patient antes de 'por', agent después
    parsed = {"entities": [], "relations": [
        _rel("Boric", "Matthei", quote="Boric fue criticado por Matthei")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["from_entity"] == "Matthei"
    assert out["relations"][0]["to_entity"] == "Boric"


def test_dedup_collapses_identical_triples():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False, dedup=True)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", quote="q1"),
        _rel("A", "B", "attacks", quote="q2"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1


def test_dedup_collapses_same_pair_different_act_type():
    # P-008: la rúbrica matchea solo por par — el segundo act_type del mismo
    # (from, to) es FP mecánico y debe descartarse. Se conserva la primera.
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False, dedup=True)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", quote="q1"),
        _rel("A", "B", "accuses", quote="q2"),
        _rel("B", "A", "attacks", quote="q3"),  # par distinto (dirección) — sobrevive
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 2
    assert out["relations"][0]["act_type"] == "attacks"
    assert out["relations"][1]["from_entity"] == "B"


def test_seed_allowed_act_types_excludes_co_occurs():
    # P-001: la semilla filtra co_occurs por defecto.
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "co_occurs", pol="neutral", quote="q"),
        _rel("C", "D", "endorses", pol="positive", quote="q"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1
    assert out["relations"][0]["act_type"] == "endorses"


def test_allowed_act_types_filters():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False,
                          allowed_act_types=["endorses"])
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", quote="q"),
        _rel("A", "B", "endorses", quote="q"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1
    assert out["relations"][0]["act_type"] == "endorses"


def test_max_relations_caps_output():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False, max_relations_per_article=1)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", quote="q"),
        _rel("C", "D", "endorses", quote="q"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1


def test_polarity_consistency_corrects_polarity():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False,
                          enforce_polarity_consistency=True)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", pol="positive", quote="q")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["polarity"] == "negative"


def test_entities_passthrough_untouched():
    vc = ValidationConfig()
    parsed = {"entities": [{"name": "Boric", "type": "roster_actor"}], "relations": []}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["entities"] == [{"name": "Boric", "type": "roster_actor"}]


# --- Construcciones de RECEPCIÓN (commit 94e4e92) ---------------------------------
# "X recibió el apoyo de Y": el sujeto X es el RECEPTOR y el agente real Y va tras
# el 'de'. El modelo tiende a emitir from=receptor → debe invertirse a from=agente.


def test_reception_swaps_subject_and_agent():
    # Modelo emite from=receptor (Boric), to=agente (Matthei). La cita es de
    # recepción → debe quedar from=Matthei (agente), to=Boric (receptor).
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=True, allowed_act_types=None)
    parsed = {"entities": [], "relations": [
        _rel("Boric", "Matthei", act="endorses",
             quote="Boric recibió el apoyo de Matthei")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["from_entity"] == "Matthei"   # agente (Y)
    assert out["relations"][0]["to_entity"] == "Boric"       # receptor (X)


def test_reception_respaldo_marker_also_swaps():
    # Otro marcador de la familia ("contó con el respaldo de") debe invertir igual.
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=True, allowed_act_types=None)
    parsed = {"entities": [], "relations": [
        _rel("Provoste", "Boric", act="allies_with",
             quote="Provoste contó con el respaldo de Boric")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["from_entity"] == "Boric"
    assert out["relations"][0]["to_entity"] == "Provoste"


def test_no_reception_marker_preserves_direction():
    # La cita contiene ' de ' pero NO un marcador de recepción → NO debe invertirse.
    # Prueba que es el MARCADOR (no la mera preposición 'de') lo que gatilla el swap.
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=True, allowed_act_types=None)
    parsed = {"entities": [], "relations": [
        _rel("Boric", "Matthei", act="endorses",
             quote="Boric habló en la casa de Matthei")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["from_entity"] == "Boric"
    assert out["relations"][0]["to_entity"] == "Matthei"


def test_passive_with_de_in_quote_still_swaps():
    # No-regresión del caso pasivo: una pasiva con 'por' que ADEMÁS contiene ' de '
    # debe seguir invirtiéndose (el nuevo código de recepción no la captura porque
    # no hay marcador de recepción). El caso canónico está en
    # test_passive_direction_swaps_from_to.
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=True, allowed_act_types=None)
    parsed = {"entities": [], "relations": [
        _rel("Boric", "Matthei", act="attacks",
             quote="Boric fue criticado por el senador de RN Matthei")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["from_entity"] == "Matthei"
    assert out["relations"][0]["to_entity"] == "Boric"


def test_shares_token_matches_plural_and_truncation():
    # Empareja por prefijo en ambas direcciones: tolera plural/truncación.
    assert _shares_token("artesanal", "pescadores artesanales") is True
    assert _shares_token("artesanales", "gremio artesanal") is True
    # control negativo: sin token significativo compartido → False
    assert _shares_token("boric", "matthei kast provoste") is False


def test_shares_token_ignores_stopwords():
    # 'para' tiene len>=4 pero es stopword → no es token significativo, así que
    # aunque aparezca en el texto NO produce match.
    assert _shares_token("para", "para todos siempre") is False
    # contraste: un token significativo sí matchea.
    assert _shares_token("partido", "el partido nuevo") is True


def test_reception_reorders_without_dropping_relations():
    # La corrección de recepción REORDENA pero nunca DESCARTA (no afecta recall).
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=True, dedup=False,
                          allowed_act_types=None)
    parsed = {"entities": [], "relations": [
        # marcador + entidades presentes en la cita → se invierte
        _rel("Boric", "Matthei", act="endorses",
             quote="Boric recibió el apoyo de Matthei"),
        # marcador presente pero las entidades NO aparecen en la cita → no se puede
        # emparejar por token → se conserva intacta (no se descarta).
        _rel("Kast", "Provoste", act="endorses",
             quote="Boric recibió el apoyo de Matthei"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    # no-drop: ninguna de las dos relaciones se descarta
    assert len(out["relations"]) == 2
    # reorder: la primera se invirtió (agente al from)
    assert (out["relations"][0]["from_entity"],
            out["relations"][0]["to_entity"]) == ("Matthei", "Boric")
    # la segunda queda intacta (sin match de tokens → sin swap, sin drop)
    assert (out["relations"][1]["from_entity"],
            out["relations"][1]["to_entity"]) == ("Kast", "Provoste")
