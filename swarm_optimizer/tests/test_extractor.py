# experiments/swarm_optimizer/tests/test_extractor.py
import json
from unittest.mock import patch, MagicMock
from swarm_optimizer.extractor import parse_llm_output, build_prompt
from swarm_optimizer.config import Config

VALID_JSON = json.dumps({
    "entities": [{"name": "Gabriel Boric", "type": "roster_actor"}],
    "relations": [{
        "from_entity": "Gabriel Boric",
        "to_entity": "el gobierno",
        "act_type": "endorses",
        "polarity": "positive",
        "issue": "government_management",
        "evidence_quote": "Boric apoyó la medida",
    }]
})

def test_parse_valid_json():
    result = parse_llm_output(VALID_JSON)
    assert len(result["entities"]) == 1
    assert len(result["relations"]) == 1
    assert result["relations"][0]["act_type"] == "endorses"

def test_parse_json_in_markdown():
    """Gemini a veces envuelve el JSON en ```json ... ```"""
    wrapped = f"```json\n{VALID_JSON}\n```"
    result = parse_llm_output(wrapped)
    assert len(result["entities"]) == 1

def test_parse_invalid_returns_empty():
    result = parse_llm_output("esto no es JSON válido")
    assert result == {"entities": [], "relations": []}

def test_build_prompt_one_pass():
    cfg = Config(prompt_text="Extrae relaciones.", few_shots=[])
    prompt = build_prompt(cfg, body="Noticia de prueba.", union={}, few_shot_examples=[])
    assert "Noticia de prueba." in prompt
    assert "Extrae relaciones." in prompt

def test_build_prompt_given_entities():
    cfg = Config(prompt_text="Extrae relaciones.", architecture="given_entities")
    union = {"U1": {"type": "roster_actor", "canonical_names": ["Boric"], "surfaces": ["Boric"]}}
    prompt = build_prompt(cfg, body="Noticia.", union=union, few_shot_examples=[])
    assert "U1" in prompt
    assert "Boric" in prompt


# ── validación determinista + verificación agéntica (rediseño evolutivo) ──── #
from swarm_optimizer.genome import Genome, ValidationConfig
from swarm_optimizer.extractor import extract_article, verify_relations

_BODY = "Boric fue criticado por Matthei. Kast respaldó la moción del gobierno."


class _FakeResp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = type("U", (), {"prompt_token_count": 10,
                                             "candidates_token_count": 5})()


class _FakeModels:
    def __init__(self, texts): self._texts = list(texts); self._i = 0
    def generate_content(self, model, contents):
        t = self._texts[min(self._i, len(self._texts) - 1)]; self._i += 1
        return _FakeResp(t)


class _FakeClient:
    def __init__(self, *texts): self.models = _FakeModels(texts)


def test_extract_article_applies_validation_substring_filter():
    g = Genome(prompt_text="p",
               validation=ValidationConfig(require_evidence_substring=True,
                                           min_quote_len=0,
                                           normalize_passive_direction=False))
    out = (
        '{"entities": [], "relations": ['
        '{"from_entity":"Matthei","to_entity":"Boric","act_type":"attacks",'
        '"polarity":"negative","issue":"x","evidence_quote":"criticado por Matthei"},'
        '{"from_entity":"X","to_entity":"Y","act_type":"attacks",'
        '"polarity":"negative","issue":"x","evidence_quote":"cita inexistente zzz"}]}'
    )
    client = _FakeClient(out)
    res = extract_article("a1", _BODY, {}, g, [], client)
    assert len(res["relations"]) == 1
    assert res["relations"][0]["from_entity"] == "Matthei"


def test_verify_relations_drops_unsupported():
    verified_json = (
        '{"relations": [{"from_entity":"Kast","to_entity":"gobierno",'
        '"act_type":"endorses","polarity":"positive","issue":"x",'
        '"evidence_quote":"Kast respaldó la moción"}]}'
    )
    client = _FakeClient(verified_json)
    rels = [
        {"from_entity": "Kast", "to_entity": "gobierno", "act_type": "endorses",
         "polarity": "positive", "issue": "x", "evidence_quote": "Kast respaldó la moción"},
        {"from_entity": "A", "to_entity": "B", "act_type": "attacks",
         "polarity": "negative", "issue": "x", "evidence_quote": "no soportada"},
    ]
    out, tokens = verify_relations(rels, _BODY, "gemini-2.5-flash", client)
    assert len(out) == 1
    assert out[0]["from_entity"] == "Kast"
    assert tokens == 15      # 10 prompt + 5 candidates del _FakeResp


def test_verify_relations_failsafe_on_bad_output():
    client = _FakeClient("no es json")
    rels = [{"from_entity": "A", "to_entity": "B", "act_type": "attacks",
             "polarity": "negative", "issue": "x", "evidence_quote": "q"}]
    out, tokens = verify_relations(rels, _BODY, "gemini-2.5-flash", client)
    assert out == rels       # fail-safe: conserva las originales
    assert tokens == 15      # la llamada sí ocurrió → sus tokens cuentan


def test_verify_relations_failsafe_on_exception():
    class _BoomModels:
        def generate_content(self, model, contents):
            raise RuntimeError("api caída")

    class _BoomClient:
        models = _BoomModels()

    rels = [{"from_entity": "A", "to_entity": "B", "act_type": "attacks",
             "polarity": "negative", "issue": "x", "evidence_quote": "q"}]
    out, tokens = verify_relations(rels, _BODY, "gemini-2.5-flash", _BoomClient())
    assert out == rels       # fail-safe: conserva las originales
    assert tokens == 0       # no hay usage que leer ante excepción


def test_verify_relations_empty_input_no_call():
    out, tokens = verify_relations([], _BODY, "gemini-2.5-flash", _FakeClient("x"))
    assert out == []
    assert tokens == 0


def test_extract_article_verify_sums_tokens_of_both_calls():
    g = Genome(prompt_text="p", verify=True,
               validation=ValidationConfig(require_evidence_substring=False,
                                           min_quote_len=0,
                                           normalize_passive_direction=False))
    extraction = ('{"entities": [], "relations": [{"from_entity":"A","to_entity":"B",'
                  '"act_type":"attacks","polarity":"negative","issue":"x","evidence_quote":"q"}]}')
    verified = ('{"relations": [{"from_entity":"A","to_entity":"B",'
                '"act_type":"attacks","polarity":"negative","issue":"x","evidence_quote":"q"}]}')
    client = _FakeClient(extraction, verified)
    res = extract_article("a1", _BODY, {}, g, [], client)
    assert res["tokens"] == 30   # 15 de extracción + 15 de verificación


# ── extensiones del informe 2026-06-09 ───────────────────────────── #
def test_build_prompt_debate_adds_internal_debate_instructions():
    g = Genome(prompt_text="Extrae relaciones.", architecture="debate")
    prompt = build_prompt(g, body="Noticia.", union={}, few_shot_examples=[])
    assert "MODO DEBATE INTERNO" in prompt
    assert "PROPONENTE" in prompt and "CRÍTICO" in prompt and "ÁRBITRO" in prompt


def test_build_prompt_one_pass_has_no_debate_block():
    g = Genome(prompt_text="Extrae relaciones.")
    prompt = build_prompt(g, body="Noticia.", union={}, few_shot_examples=[])
    assert "MODO DEBATE INTERNO" not in prompt


def test_parse_llm_output_extracts_trailing_json_after_reasoning():
    text = (
        "PROPONENTE: veo una relación A→B.\n"
        "CRÍTICO: ¿hay verbo conector? Sí.\n"
        "ÁRBITRO: emito la relación.\n"
        + VALID_JSON
    )
    result = parse_llm_output(text)
    assert len(result["relations"]) == 1
    assert result["relations"][0]["from_entity"] == "Gabriel Boric"


def test_verify_flag_off_skips_verification():
    g = Genome(prompt_text="p", verify=False,
               validation=ValidationConfig(require_evidence_substring=False,
                                           min_quote_len=0,
                                           normalize_passive_direction=False))
    out = ('{"entities": [], "relations": [{"from_entity":"A","to_entity":"B",'
           '"act_type":"attacks","polarity":"negative","issue":"x","evidence_quote":"q"}]}')
    client = _FakeClient(out)
    res = extract_article("a1", _BODY, {}, g, [], client)
    assert len(res["relations"]) == 1
