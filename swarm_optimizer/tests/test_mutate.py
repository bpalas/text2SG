from swarm_optimizer.genome import Genome, ValidationConfig
from swarm_optimizer.mutate import (
    apply_diff, parse_search_replace, apply_validation_patch, propose,
    diagnose, fresh_genome, meta_review, merge_genomes, cross_pollinate,
)


# ── system-aware merge por componente (GEPA) ─────────────────────────────────── #

def test_merge_takes_changed_component_from_each_parent():
    anc = {"prompt_text": "P0", "architecture": "given_entities", "validation": {"min_quote_len": 8},
           "analysis": None}
    a = {**anc, "prompt_text": "P_dir"}                       # A cambió solo el prompt
    b = {**anc, "validation": {"min_quote_len": 12}}          # B cambió solo la validación
    merged = merge_genomes(a, b, anc, score_a=0.9, score_b=0.8)
    assert merged["prompt_text"] == "P_dir"                   # de A
    assert merged["validation"] == {"min_quote_len": 12}      # de B
    assert merged["architecture"] == "given_entities"         # ninguno tocó → ancestro


def test_merge_breaks_tie_by_score_when_both_changed():
    anc = {"prompt_text": "P0", "validation": {}, "analysis": None}
    a = {**anc, "prompt_text": "P_a"}
    b = {**anc, "prompt_text": "P_b"}
    assert merge_genomes(a, b, anc, 0.9, 0.7)["prompt_text"] == "P_a"   # gana el de mayor score
    assert merge_genomes(a, b, anc, 0.6, 0.7)["prompt_text"] == "P_b"


def test_merge_keeps_ancestor_when_neither_changed():
    anc = {"prompt_text": "P0", "verify": False, "validation": {}, "analysis": None}
    a = {**anc, "verify": True}
    b = {**anc}
    merged = merge_genomes(a, b, anc, 0.9, 0.8)
    assert merged["verify"] is True and merged["prompt_text"] == "P0"


class FakeResp:
    def __init__(self, text): self.text = text


class FakeModels:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None   # mensaje user (contexto de la iteración)
        self.last_system = None   # system prompt robusto (rol/formato/invariantes)

    def generate_content(self, model, contents, system=None):
        self.last_prompt = contents
        self.last_system = system
        return FakeResp(self._text)


class FakeClient:
    def __init__(self, text): self.models = FakeModels(text)


def test_apply_diff_replaces_once():
    out, ok = apply_diff("hola mundo mundo", "mundo", "tierra")
    assert ok is True
    assert out == "hola tierra mundo"


def test_apply_diff_noop_when_search_absent():
    out, ok = apply_diff("abc", "zzz", "x")
    assert ok is False
    assert out == "abc"


def test_apply_diff_tolerates_whitespace_differences():
    # el LLM re-indentó: search con espacios simples, texto con múltiples
    out, ok = apply_diff("Regla    X   aquí", "Regla X aquí", "Regla Y")
    assert ok is True
    assert out == "Regla Y"


def test_parse_search_replace_extracts_blocks():
    text = (
        "blah\n<<<<<<< SEARCH\nfoo bar\n=======\nfoo BAZ\n>>>>>>> REPLACE\ntrailing"
    )
    res = parse_search_replace(text)
    assert res == ("foo bar", "foo BAZ")


def test_parse_search_replace_none_when_malformed():
    assert parse_search_replace("no markers here") is None


def test_apply_validation_patch_updates_fields():
    vc = ValidationConfig(min_quote_len=8)
    patched = apply_validation_patch(vc, {"min_quote_len": 15, "dedup": False})
    assert patched.min_quote_len == 15
    assert patched.dedup is False
    assert patched.require_evidence_substring == vc.require_evidence_substring


def test_apply_validation_patch_ignores_unknown_keys():
    vc = ValidationConfig()
    patched = apply_validation_patch(vc, {"bogus_key": 1, "min_quote_len": 3})
    assert patched.min_quote_len == 3
    assert not hasattr(patched, "bogus_key")


def test_propose_artifact_a_applies_diff_to_prompt():
    g = Genome(prompt_text="Eres un extractor. Regla X.")
    client = FakeClient(
        '{"artifact": "A", "diff": '
        '"<<<<<<< SEARCH\\nRegla X.\\n=======\\nRegla X mejorada.\\n>>>>>>> REPLACE"}'
    )
    child, mtype, touched = propose(g, "diagnóstico", client)
    assert "Regla X mejorada." in child.prompt_text
    assert mtype == "diff_a" and touched == "A"


def test_propose_artifact_b_patches_validation():
    g = Genome(prompt_text="p")
    client = FakeClient('{"artifact": "B", "patch": {"min_quote_len": 20}}')
    child, mtype, touched = propose(g, "diagnóstico", client)
    assert child.validation.min_quote_len == 20
    assert mtype == "diff_b" and touched == "B"


def test_propose_invalid_json_returns_noop_clone():
    g = Genome(prompt_text="p")
    client = FakeClient("esto no es json")
    child, mtype, touched = propose(g, "diag", client)
    assert child.prompt_text == "p"
    assert mtype == "noop" and touched is None


def test_propose_diff_that_does_not_apply_is_noop():
    g = Genome(prompt_text="contenido real")
    client = FakeClient(
        '{"artifact": "A", "diff": '
        '"<<<<<<< SEARCH\\nNO EXISTE\\n=======\\nx\\n>>>>>>> REPLACE"}'
    )
    child, mtype, touched = propose(g, "diag", client)
    assert child.prompt_text == "contenido real"
    assert mtype == "noop"


# ── extensiones del informe 2026-06-09 ───────────────────────────── #
def test_propose_injects_retrospective_memory_into_prompt():
    g = Genome(prompt_text="p")
    client = FakeClient('{"artifact": "B", "patch": {"min_quote_len": 10}}')
    memory = "Funcionó:\n- diff_b sobre artefacto B: delta +0.030"
    propose(g, "diag", client, memory=memory)
    assert "Intentos previos del linaje" in client.models.last_prompt
    assert "delta +0.030" in client.models.last_prompt


def test_propose_without_memory_omits_memory_block():
    g = Genome(prompt_text="p")
    client = FakeClient('{"artifact": "B", "patch": {"min_quote_len": 10}}')
    propose(g, "diag", client)
    assert "Intentos previos del linaje" not in client.models.last_prompt


def test_propose_force_artifact_adds_constraint():
    g = Genome(prompt_text="p")
    client = FakeClient('{"artifact": "B", "patch": {"min_quote_len": 10}}')
    propose(g, "diag", client, force_artifact="B")
    assert "DEBES elegir el artefacto B" in client.models.last_prompt


def test_diagnose_injects_meta_review():
    client = FakeClient("1. causa")
    diagnose(["fp1"], ["fn1"], client, meta_review_text="patrón co_occurs espurio")
    assert "patrón co_occurs espurio" in client.models.last_prompt


def test_meta_review_returns_text():
    client = FakeClient("1. patrón sistémico X")
    out = meta_review(["fp1", "fp2"], ["fn1"], client)
    assert out == "1. patrón sistémico X"
    assert "fp1" in client.models.last_prompt


def test_meta_review_failsafe_empty_on_error():
    class Boom:
        class models:
            @staticmethod
            def generate_content(model, contents, system=None): raise RuntimeError("x")
    assert meta_review(["fp"], ["fn"], Boom()) == ""


def test_fresh_genome_generates_new_prompt():
    p1 = Genome(prompt_text="prompt uno")
    p2 = Genome(prompt_text="prompt dos")
    client = FakeClient("PROMPT NUEVO desde cero")
    child, mtype, touched = fresh_genome(p1, p2, client)
    assert child.prompt_text == "PROMPT NUEVO desde cero"
    assert mtype == "fresh" and touched == "A"
    assert "prompt uno" in client.models.last_prompt
    assert "prompt dos" in client.models.last_prompt


def test_fresh_genome_failsafe_noop():
    p1 = Genome(prompt_text="prompt uno")
    p2 = Genome(prompt_text="prompt dos")
    child, mtype, touched = fresh_genome(p1, p2, FakeClient(""))
    assert mtype == "noop"
    assert child.prompt_text == "prompt uno"


# ── split system/user: el system robusto viaja como rol `system` ─────────────── #
def test_propose_sends_robust_system_prompt():
    from swarm_optimizer.prompts import SYSTEM_PROPOSE
    g = Genome(prompt_text="p")
    client = FakeClient('{"artifact": "B", "patch": {"min_quote_len": 10}}')
    propose(g, "diag", client)
    assert client.models.last_system == SYSTEM_PROPOSE
    assert len(client.models.last_system) > 200          # system no trivial
    assert "diag" in client.models.last_prompt           # contexto en el user


def test_diagnose_sends_robust_system_prompt():
    from swarm_optimizer.prompts import SYSTEM_DIAGNOSE
    client = FakeClient("1. causa")
    diagnose(["fp1"], ["fn1"], client)
    assert client.models.last_system == SYSTEM_DIAGNOSE


def test_cross_pollinate_not_blind_includes_metrics_and_diagnosis():
    from swarm_optimizer.prompts import SYSTEM_CROSS
    p1 = Genome(prompt_text="prompt uno")
    p2 = Genome(prompt_text="prompt dos")
    client = FakeClient("HIJO FUSIONADO")
    child, mtype, touched = cross_pollinate(
        p1, p2, client,
        parent1_metrics={"Precision_rel": 0.9, "Recall_rel": 0.4, "f05": 0.8},
        parent2_metrics={"Precision_rel": 0.5, "Recall_rel": 0.85, "f05": 0.6},
        diagnosis="co_occurs espurio")
    assert mtype == "cross" and touched == "A"
    assert child.prompt_text == "HIJO FUSIONADO"
    assert client.models.last_system == SYSTEM_CROSS
    # ya NO ciego: métricas + diagnóstico + ambos prompts llegan al user
    assert "Precision_rel" in client.models.last_prompt
    assert "co_occurs espurio" in client.models.last_prompt
    assert "prompt uno" in client.models.last_prompt
    assert "prompt dos" in client.models.last_prompt


def test_fresh_genome_not_blind_includes_meta_review():
    from swarm_optimizer.prompts import SYSTEM_FRESH
    p1 = Genome(prompt_text="prompt uno")
    p2 = Genome(prompt_text="prompt dos")
    client = FakeClient("PROMPT NUEVO")
    child, mtype, _ = fresh_genome(p1, p2, client, meta_review_text="patrón X transversal")
    assert child.prompt_text == "PROMPT NUEVO"
    assert client.models.last_system == SYSTEM_FRESH
    # ya NO ciego: los patrones del meta-revisor llegan al user
    assert "patrón X transversal" in client.models.last_prompt
