from swarm_optimizer.genome import Genome, ValidationConfig


def test_seed_has_prompt_and_default_validation():
    g = Genome.from_seed()
    assert "extractor de relaciones" in g.prompt_text.lower()
    assert g.architecture == "one_pass"
    assert g.verify is False
    assert isinstance(g.validation, ValidationConfig)
    assert g.validation.require_evidence_substring is True


def test_roundtrip_json_preserves_validation():
    g = Genome.from_seed()
    g.validation.min_quote_len = 12
    g.verify = True
    restored = Genome.from_json(g.to_json())
    assert restored.validation.min_quote_len == 12
    assert restored.verify is True
    assert restored.prompt_text == g.prompt_text


def test_validation_defaults():
    v = ValidationConfig()
    assert v.dedup is True
    assert v.enforce_polarity_consistency is False
    # P-001: la semilla excluye co_occurs por defecto
    assert v.allowed_act_types is not None
    assert "co_occurs" not in v.allowed_act_types
    assert "endorses" in v.allowed_act_types
    assert v.max_relations_per_article is None
