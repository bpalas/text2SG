# experiments/swarm_optimizer/tests/test_config.py
import json
from swarm_optimizer.config import Config, DEFAULT_WEIGHTS, SEED_PROMPT

def test_config_defaults():
    c = Config(prompt_text="hola")
    assert c.architecture == "one_pass"
    assert c.model == "gemini-2.5-flash"
    assert c.rubric_weights == DEFAULT_WEIGHTS

def test_score_perfect():
    c = Config(prompt_text="x")
    metrics = {"F1_rel": 1.0, "Polarity_acc": 1.0, "F1_ent": 1.0, "Act_acc": 1.0}
    assert c.score(metrics) == 1.0

def test_score_partial():
    c = Config(prompt_text="x")
    metrics = {"F1_rel": 0.8, "Polarity_acc": 0.6, "F1_ent": 0.5, "Act_acc": 0.4}
    expected = 0.40 * 0.8 + 0.30 * 0.6 + 0.15 * 0.5 + 0.15 * 0.4
    assert abs(c.score(metrics) - expected) < 1e-9

def test_roundtrip_json():
    c = Config(prompt_text="test prompt", few_shots=["abc123"], architecture="given_entities")
    c2 = Config.from_json(c.to_json())
    assert c2.prompt_text == "test prompt"
    assert c2.few_shots == ["abc123"]
    assert c2.architecture == "given_entities"

def test_seed_prompt_not_empty():
    c = Config.from_seed()
    assert len(c.prompt_text) > 100
    assert c.architecture == "one_pass"
