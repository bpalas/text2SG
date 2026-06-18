import sys
import types

import pytest

import text2sg.llm_backends as lb


def test_ollama_client_shapes_response(monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout=600):
        captured["url"] = url
        captured["payload"] = payload
        return {"response": '{"ok": true}', "prompt_eval_count": 12, "eval_count": 7}

    monkeypatch.setattr(lb, "_post", fake_post)
    client = lb.OllamaClient()
    r = client.models.generate_content(model="qwen2.5:7b", contents="hola")

    assert r.text == '{"ok": true}'
    assert r.usage_metadata.prompt_token_count == 12
    assert r.usage_metadata.candidates_token_count == 7
    assert captured["payload"]["model"] == "qwen2.5:7b"
    assert captured["payload"]["stream"] is False
    assert captured["url"].endswith("/api/generate")


def test_anthropic_client_shapes_response(monkeypatch):
    captured = {}

    class _FakeUsage:
        input_tokens = 12
        output_tokens = 7

    class _FakeBlock:
        type = "text"
        text = '{"ok": true}'

    class _FakeMsg:
        content = [_FakeBlock()]
        usage = _FakeUsage()

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeMsg()

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self.messages = _FakeMessages()

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    client = lb.AnthropicClient(max_tokens=4096)
    r = client.models.generate_content(model="claude-haiku-4-5", contents="hola")

    assert r.text == '{"ok": true}'
    assert r.usage_metadata.prompt_token_count == 12
    assert r.usage_metadata.candidates_token_count == 7
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["max_tokens"] == 4096
    assert captured["messages"] == [{"role": "user", "content": "hola"}]
    assert "system" not in captured   # sin system → no se manda el param (retrocompat)


# ── split system/user: cada backend enruta `system` a su canal nativo ───────── #

def _fake_anthropic(captured):
    class _FakeUsage:
        input_tokens = 1
        output_tokens = 1

    class _FakeBlock:
        type = "text"
        text = "{}"

    class _FakeMsg:
        content = [_FakeBlock()]
        usage = _FakeUsage()

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeMsg()

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = _FakeMessages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = _FakeAnthropic
    return mod


def test_ollama_routes_system_field(monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout=600):
        captured["payload"] = payload
        return {"response": "{}", "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(lb, "_post", fake_post)
    lb.OllamaClient().models.generate_content(
        model="m", contents="USER", system="SYSTEM_ROBUSTO")
    assert captured["payload"]["system"] == "SYSTEM_ROBUSTO"
    assert captured["payload"]["prompt"] == "USER"


def test_anthropic_routes_system_param(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(captured))
    lb.AnthropicClient().models.generate_content(
        model="claude-x", contents="USER", system="SYSTEM_ROBUSTO")
    assert captured["system"] == "SYSTEM_ROBUSTO"
    assert captured["messages"] == [{"role": "user", "content": "USER"}]


def test_openai_prepends_system_message(monkeypatch):
    captured = {}

    class _FakeUsage:
        prompt_tokens = 1
        completion_tokens = 1

    class _FakeChoice:
        class message:
            content = "{}"

    class _FakeResp:
        choices = [_FakeChoice()]
        usage = _FakeUsage()

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = _FakeChat()

    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)

    lb.OpenAIClient().models.generate_content(
        model="gpt-x", contents="USER", system="SYSTEM_ROBUSTO")
    assert captured["messages"][0] == {"role": "system", "content": "SYSTEM_ROBUSTO"}
    assert captured["messages"][1] == {"role": "user", "content": "USER"}


def test_gemini_routes_system_instruction(monkeypatch):
    pytest.importorskip("google.genai")
    import google.genai as genai_mod
    captured = {}

    class _FakeUsage:
        prompt_token_count = 5
        candidates_token_count = 3

    class _FakeResp:
        text = "{}"
        usage_metadata = _FakeUsage()

    class _FakeInner:
        class models:
            @staticmethod
            def generate_content(model, contents, config=None):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return _FakeResp()

    monkeypatch.setattr(genai_mod, "Client", lambda **kw: _FakeInner())
    client = lb.GeminiClient(api_key="x")

    # con system → enruta a system_instruction
    client.models.generate_content(model="gemini-x", contents="USER", system="SYSTEM_ROBUSTO")
    assert captured["config"].system_instruction == "SYSTEM_ROBUSTO"
    assert captured["contents"] == "USER"

    # sin system → llamada cruda sin config (retrocompat con el extractor)
    captured.clear()
    r = client.models.generate_content(model="gemini-x", contents="USER")
    assert captured["config"] is None   # no se pasó config → default del fake
    assert r.usage_metadata.prompt_token_count == 5
