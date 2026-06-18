"""Backends LLM intercambiables. Exponen la MISMA interfaz que google-genai
(`client.models.generate_content(model=..., contents=...)` -> objeto con `.text` y
`.usage_metadata`) para que loop/extractor/mutate funcionen sin cambios.

OllamaClient permite correr el loop 100% local (cero costo de API).
AnthropicClient permite correr con modelos Claude (requiere ANTHROPIC_API_KEY)."""
from __future__ import annotations

import json
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_NUM_CTX = 8192


def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class _Usage:
    def __init__(self, prompt_token_count: int, candidates_token_count: int):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _Resp:
    def __init__(self, text: str, usage: _Usage):
        self.text = text
        self.usage_metadata = usage


class _Models:
    def __init__(self, host: str, num_ctx: int, temperature: float):
        self.host = host
        self.num_ctx = num_ctx
        self.temperature = temperature

    def generate_content(self, model: str, contents: str,
                         system: str | None = None) -> _Resp:
        payload = {
            "model": model,
            "prompt": contents,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if system:
            payload["system"] = system   # /api/generate soporta rol system nativo
        out = _post(f"{self.host}/api/generate", payload)
        usage = _Usage(out.get("prompt_eval_count", 0), out.get("eval_count", 0))
        return _Resp(out.get("response", ""), usage)


class OllamaClient:
    """Drop-in replacement del genai.Client para correr local con Ollama."""

    def __init__(self, host: str = DEFAULT_HOST, num_ctx: int = DEFAULT_NUM_CTX,
                 temperature: float = 0.0):
        self.models = _Models(host, num_ctx, temperature)


class _AnthropicModels:
    def __init__(self, max_tokens: int, api_key: str | None):
        import anthropic  # import perezoso: solo se exige el SDK si se usa este backend
        kwargs = {"api_key": api_key} if api_key else {}
        self._client = anthropic.Anthropic(**kwargs)
        self.max_tokens = max_tokens

    def generate_content(self, model: str, contents: str,
                         system: str | None = None) -> _Resp:
        kwargs = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": contents}],
        }
        if system:
            kwargs["system"] = system   # param top-level dedicado de la API de Claude
        msg = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = _Usage(msg.usage.input_tokens, msg.usage.output_tokens)
        return _Resp(text, usage)


class AnthropicClient:
    """Drop-in replacement del genai.Client para usar modelos Claude.

    El modelo se toma del genoma (campo `model`), igual que con Gemini.
    IDs sugeridos: `claude-haiku-4-5` (barato/rápido), `claude-sonnet-4-6`.
    Lee ANTHROPIC_API_KEY del entorno si no se pasa api_key explícita.
    """

    def __init__(self, max_tokens: int = 8192, api_key: str | None = None):
        self.models = _AnthropicModels(max_tokens, api_key)


class _OpenAIModels:
    def __init__(self, max_tokens: int, api_key: str | None):
        from openai import OpenAI  # import perezoso: solo exige el SDK si se usa
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.max_tokens = max_tokens

    def generate_content(self, model: str, contents: str,
                         system: str | None = None) -> _Resp:
        # max_completion_tokens es el parámetro moderno (gpt-4o-mini y gpt-5.x lo aceptan);
        # no fijamos temperature porque varios modelos de razonamiento solo aceptan el default.
        messages = [{"role": "user", "content": contents}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = _Usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return _Resp(text, usage)


class OpenAIClient:
    """Drop-in replacement del genai.Client para usar modelos OpenAI.

    El modelo se toma del genoma (campo `model`). IDs: `gpt-4o-mini`, `gpt-5.4-mini`,
    `gpt-5.4-nano` (verificar disponibilidad contra la API antes de gastar).
    Lee OPENAI_API_KEY del entorno si no se pasa api_key explícita.
    """

    def __init__(self, max_tokens: int = 4096, api_key: str | None = None):
        self.models = _OpenAIModels(max_tokens, api_key)


class _GeminiModels:
    def __init__(self, api_key: str | None, temperature: float):
        from google import genai  # import perezoso: solo exige el SDK si se usa
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.temperature = temperature

    def generate_content(self, model: str, contents: str,
                         system: str | None = None):
        """Enruta `system` al system_instruction de google-genai. Sin system,
        llama igual que `genai.Client` crudo (retrocompatible con el extractor).
        Devuelve la respuesta nativa de genai (ya tiene .text y .usage_metadata)."""
        if system:
            from google.genai import types
            cfg = types.GenerateContentConfig(
                system_instruction=system, temperature=self.temperature)
            return self._client.models.generate_content(
                model=model, contents=contents, config=cfg)
        return self._client.models.generate_content(model=model, contents=contents)


class GeminiClient:
    """Wrapper sobre genai.Client que añade soporte de rol `system` sin romper la
    interfaz. loop.py lo usa como cliente default para que los meta-agentes puedan
    pasar un system prompt; el extractor sigue con genai.Client crudo (no pasa system).
    """

    def __init__(self, api_key: str | None = None, temperature: float = 0.0):
        self.models = _GeminiModels(api_key, temperature)
