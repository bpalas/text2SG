"""
Multi-model pipeline for political relation extraction.

Allows assigning different LLM backends/models to each role:
  - extractor: main relation extraction call
  - ner:       NER pass (end2end mode only, defaults to extractor)
  - verifier:  optional agentic verification pass ($0 if omitted)

Example:
    config = PipelineConfig(
        mode="end2end",
        extractor=AgentDef("ollama", "qwen2.5:7b"),
        ner=AgentDef("gemini", "gemini-2.0-flash-lite"),
        verifier=AgentDef("anthropic", "claude-haiku-4-5"),
    )
    result = extract_text(article_text, genome, config)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass


_BACKENDS = ("gemini", "anthropic", "openai", "ollama")


@dataclass
class AgentDef:
    """One LLM role: a backend name + model identifier.

    Parse from a colon-separated spec — first token is the backend,
    the rest is the model name (model names may contain colons, e.g. qwen2.5:7b).

        AgentDef.from_str("ollama:qwen2.5:7b")
        AgentDef.from_str("gemini:gemini-2.0-flash-lite")
        AgentDef.from_str("anthropic:claude-haiku-4-5")
    """
    backend: str
    model: str

    @classmethod
    def from_str(cls, spec: str) -> "AgentDef":
        parts = spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"AgentDef spec must be 'backend:model', got {spec!r}. "
                f"Example: 'ollama:qwen2.5:7b'"
            )
        backend = parts[0].lower()
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from: {_BACKENDS}")
        return cls(backend=backend, model=parts[1])

    def make_client(self):
        """Instantiate the LLM client for this agent."""
        from text2sg.llm_backends import (
            GeminiClient, AnthropicClient, OpenAIClient, OllamaClient,
        )
        return {
            "gemini":    GeminiClient,
            "anthropic": AnthropicClient,
            "openai":    OpenAIClient,
            "ollama":    OllamaClient,
        }[self.backend]()

    def __str__(self) -> str:
        return f"{self.backend}:{self.model}"


@dataclass
class PipelineConfig:
    """Which models to use for each role in the extraction pipeline.

    Args:
        mode:      "given_entities" (actors pre-supplied) or "end2end" (NER first)
        extractor: main extraction LLM — required
        ner:       NER agent for end2end mode; defaults to extractor if omitted
        verifier:  optional agentic verify pass; None = skip (saves tokens)
    """
    mode: str = "given_entities"
    extractor: AgentDef = None
    ner: AgentDef = None
    verifier: AgentDef = None

    def __post_init__(self):
        if self.extractor is None:
            raise ValueError("PipelineConfig requires at least extractor=AgentDef(...)")
        if self.mode == "end2end" and self.ner is None:
            self.ner = self.extractor

    @classmethod
    def from_cli_args(
        cls,
        mode: str,
        extractor: str,
        ner: str | None = None,
        verifier: str | None = None,
    ) -> "PipelineConfig":
        return cls(
            mode=mode,
            extractor=AgentDef.from_str(extractor),
            ner=AgentDef.from_str(ner) if ner else None,
            verifier=AgentDef.from_str(verifier) if verifier else None,
        )


def _actors_to_union(actors: list[str]) -> dict:
    """Convert a flat list of actor names to the internal union dict format."""
    return {
        f"U{i + 1}": {"canonical_names": [name], "type": "roster_actor"}
        for i, name in enumerate(actors)
    }


def extract_text(
    text: str,
    genome,
    config: PipelineConfig,
    actors: list[str] | None = None,
    article_id: str = "article",
) -> dict:
    """Extract political relations from a single text string.

    Args:
        text:       the article body (Spanish)
        genome:     Genome object — prompt_text + ValidationConfig + AnalysisConfig
        config:     PipelineConfig — which model handles each role
        actors:     known actor names for given_entities mode (ignored in end2end)
        article_id: identifier included in the result

    Returns:
        {
            "article_id": str,
            "relations": [{"from_entity", "to_entity", "act_type", "polarity",
                           "issue", "evidence_quote", ...}],
            "entities":  [...],
            "tokens":    {"ner": int, "extractor": int, "verifier": int, "total": int},
        }
    """
    from text2sg.extractor import extract_entities, extract_article, verify_relations

    token_counts: dict[str, int] = {"ner": 0, "extractor": 0, "verifier": 0}

    # ── 1. NER pass (end2end only) ────────────────────────────────────────── #
    if config.mode == "end2end":
        ner_agent = config.ner  # already defaulted to extractor in __post_init__
        ner_client = ner_agent.make_client()
        union, ner_tokens = extract_entities(text, ner_agent.model, ner_client)
        token_counts["ner"] = ner_tokens
    else:
        union = _actors_to_union(actors or [])

    # ── 2. Extraction pass ────────────────────────────────────────────────── #
    # Clone genome so we can override model + disable internal verify without
    # mutating the caller's object.
    g = copy.copy(genome)
    g.model = config.extractor.model
    g.verify = False          # we run verify separately with its own client

    ext_client = config.extractor.make_client()
    result = extract_article(
        article_id, text, union, g,
        few_shot_examples=[],   # few-shots require a gold DataFrame; skip in standalone mode
        client=ext_client,
    )
    token_counts["extractor"] = result.get("tokens", 0)

    # ── 3. Optional agentic verify pass ──────────────────────────────────── #
    if config.verifier is not None:
        ver_client = config.verifier.make_client()
        verified_rels, ver_tokens = verify_relations(
            result.get("relations", []), text, config.verifier.model, ver_client,
        )
        result["relations"] = verified_rels
        token_counts["verifier"] = ver_tokens

    token_counts["total"] = sum(token_counts.values())
    result["tokens"] = token_counts
    return result
