from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

from swarm_optimizer.config import SEED_PROMPT


# P-001: act_types permitidos en la semilla — excluye co_occurs (la co-presencia
# sin verbo conector era la mayor fuente de FP; señal +0.01-0.015 P_rel, costo R ~0).
SEED_ALLOWED_ACT_TYPES = [
    "endorses", "accuses", "allies_with", "calls_on", "distances_from",
    "attacks", "questions", "negotiates_with", "competes_with",
]


@dataclass
class ValidationConfig:
    """Artefacto B: post-proceso determinista (costo $0)."""
    require_evidence_substring: bool = True
    min_quote_len: int = 8
    normalize_passive_direction: bool = True
    dedup: bool = True
    enforce_polarity_consistency: bool = False
    allowed_act_types: list[str] | None = field(
        default_factory=lambda: list(SEED_ALLOWED_ACT_TYPES))
    max_relations_per_article: int | None = None
    # P-015: filtro de confianza ordinal (explicit/strongly_implied/speculative → 1/0.7/0.4)
    min_confidence: float | None = None   # None = sin filtro; ej: 0.7 descarta speculative
    # P-016: ambos actores deben aparecer en la evidence_quote
    require_both_in_quote: bool = False


@dataclass
class AnalysisConfig:
    """Artefacto C: análisis determinista pre-extracción (costo $0).
    Cada flag gatea una sección del bloque que produce analysis.build_analysis()."""
    emit_dossier: bool = True
    emit_alias_map: bool = True
    emit_role_hints: bool = True
    emit_direction_scaffold: bool = True
    emit_main_speaker: bool = True
    emit_comention_pairs: bool = True
    emit_act_type_canon: bool = True
    emit_domain_gate: bool = True
    role_window: int = 80          # ventana ±chars alrededor de cada mención para detectar rol
    role_keywords: dict | None = None   # None → usa DEFAULT_ROLE_KEYWORDS de analysis.py


@dataclass
class Genome:
    """Artefacto A (prompt) + flags + Artefacto B (validation) + Artefacto C (analysis)."""
    prompt_text: str
    few_shots: list[str] = field(default_factory=list)
    architecture: str = "one_pass"          # "one_pass" | "given_entities" | "debate"
    model: str = "gemini-2.5-flash"
    verify: bool = False                     # verificación agéntica en inferencia
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    analysis: "AnalysisConfig | None" = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        data = dict(data)
        vc = data.pop("validation", {}) or {}
        adata = data.pop("analysis", None)
        analysis = AnalysisConfig(**adata) if adata else None
        return cls(validation=ValidationConfig(**vc), analysis=analysis, **data)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_seed(cls) -> "Genome":
        return cls(prompt_text=SEED_PROMPT)

    @classmethod
    def seed_variants(cls) -> list["Genome"]:
        """Semillas competidoras (tarea 1.4 del informe 2026-06-09): seed base,
        variante con verificación agéntica y variante con debate interno
        (Societies of Thought). Se añaden al archivo y ELO decide."""
        return [
            cls(prompt_text=SEED_PROMPT),
            cls(prompt_text=SEED_PROMPT, verify=True),
            cls(prompt_text=SEED_PROMPT, architecture="debate"),
        ]
