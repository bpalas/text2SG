"""Paquete de prompts: fuente única de cada agente (runtime + rubric de calidad).

Cada módulo exporta su `PromptSpec` (system robusto + build_user + rubric). mutate.py
consume los SYSTEM_* y build_user_* en runtime; test_prompt_quality.py consume ALL_SPECS.

Estructura alineada a gepa-ai/gepa: el prompt vive junto a su lógica.
"""
from __future__ import annotations

from swarm_optimizer.prompts.base import PromptSpec
from swarm_optimizer.prompts.cross import CROSS_SPEC, build_user_cross
from swarm_optimizer.prompts.diagnose import DIAGNOSE_SPEC, build_user_diagnose
from swarm_optimizer.prompts.extract import EXTRACT_SPEC
from swarm_optimizer.prompts.fresh import FRESH_SPEC, build_user_fresh
from swarm_optimizer.prompts.meta_review import META_REVIEW_SPEC, build_user_meta_review
from swarm_optimizer.prompts.propose import PROPOSE_SPEC, build_user_propose
from swarm_optimizer.prompts.verify import VERIFY_SPEC
from swarm_optimizer.prompts._authored import (
    SYSTEM_CROSS, SYSTEM_DIAGNOSE, SYSTEM_FRESH, SYSTEM_META_REVIEW, SYSTEM_PROPOSE,
)

# Los 5 meta-agentes del self-evolve + los 2 agentes-tarea (cobertura).
ALL_SPECS: list[PromptSpec] = [
    DIAGNOSE_SPEC,
    META_REVIEW_SPEC,
    PROPOSE_SPEC,
    CROSS_SPEC,
    FRESH_SPEC,
    EXTRACT_SPEC,
    VERIFY_SPEC,
]

__all__ = [
    "PromptSpec", "ALL_SPECS",
    "DIAGNOSE_SPEC", "META_REVIEW_SPEC", "PROPOSE_SPEC", "CROSS_SPEC", "FRESH_SPEC",
    "EXTRACT_SPEC", "VERIFY_SPEC",
    "build_user_diagnose", "build_user_meta_review", "build_user_propose",
    "build_user_cross", "build_user_fresh",
    "SYSTEM_DIAGNOSE", "SYSTEM_META_REVIEW", "SYSTEM_PROPOSE", "SYSTEM_CROSS", "SYSTEM_FRESH",
]
