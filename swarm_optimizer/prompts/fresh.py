"""Agente FRESH: diseña un extractor nuevo desde cero (ya NO ciego).

Recibe los top-2 como inspiración + el diagnóstico (qué falla hoy) + los patrones del
meta-revisor, para diseñar contra debilidades concretas en vez de adivinar.
"""
from __future__ import annotations

from swarm_optimizer.prompts._authored import STRUCT_FRESH, SYSTEM_FRESH
from swarm_optimizer.prompts.base import PromptSpec


def build_user_fresh(parent1_prompt: str, parent2_prompt: str,
                     diagnosis: str = "", meta_review: str = "") -> str:
    diag = (f"\nDiagnóstico (qué falla hoy — atácalo deliberadamente):\n{diagnosis}\n"
            if diagnosis else "")
    mr = (f"\nPatrones sistémicos del meta-revisor (frente de Pareto):\n{meta_review}\n"
          if meta_review else "")
    return (
        "Prompts de los dos mejores extractores como INSPIRACIÓN (no los copies; diseña un "
        "enfoque estructuralmente distinto que ataque las debilidades de abajo).\n\n"
        f"PROMPT top-1:\n{parent1_prompt}\n\n"
        f"PROMPT top-2:\n{parent2_prompt}\n"
        f"{diag}{mr}"
    )


FRESH_SPEC = PromptSpec(
    agent="fresh",
    system=SYSTEM_FRESH,
    build_user=build_user_fresh,
    required_structure=STRUCT_FRESH,
    required_context=("FRESH_P1_SENTINEL", "FRESH_P2_SENTINEL",
                      "FRESH_DIAG_SENTINEL", "FRESH_MR_SENTINEL"),
    probe={
        "parent1_prompt": "FRESH_P1_SENTINEL",
        "parent2_prompt": "FRESH_P2_SENTINEL",
        "diagnosis": "FRESH_DIAG_SENTINEL",
        "meta_review": "FRESH_MR_SENTINEL",
    },
)
