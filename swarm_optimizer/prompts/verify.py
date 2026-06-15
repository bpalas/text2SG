"""Spec de cobertura del agente-tarea VERIFY (verificación agéntica post-extracción).

`runtime_split=False`: igual que extract, su runtime (extractor.verify_relations) NO se
toca. Este spec da cobertura de rubric al prompt de verificación, reusando el template
real `_VERIFY_PROMPT` para evitar drift entre el rubric y el runtime.
"""
from __future__ import annotations

from swarm_optimizer.extractor import _VERIFY_PROMPT
from swarm_optimizer.prompts.base import PromptSpec


def build_user_verify(body: str, relations: str) -> str:
    """Reusa el template real de verificación (sin drift)."""
    return _VERIFY_PROMPT.format(body=body, relations=relations)


VERIFY_SPEC = PromptSpec(
    agent="verify",
    system="",
    build_user=build_user_verify,
    required_structure=(
        "Revisa estas relaciones",                       # TASK
        "Elimina las que NO estén soportadas",           # CONSTRAINT
        "corrige dirección/polaridad",                   # CONSTRAINT
        "JSON corregido",                                # OUTPUT_FORMAT
    ),
    required_context=("VERIFY_BODY_SENTINEL_X9", "VERIFY_REL_SENTINEL_X9"),
    probe={"body": "VERIFY_BODY_SENTINEL_X9", "relations": "VERIFY_REL_SENTINEL_X9"},
    runtime_split=False,
)
