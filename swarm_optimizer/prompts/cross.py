"""Agente CROSS: fusión dirigida por evidencia de dos prompts top (ya NO ciego).

Recibe los dos prompts padres + sus métricas (P/R/f05) + el diagnóstico vigente, para
heredar de cada padre lo que su métrica justifica. Antes recibía solo los dos prompts.
"""
from __future__ import annotations

from swarm_optimizer.prompts._authored import STRUCT_CROSS, SYSTEM_CROSS
from swarm_optimizer.prompts.base import PromptSpec


def _fmt_metrics(m: dict) -> str:
    return (f"Precision_rel={m.get('Precision_rel', 0.0):.3f} "
            f"Recall_rel={m.get('Recall_rel', 0.0):.3f} "
            f"f05={m.get('f05', 0.0):.3f}")


def build_user_cross(parent1_prompt: str, parent2_prompt: str,
                     parent1_metrics: dict, parent2_metrics: dict,
                     diagnosis: str = "") -> str:
    diag = (f"\nDiagnóstico / patrones vigentes (resuelve conflictos entre reglas a su "
            f"favor):\n{diagnosis}\n" if diagnosis else "")
    return (
        "Dos extractores top que destacan por motivos distintos. Combina sus FORTALEZAS "
        "complementarias guiándote por las métricas de cada uno.\n\n"
        f"PADRE 1 — métricas: {_fmt_metrics(parent1_metrics)}\n"
        f"PROMPT del padre 1:\n{parent1_prompt}\n\n"
        f"PADRE 2 — métricas: {_fmt_metrics(parent2_metrics)}\n"
        f"PROMPT del padre 2:\n{parent2_prompt}\n"
        f"{diag}"
    )


CROSS_SPEC = PromptSpec(
    agent="cross",
    system=SYSTEM_CROSS,
    build_user=build_user_cross,
    required_structure=STRUCT_CROSS,
    required_context=("P1_PROMPT_SENTINEL", "P2_PROMPT_SENTINEL",
                      "Precision_rel", "CROSS_DIAG_SENTINEL"),
    probe={
        "parent1_prompt": "P1_PROMPT_SENTINEL",
        "parent2_prompt": "P2_PROMPT_SENTINEL",
        "parent1_metrics": {"Precision_rel": 0.9, "Recall_rel": 0.5, "f05": 0.8},
        "parent2_metrics": {"Precision_rel": 0.6, "Recall_rel": 0.85, "f05": 0.65},
        "diagnosis": "CROSS_DIAG_SENTINEL",
    },
)
