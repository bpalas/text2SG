"""Agente PROPOSE: el motor de mutación (artefacto A vía SEARCH/REPLACE o B vía patch).

build_user reúne TODO el contexto que un experto humano usaría para proponer a mano:
ejemplos-trayectoria, diagnóstico, gap de dirección, memoria del linaje (con deltas) y
el genoma actual (A+B). El rol/formato/invariantes viven en el SYSTEM robusto.
"""
from __future__ import annotations

import dataclasses
import json

from text2sg.genome import Genome
from text2sg.prompts._authored import STRUCT_PROPOSE
from text2sg.prompts._authored import SYSTEM_PROPOSE as _SYSTEM_PROPOSE_BASE
from text2sg.prompts.base import PromptSpec

# Refuerzo de formato para modelos pequeños (medido con gemini-2.5-flash-lite): el
# formato SEARCH/REPLACE embebido es FRÁGIL — el modelo chico emite JSON válido pero
# mangla los marcadores (escribe "<<<<<<< PROMPT actual" en vez de "<<<<<<< SEARCH").
# Los campos JSON estructurados juegan a su fortaleza (JSON) y eliminan los marcadores.
# El parser de mutate.propose acepta AMBAS formas (estructurada y embebida).
_STRUCTURED_FORMAT = (
    "\n\nFORMATO PREFERIDO para artefacto A (más robusto — úsalo por defecto):\n"
    'devuelve JSON con campos SEPARADOS: {"artifact": "A", "search": "<texto EXACTO y '
    'MÍNIMO, copiado carácter por carácter del prompt actual, que identifique unívocamente '
    'el punto a editar>", "replace": "<texto nuevo>"}.\n'
    "NO uses marcadores <<<<<<< SEARCH embebidos: search y replace son strings JSON planos."
)
SYSTEM_PROPOSE = _SYSTEM_PROPOSE_BASE + _STRUCTURED_FORMAT


def _format_examples(examples: list[dict]) -> str:
    """Renderiza tripletas (artículo, predicción, gold) para reflexión GEPA."""
    blocks = []
    for i, ex in enumerate(examples, 1):
        pred = "; ".join(f"{r.get('from')}-{r.get('act_type')}->{r.get('to')}"
                         for r in ex.get("predicted", [])) or "(ninguna)"
        gold = "; ".join(f"{r.get('from')}-{r.get('act_type')}->{r.get('to')}"
                         for r in ex.get("gold", [])) or "(ninguna)"
        body = (ex.get("body") or "").strip()
        blocks.append(f"[{i}] (score {ex.get('score')}) ARTÍCULO: {body}\n"
                      f"    EXTRAJO: {pred}\n    GOLD:    {gold}")
    return "\n".join(blocks)


def build_user_propose(genome: Genome, diagnosis: str, memory: str = "",
                       examples: list[dict] | None = None, gap_hint: str = "",
                       force_artifact: str | None = None) -> str:
    """Contexto de la iteración para el motor de mutación. `examples` activa la
    reflexión sobre trayectorias; `memory` evita repetir fracasos del linaje."""
    parts: list[str] = []
    if examples:
        parts.append("Ejemplos concretos (artículo / lo que extrajo el modelo / el gold "
                     "esperado) — extrae una REGLA GENERALIZABLE de dominio:\n"
                     + _format_examples(examples))
    parts.append("Diagnóstico de errores:\n" + diagnosis)
    if gap_hint:
        parts.append(gap_hint)
    if memory:
        parts.append("Intentos previos del linaje (qué funcionó / qué no — no repitas lo "
                     "que falló):\n" + memory)
    parts.append("PROMPT actual (artefacto A):\n" + genome.prompt_text)
    parts.append("ValidationConfig actual (artefacto B):\n"
                 + json.dumps(dataclasses.asdict(genome.validation), ensure_ascii=False))
    if force_artifact in ("A", "B"):
        parts.append(f"DEBES elegir el artefacto {force_artifact}.")
    return "\n\n".join(parts)


PROPOSE_SPEC = PromptSpec(
    agent="propose",
    system=SYSTEM_PROPOSE,
    build_user=build_user_propose,
    required_structure=STRUCT_PROPOSE,
    required_context=("DIAG_SENTINEL", "MEMORY_SENTINEL", "EXBODY_SENTINEL",
                      "PROMPT_A_SENTINEL", "min_quote_len", "GAP_SENTINEL"),
    probe={
        "genome": Genome(prompt_text="PROMPT_A_SENTINEL"),
        "diagnosis": "DIAG_SENTINEL",
        "memory": "MEMORY_SENTINEL",
        "examples": [{"body": "EXBODY_SENTINEL", "predicted": [], "gold": [], "score": 0.5}],
        "gap_hint": "GAP_SENTINEL",
    },
)
