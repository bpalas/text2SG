"""PromptSpec: contrato único de cada agente (runtime + rubric de calidad).

Cada módulo de `prompts/` exporta un `PromptSpec` que co-localiza:
- el SYSTEM prompt robusto (estable, versionado),
- el constructor del mensaje USER (`build_user`, contexto dinámico de la iteración),
- el rubric de calidad: qué estructura debe tener el system y qué contexto debe
  llegar al user.

El test `test_prompt_quality.py` consume estos specs sin costo de API. mutate.py
los consume en runtime para llamar al LLM con rol `system` real.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptSpec:
    """Contrato de un agente. Ver el módulo para el detalle de cada campo."""

    agent: str
    # SYSTEM prompt robusto. "" cuando runtime_split=False (agente-tarea de cobertura).
    system: str
    # Construye el mensaje USER a partir del contexto de la iteración. La firma
    # concreta varía por agente; `probe` provee inputs representativos.
    build_user: Callable[..., str]
    # Rubric estructural: substrings concretos que evidencian ROLE/TASK/
    # OUTPUT_FORMAT/CONSTRAINTS. Deben aparecer en `structure_target()`.
    required_structure: tuple[str, ...]
    # Rubric de contexto: claves de `probe` cuyos valores-sentinela DEBEN aparecer
    # en el USER renderizado. Garantiza que el call-site inyecta toda la info.
    required_context: tuple[str, ...]
    # Inputs representativos (con sentinelas únicos) para renderizar el USER en tests.
    probe: dict
    # True: el agente usa rol `system` real en runtime (los 5 meta-agentes).
    # False: agente-tarea (extract/verify); el spec es solo cobertura de rubric y la
    #        estructura se valida sobre el USER (su prompt completo vive ahí).
    runtime_split: bool = True

    def render_user(self) -> str:
        """Renderiza el USER con los inputs-sonda. Usado por el rubric de contexto."""
        return self.build_user(**self.probe)

    def structure_target(self) -> str:
        """Texto donde el rubric estructural busca sus tokens: el system robusto
        para meta-agentes; el USER completo para agentes-tarea de cobertura."""
        return self.system if self.runtime_split else self.render_user()
