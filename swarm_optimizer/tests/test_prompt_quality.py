"""Rubric de calidad de prompts por agente (sin costo de API).

Garantiza, para CADA agente del sistema:
1. structure  — el SYSTEM robusto (o el prompt completo en agentes-tarea) contiene los
   tokens que evidencian ROLE/TASK/OUTPUT_FORMAT/CONSTRAINTS.
2. context    — el USER renderizado incluye TODA la info necesaria (sentinelas únicos por
   input). Si un builder ignora un arg (ej. cross ignora diagnosis → ciego), falla.
3. robustness — los meta-agentes tienen un system no trivial con rubric suficiente.

El call-site real (que el system viaja como rol `system`) se testea en test_mutate.py.
"""
from __future__ import annotations

import pytest

from swarm_optimizer.prompts import ALL_SPECS


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.agent)
def test_system_prompt_has_required_structure(spec):
    target = spec.structure_target()
    for token in spec.required_structure:
        assert token in target, (
            f"{spec.agent}: el prompt no contiene el token estructural «{token}». "
            f"Las 4 dimensiones (ROLE/TASK/OUTPUT_FORMAT/CONSTRAINTS) deben ser explícitas."
        )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.agent)
def test_user_message_carries_full_context(spec):
    user = spec.render_user()
    for sentinel in spec.required_context:
        assert sentinel in user, (
            f"{spec.agent}: el contexto «{sentinel}» NO llega al mensaje user. "
            f"El call-site debe inyectar toda la info disponible (no agentes ciegos)."
        )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.agent)
def test_meta_agents_have_robust_system(spec):
    if not spec.runtime_split:
        pytest.skip("agente-tarea: su prompt es el artefacto evolucionado (cobertura por user)")
    assert len(spec.system.strip()) >= 200, (
        f"{spec.agent}: system demasiado corto ({len(spec.system)} chars) para ser robusto"
    )
    assert len(spec.required_structure) >= 4, (
        f"{spec.agent}: rubric estructural insuficiente (<4 tokens)"
    )


def test_all_meta_agents_are_covered():
    """Ningún meta-agente del self-evolve queda sin spec (regresión de cobertura)."""
    agents = {s.agent for s in ALL_SPECS}
    required = {"diagnose", "meta_review", "propose", "cross", "fresh"}
    missing = required - agents
    assert not missing, f"meta-agentes sin PromptSpec: {missing}"
