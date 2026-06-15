"""Spec de cobertura del agente-tarea EXTRACT (la semilla del extractor).

`runtime_split=False`: el prompt de extracción es el ARTEFACTO QUE SE EVOLUCIONA
(genome.prompt_text), no una maquinaria fija; su runtime vive en extractor.build_prompt
y NO se toca (CLAUDE.md: no romper la comparabilidad del gold). Este spec solo da
COBERTURA de rubric a la semilla: valida que el prompt-semilla cumpla los estándares
estructurales mínimos (rol, campos, vocabulario, dirección, formato de salida).
"""
from __future__ import annotations

from swarm_optimizer.config import SEED_PROMPT
from swarm_optimizer.prompts.base import PromptSpec


def build_user_extract(body: str, actors: str) -> str:
    """Mirror ligero de extractor.build_prompt (given_entities): semilla + actores +
    artículo. Solo para el rubric de cobertura; el runtime real está en extractor.py."""
    return (
        f"{SEED_PROMPT.strip()}\n\n"
        f"ACTORES PRESENTES EN EL ARTÍCULO:\n{actors}\n\n"
        f"ARTÍCULO A ANALIZAR:\n{body}\n\nRESPUESTA JSON:"
    )


EXTRACT_SPEC = PromptSpec(
    agent="extract",
    system="",
    build_user=build_user_extract,
    required_structure=(
        "extractor de relaciones políticas chilenas",   # ROLE
        "evidence_quote",                                # campo de salida
        "act_type",                                      # vocabulario de actos
        "Regla de dirección",                            # CONSTRAINT
        '{"entities"',                                   # OUTPUT_FORMAT (línea schema)
    ),
    required_context=("ART_BODY_SENTINEL_X9", "ACTORS_SENTINEL_X9"),
    probe={"body": "ART_BODY_SENTINEL_X9", "actors": "ACTORS_SENTINEL_X9"},
    runtime_split=False,
)
