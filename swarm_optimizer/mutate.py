"""El 'Evolution AI': diagnóstico + mutación por diff/patch + cross-pollination + fresh.

Los prompts de cada meta-agente viven en `swarm_optimizer.prompts` (SYSTEM robusto +
build_user). Aquí queda la lógica de orquestación y parseo. Las funciones puras
(apply_diff, parse_search_replace, apply_validation_patch, merge_genomes) son testeables
sin API; las que llaman al LLM reciben un `client` inyectable y usan rol `system` real.
"""
from __future__ import annotations

import dataclasses
import json
import re
from copy import deepcopy

from swarm_optimizer.genome import Genome, ValidationConfig
from swarm_optimizer.prompts import (
    SYSTEM_CROSS, SYSTEM_DIAGNOSE, SYSTEM_FRESH, SYSTEM_META_REVIEW, SYSTEM_PROPOSE,
    build_user_cross, build_user_diagnose, build_user_fresh, build_user_meta_review,
    build_user_propose,
)

_SR_RE = re.compile(
    r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


# ── puros ─────────────────────────────────────────────────────────── #
def apply_diff(text: str, search: str, replace: str) -> tuple[str, bool]:
    # 1. exact match (rápido y determinista)
    if search in text:
        return text.replace(search, replace, 1), True
    # 2. tolerante a whitespace: tokens de search separados por \s+. Atrapa la
    #    re-indentación del LLM SIN el riesgo de corrupción semántica de un fuzzy
    #    por similitud (un noop es más seguro que un mal-apply plausible).
    tokens = search.split()
    if not tokens:
        return text, False
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in tokens))
    m = pattern.search(text)
    if m:
        return text[: m.start()] + replace + text[m.end():], True
    return text, False


def parse_search_replace(text: str) -> tuple[str, str] | None:
    m = _SR_RE.search(text or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def apply_validation_patch(vc: ValidationConfig, patch: dict) -> ValidationConfig:
    valid = {f.name for f in dataclasses.fields(ValidationConfig)}
    updates = {k: v for k, v in patch.items() if k in valid}
    return dataclasses.replace(vc, **updates)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── con LLM (client inyectable, rol `system` real) ────────────────── #
def diagnose(fps: list[str], fns: list[str], client, model: str = "gemini-2.5-flash",
             meta_review_text: str = "") -> str:
    user = build_user_diagnose(fps, fns, meta_review_text=meta_review_text)
    try:
        resp = client.models.generate_content(
            model=model, contents=user, system=SYSTEM_DIAGNOSE)
        return resp.text or ""
    except Exception:
        return "(diagnóstico no disponible)"


def meta_review(fps: list[str], fns: list[str], client,
                model: str = "gemini-2.5-flash") -> str:
    """Meta-review agent (AI Co-Scientist): síntesis de patrones transversales
    sobre los errores agregados del top de la población. 1 llamada por championship."""
    user = build_user_meta_review(fps, fns)
    try:
        resp = client.models.generate_content(
            model=model, contents=user, system=SYSTEM_META_REVIEW)
        return resp.text or ""
    except Exception:
        return ""


def propose(genome: Genome, diagnosis: str, client,
            model: str = "gemini-2.5-flash", memory: str = "",
            force_artifact: str | None = None,
            examples: list[dict] | None = None, gap_hint: str = "") -> tuple[Genome, str, str | None]:
    """Devuelve (hijo, mutation_type, artifact_touched).
    mutation_type ∈ {'diff_a','diff_b','noop'}.

    memory: intentos previos del linaje (memoria retrospectiva v1, MLEvolve).
    force_artifact: 'A'|'B' — restricción del meta-agente sobre qué artefacto mutar.
    examples: tripletas concretas (artículo/predicción/gold) → reflexión GEPA.
    gap_hint: pista del gap directed→undirected (cuánto cuesta hoy la dirección)."""
    user = build_user_propose(genome, diagnosis, memory=memory, examples=examples,
                              gap_hint=gap_hint, force_artifact=force_artifact)
    try:
        resp = client.models.generate_content(
            model=model, contents=user, system=SYSTEM_PROPOSE)
        data = json.loads(_strip_fences(resp.text or ""))
    except Exception:
        return deepcopy(genome), "noop", None

    child = deepcopy(genome)
    artifact = data.get("artifact")

    if artifact == "A":
        sr = parse_search_replace(data.get("diff", ""))
        if not sr:
            return child, "noop", None
        new_text, ok = apply_diff(child.prompt_text, sr[0], sr[1])
        if not ok:
            return child, "noop", None
        child.prompt_text = new_text
        return child, "diff_a", "A"

    if artifact == "B":
        patch = data.get("patch", {})
        if not isinstance(patch, dict) or not patch:
            return child, "noop", None
        child.validation = apply_validation_patch(child.validation, patch)
        return child, "diff_b", "B"

    return child, "noop", None


# Componentes del genoma que se evolucionan (para el merge por componente, estilo GEPA).
# model se omite a propósito (es el tier, no se muta).
MERGE_COMPONENTS = ("prompt_text", "few_shots", "architecture", "verify", "validation", "analysis")


def merge_genomes(parent_a: dict, parent_b: dict, ancestor: dict,
                  score_a: float, score_b: float) -> dict:
    """System-aware merge (GEPA proposer/merge.py) por componente, sobre genomas-dict.

    Para cada componente, respecto al ancestro común:
    - si solo un padre lo cambió → adopta ese cambio,
    - si ambos lo cambiaron → toma el del padre con mayor score agregado,
    - si ninguno lo cambió → conserva el del ancestro.
    Sintetiza dos descendientes que mejoraron componentes DISTINTOS (ej: A de uno + B del otro)."""
    from copy import deepcopy
    child = deepcopy(ancestor)
    for key in MERGE_COMPONENTS:
        a, b, anc = parent_a.get(key), parent_b.get(key), ancestor.get(key)
        a_diff, b_diff = (a != anc), (b != anc)
        if a_diff and not b_diff:
            child[key] = deepcopy(a)
        elif b_diff and not a_diff:
            child[key] = deepcopy(b)
        elif a_diff and b_diff:
            child[key] = deepcopy(a if score_a >= score_b else b)
        # ninguno cambió → se queda el del ancestro (ya está en child)
    return child


def cross_pollinate(parent1: Genome, parent2: Genome, client,
                    model: str = "gemini-2.5-flash",
                    parent1_metrics: dict | None = None,
                    parent2_metrics: dict | None = None,
                    diagnosis: str = "") -> tuple[Genome, str, str]:
    """Fusión dirigida por evidencia (ya NO ciega): combina los prompts de dos padres
    top guiándose por sus métricas (P/R/f05) y el diagnóstico vigente."""
    user = build_user_cross(parent1.prompt_text, parent2.prompt_text,
                            parent1_metrics or {}, parent2_metrics or {}, diagnosis)
    child = deepcopy(parent1)
    try:
        resp = client.models.generate_content(
            model=model, contents=user, system=SYSTEM_CROSS)
        new_prompt = _strip_fences(resp.text or "")
        if new_prompt:
            child.prompt_text = new_prompt
    except Exception:
        pass
    return child, "cross", "A"


def fresh_genome(parent1: Genome, parent2: Genome, client,
                 model: str = "gemini-2.5-flash",
                 diagnosis: str = "", meta_review_text: str = "") -> tuple[Genome, str, str]:
    """Operador 'fresh' (Evolution agent de AI Co-Scientist): genera un prompt nuevo
    desde cero inspirado en los top, atacando el diagnóstico y los patrones del
    meta-revisor (ya NO ciego). Fail-safe: noop."""
    user = build_user_fresh(parent1.prompt_text, parent2.prompt_text,
                            diagnosis=diagnosis, meta_review=meta_review_text)
    child = deepcopy(parent1)
    try:
        resp = client.models.generate_content(
            model=model, contents=user, system=SYSTEM_FRESH)
        new_prompt = _strip_fences(resp.text or "")
        if not new_prompt:
            return child, "noop", None
        child.prompt_text = new_prompt
    except Exception:
        return child, "noop", None
    return child, "fresh", "A"
