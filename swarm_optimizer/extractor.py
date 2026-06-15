# experiments/swarm_optimizer/extractor.py
from __future__ import annotations
import json
import os
import re
import time

import pandas as pd
from google import genai

from swarm_optimizer.config import Config
from swarm_optimizer.rubric import load_union
from swarm_optimizer.validation import apply_validation


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key)


def parse_llm_output(text: str) -> dict:
    """Parsea el JSON de salida de Gemini. Maneja markdown fences y texto de
    razonamiento previo al JSON (arquitectura 'debate'). En error → vacío."""
    text = text.strip()
    # strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return {
            "entities": data.get("entities", []),
            "relations": data.get("relations", []),
        }
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: extraer el ÚLTIMO objeto JSON del texto (el debate interno puede
    # emitir razonamiento antes de la respuesta final).
    start = text.rfind("{\"entities\"")
    if start == -1:
        start = text.find("{")
    if start != -1:
        snippet = text[start: text.rfind("}") + 1]
        try:
            data = json.loads(snippet)
            return {
                "entities": data.get("entities", []),
                "relations": data.get("relations", []),
            }
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"entities": [], "relations": []}


def build_prompt(
    config: Config,
    body: str,
    union: dict,
    few_shot_examples: list[dict],
) -> str:
    """Construye el prompt final para Gemini según la arquitectura del config."""
    parts = [config.prompt_text.strip(), ""]

    if few_shot_examples:
        parts.append("=== EJEMPLOS ===")
        for ex in few_shot_examples:
            parts.append(f"\nARTÍCULO:\n{ex['body']}\n")
            parts.append(f"RESPUESTA CORRECTA:\n{json.dumps(ex['output'], ensure_ascii=False)}\n")
        parts.append("=== FIN EJEMPLOS ===\n")

    if config.architecture == "debate":
        parts.append(_DEBATE_INSTRUCTIONS)

    analysis_cfg = getattr(config, "analysis", None)
    if analysis_cfg is not None and union:
        from swarm_optimizer.analysis import build_analysis
        block = build_analysis(union, body, analysis_cfg)
        if block:
            parts.append(block)
    elif config.architecture == "given_entities" and union:
        actor_list = "\n".join(
            f"  {uid}: {ent.get('canonical_names', ['?'])[0]} (tipo: {ent.get('type', '?')})"
            for uid, ent in union.items()
            if ent.get("type") != "NIL"
        )
        parts.append(f"ACTORES PRESENTES EN EL ARTÍCULO:\n{actor_list}\n")

    parts.append(f"ARTÍCULO A ANALIZAR:\n{body}\n\nRESPUESTA JSON:")
    return "\n".join(parts)


def _load_few_shot_examples(
    few_shot_ids: list[str],
    articles_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> list[dict]:
    """Formatea los few-shots como pares (body, relaciones gold)."""
    examples = []
    for art_id in few_shot_ids:
        row = articles_df[articles_df["article_id"] == art_id]
        rels = gold_df[gold_df["article_id"] == art_id]
        if row.empty or rels.empty:
            continue
        output = {
            "entities": [],
            "relations": [
                {
                    "from_entity": r.u_from, "to_entity": r.u_to,
                    "act_type": r.act_type, "polarity": r.polarity,
                    "issue": r.issue, "evidence_quote": r.evidence_quote,
                }
                for r in rels.itertuples()
            ],
        }
        examples.append({"body": row.iloc[0]["body"][:1500], "output": output})
    return examples


_DEBATE_INSTRUCTIONS = """\
MODO DEBATE INTERNO (Societies of Thought) — antes de responder, razona en tres voces:
1. PROPONENTE: lista todas las relaciones candidatas que encuentres en el texto.
2. CRÍTICO: objeta cada candidata — ¿hay un verbo conector explícito entre los actores?,
   ¿la cita literal soporta la dirección (quién actúa sobre quién)?, ¿la polaridad es la
   que el texto reporta, no la que se infiere?
3. ÁRBITRO: emite SOLO las relaciones que sobreviven a las objeciones.
Puedes escribir el debate como texto libre, pero tu respuesta DEBE TERMINAR con el
JSON final válido en una sola línea (ese JSON es lo único que será evaluado).
"""

_VERIFY_PROMPT = """\
Revisa estas relaciones extraídas de un artículo. Elimina las que NO estén soportadas
literalmente por el texto y corrige dirección/polaridad cuando el texto lo indique.

ARTÍCULO:
{body}

RELACIONES (JSON):
{relations}

Devuelve ÚNICAMENTE el JSON corregido: {{"relations": [...]}}, sin markdown.
"""


def verify_relations(relations: list[dict], body: str, model: str, client,
                     temperatures=(0.0, 0.2)) -> tuple[list[dict], int]:
    """Verificación agéntica (RoboPhD). Retorna (relations, tokens).
    Fail-safe: ante error conserva las originales con tokens=0."""
    if not relations:
        return relations, 0
    prompt = _VERIFY_PROMPT.format(
        body=body[:4000],
        relations=json.dumps(relations, ensure_ascii=False),
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        usage = getattr(response, "usage_metadata", None)
        tokens = 0
        if usage:
            tokens = (
                getattr(usage, "prompt_token_count", 0)
                + getattr(usage, "candidates_token_count", 0)
            )
        parsed = parse_llm_output(response.text or "")
        verified = parsed.get("relations", [])
        return (verified if verified else relations), tokens
    except Exception:
        return relations, 0


def extract_article(
    article_id: str,
    body: str,
    union: dict,
    config: Config,
    few_shot_examples: list[dict],
    client: genai.Client,
    retry: int = 3,
) -> dict:
    """
    Corre el extractor sobre un artículo. Retorna:
    {"article_id": ..., "entities": [...], "relations": [...], "tokens": int}
    """
    prompt = build_prompt(config, body, union, few_shot_examples)

    for attempt in range(retry):
        try:
            response = client.models.generate_content(
                model=config.model,
                contents=prompt,
            )
            text = response.text or ""
            parsed = parse_llm_output(text)

            # Verificación agéntica opcional (gateada por flag del genoma)
            verify_tokens = 0
            if getattr(config, "verify", False):
                parsed["relations"], verify_tokens = verify_relations(
                    parsed.get("relations", []), body, config.model, client
                )

            # Validación determinista (artefacto B)
            vc = getattr(config, "validation", None)
            if vc is not None:
                parsed = apply_validation(parsed, body, union, vc)

            usage = getattr(response, "usage_metadata", None)
            token_count = verify_tokens
            if usage:
                token_count += (
                    getattr(usage, "prompt_token_count", 0)
                    + getattr(usage, "candidates_token_count", 0)
                )
            return {
                "article_id": article_id,
                "entities": parsed["entities"],
                "relations": parsed["relations"],
                "tokens": token_count,
            }
        except Exception as e:
            import sys
            print(f"[extractor] attempt {attempt+1}/{retry} failed: {e}", file=sys.stderr)
            if attempt == retry - 1:
                return {"article_id": article_id, "entities": [], "relations": [], "tokens": 0}
            time.sleep(2 ** attempt)

    return {"article_id": article_id, "entities": [], "relations": [], "tokens": 0}


def run_extraction(
    article_ids: list[str],
    articles_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    union_map: dict[str, dict],
    config: Config,
    client=None,
) -> tuple[list[dict], int]:
    """
    Corre la extracción sobre una lista de artículos.
    Retorna (predictions, total_tokens).

    client: cliente LLM inyectable (genai.Client u OllamaClient). Si None, usa genai.
    """
    client = client or _client()
    few_shot_examples = _load_few_shot_examples(
        config.few_shots, articles_df, gold_df
    )

    predictions = []
    total_tokens = 0
    for art_id in article_ids:
        row = articles_df[articles_df["article_id"] == art_id]
        if row.empty:
            continue
        body = str(row.iloc[0]["body"] or "")
        union = union_map.get(art_id, {})
        pred = extract_article(
            art_id, body, union, config, few_shot_examples, client
        )
        predictions.append(pred)
        total_tokens += pred.get("tokens", 0)

    return predictions, total_tokens
