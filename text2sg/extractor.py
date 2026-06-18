# experiments/text2sg/extractor.py
from __future__ import annotations
import json
import os
import re
import time

import pandas as pd
from google import genai

from text2sg.config import Config
from text2sg.rubric import load_union
from text2sg.validation import apply_validation


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
        from text2sg.analysis import build_analysis
        block = build_analysis(union, body, analysis_cfg)
        if block:
            parts.append(block)
    elif config.architecture in ("given_entities", "end2end") and union:
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


_NER_PROMPT = """\
You are a Chilean political NER system. Given a press article (in Spanish), identify all political actors who could participate in explicit political relations.

For each actor output:
- uid: sequential ID starting at U1
- canonical_name: the most complete formal name used in the article (e.g. "Gabriel Boric", not "el presidente")
- type: "roster_actor" (named politician, minister, senator) | "institutional_actor" (party, ministry, court, committee) | "non_roster_actor" (journalist, expert, business figure)

Only include actors who perform or receive political acts. Ignore abstract concepts, locations, and purely passing references.

Output only valid JSON, no markdown:
{{"actors": [{{"uid": "U1", "canonical_name": "...", "type": "..."}}]}}

ARTICLE:
{body}
"""


def extract_entities(body: str, model: str, client) -> tuple[dict, int]:
    """NER pass para el modo end2end: extrae actores políticos desde el texto.
    Retorna (union_dict, tokens). union_dict tiene el mismo formato que unions.json:
    {uid: {canonical_names: [name], type: str}}
    """
    prompt = _NER_PROMPT.format(body=body[:6000])
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        usage = getattr(response, "usage_metadata", None)
        tokens = 0
        if usage:
            tokens = (
                getattr(usage, "prompt_token_count", 0)
                + getattr(usage, "candidates_token_count", 0)
            )
        text = (response.text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        union = {
            a["uid"]: {
                "canonical_names": [a["canonical_name"]],
                "type": a.get("type", "non_roster_actor"),
            }
            for a in data.get("actors", [])
            if "uid" in a and "canonical_name" in a
        }
        return union, tokens
    except Exception:
        return {}, 0


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
    collect_trace: bool = False,
) -> dict:
    """
    Corre el extractor sobre un artículo. Retorna:
    {"article_id": ..., "entities": [...], "relations": [...], "tokens": int}

    Con `collect_trace=True` agrega "trace": {rel_raw, n_before_validation,
    n_after_validation, n_dropped} — el crudo del LLM (cómo "pensó") + qué descartó la
    validación determinista (artefacto B). Cuesta ~0 tokens (el crudo ya viene en la
    respuesta); solo se guarda en vez de descartarse.
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
            n_before = len(parsed.get("relations", []))

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
            out = {
                "article_id": article_id,
                "entities": parsed["entities"],
                "relations": parsed["relations"],
                "tokens": token_count,
            }
            if collect_trace:
                n_after = len(parsed["relations"])
                out["trace"] = {
                    "rel_raw": text[:4000],
                    "n_before_validation": n_before,
                    "n_after_validation": n_after,
                    "n_dropped": n_before - n_after,
                }
            return out
        except Exception as e:
            import sys
            print(f"[extractor] attempt {attempt+1}/{retry} failed: {e}", file=sys.stderr)
            if attempt == retry - 1:
                err = {"article_id": article_id, "entities": [], "relations": [], "tokens": 0}
                if collect_trace:
                    err["trace"] = {"rel_raw": "", "n_before_validation": 0,
                                    "n_after_validation": 0, "n_dropped": 0, "error": str(e)}
                return err
            time.sleep(2 ** attempt)

    return {"article_id": article_id, "entities": [], "relations": [], "tokens": 0}


def run_extraction(
    article_ids: list[str],
    articles_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    union_map: dict[str, dict],
    config: Config,
    client=None,
    collect_trace: bool = False,
) -> tuple[list[dict], int]:
    """
    Corre la extracción sobre una lista de artículos.
    Retorna (predictions, total_tokens).

    client: cliente LLM inyectable (genai.Client u OllamaClient). Si None, usa genai.
    collect_trace: si True, cada pred lleva "trace" con el crudo del LLM + n_actores del NER
        (para auditar a escala sin gold — ver proxy_metrics()).
    """
    client = client or _client()
    few_shot_examples = _load_few_shot_examples(
        config.few_shots, articles_df, gold_df
    )

    predictions = []
    total_tokens = 0
    _end2end = getattr(config, "architecture", None) == "end2end"
    for art_id in article_ids:
        row = articles_df[articles_df["article_id"] == art_id]
        if row.empty:
            continue
        body = str(row.iloc[0]["body"] or "")
        ner_tokens = 0
        if _end2end:
            union, ner_tokens = extract_entities(body, config.model, client)
        else:
            union = union_map.get(art_id, {})
        pred = extract_article(
            art_id, body, union, config, few_shot_examples, client,
            collect_trace=collect_trace,
        )
        pred["tokens"] = pred.get("tokens", 0) + ner_tokens
        if collect_trace:
            pred.setdefault("trace", {})["n_actors_ner"] = len(union)
        predictions.append(pred)
        total_tokens += pred.get("tokens", 0)

    return predictions, total_tokens


def proxy_metrics(predictions: list[dict]) -> dict:
    """Métricas-proxy SIN gold (para corridas sobre el corpus real, donde no hay verdad
    plantada). Detectan deriva/degeneración del extractor: relaciones por artículo, % de
    artículos sin relaciones (proxy de FN/abstención), distribución de act_types, y drops de
    validación si hay trace.
    """
    n = len(predictions)
    if not n:
        return {"n": 0}
    rels = [len(p.get("relations", [])) for p in predictions]
    n_zero = sum(1 for r in rels if r == 0)
    act_types: dict[str, int] = {}
    for p in predictions:
        for r in p.get("relations", []):
            at = r.get("act_type", "?")
            act_types[at] = act_types.get(at, 0) + 1
    out = {
        "n": n,
        "rel_per_article_mean": round(sum(rels) / n, 3),
        "pct_zero_relations": round(n_zero / n, 3),
        "total_relations": sum(rels),
        "act_type_dist": dict(sorted(act_types.items(), key=lambda kv: -kv[1])),
    }
    dropped = [p["trace"]["n_dropped"] for p in predictions if "trace" in p and "n_dropped" in p["trace"]]
    if dropped:
        out["validation_dropped_total"] = sum(dropped)
    return out
