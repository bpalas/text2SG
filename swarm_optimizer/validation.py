"""
Artefacto B: capa de validación determinista (post-proceso puro, costo $0).
Limpia el output del LLM antes de scoring. Sube precisión sin gastar tokens.
"""
from __future__ import annotations

from swarm_optimizer.genome import ValidationConfig
from swarm_optimizer.rubric import normalize

# act_type -> polaridad esperada (coherencia)
_POLARITY_MAP = {
    "attacks": "negative",
    "accuses": "negative",
    "questions": "negative",
    "competes_with": "negative",
    "distances_from": "negative",
    "endorses": "positive",
    "allies_with": "positive",
    "calls_on": "neutral",
    # gold 2026-06-10: negotiates_with es 73% positive (8/11), no neutral
    "negotiates_with": "positive",
    "co_occurs": "neutral",
}


# Construcciones de RECEPCIÓN: el sujeto gramatical es el RECEPTOR, el agente
# (true from) va DESPUÉS de la preposición 'de'. "X recibió el apoyo de Y" → Y→X.
# Cada marcador se busca como substring; el agente está tras el " de " siguiente.
_RECEPTION_MARKERS = (
    "recibio el apoyo",
    "recibio el respaldo",
    "recibio el aval",
    "conto con el apoyo",
    "conto con el respaldo",
    "conto con el aval",
)

# Palabras vacías que NO sirven para emparejar entidad↔texto por token.
_STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "y", "en", "a", "con", "su", "sus",
    "un", "una", "que", "se", "por", "para",
}


def _entity_tokens(name: str) -> set[str]:
    """Tokens significativos (len>=4, no stopword) del nombre normalizado."""
    return {w for w in name.split() if len(w) >= 4 and w not in _STOPWORDS}


def _shares_token(name: str, text: str) -> bool:
    """True si algún token significativo de la entidad aparece (como prefijo,
    tolerando plural/truncación: 'artesanales' ~ 'artesanal') en el texto."""
    text_tokens = [w for w in text.split() if len(w) >= 4]
    for tok in _entity_tokens(name):
        for w in text_tokens:
            # Igualdad o prefijo compartido en cualquier dirección (plural/truncación).
            if w == tok or w.startswith(tok) or tok.startswith(w):
                return True
    return False


def _maybe_swap_direction(rel: dict) -> dict:
    """Corrige inversiones de dirección de forma determinista ($0).

    1) Pasiva 'X fue criticado por Y': el agente va DESPUÉS de 'por'.
    2) Recepción 'X recibió el apoyo de Y' / 'X contó con el respaldo de Y':
       el sujeto X es el RECEPTOR y el agente Y va DESPUÉS del 'de' que sigue
       al marcador. El modelo tiende a emitir from=X (sujeto) → se invierte.
    En ambos casos sólo se reordena el par; nunca se descarta la relación
    (no afecta recall)."""
    quote = normalize(rel.get("evidence_quote", ""))
    f = normalize(rel.get("from_entity", ""))
    t = normalize(rel.get("to_entity", ""))
    if not f or not t:
        return rel

    # Caso 2: recepción ("... <marcador> ... de <agente>"). Emparejado por token
    # (tolera plural/truncación de la cita); el agente real va tras el 'de'.
    for marker in _RECEPTION_MARKERS:
        m = quote.find(marker)
        if m == -1:
            continue
        de = quote.find(" de ", m + len(marker))
        if de == -1:
            continue
        before, after = quote[:de], quote[de + 4:]
        # from emitido = receptor (antes del 'de'); to emitido = agente (después).
        if _shares_token(f, before) and _shares_token(t, after):
            return {**rel, "from_entity": rel["to_entity"], "to_entity": rel["from_entity"]}

    # Caso 1: pasiva con 'por'.
    if " por " not in quote:
        return rel
    idx = quote.find(" por ")
    before, after = quote[:idx], quote[idx + 5:]
    if f in before and t in after:
        return {**rel, "from_entity": rel["to_entity"], "to_entity": rel["from_entity"]}
    return rel


# P-015: mapa de etiquetas ordinales a scores de confianza
_CONFIDENCE_SCORES = {
    "explicit": 1.0,
    "strongly_implied": 0.7,
    "speculative": 0.4,
}


def _confidence_score(rel: dict) -> float:
    """Convierte la etiqueta ordinal de confianza a float. Default 1.0 si no existe."""
    raw = str(rel.get("confidence", "explicit")).lower().strip()
    return _CONFIDENCE_SCORES.get(raw, 1.0)


def apply_validation(parsed: dict, body: str, union: dict, vc: ValidationConfig) -> dict:
    """parsed = {'entities': [...], 'relations': [...]} -> versión limpia."""
    body_norm = normalize(body)
    relations = list(parsed.get("relations", []))
    cleaned: list[dict] = []
    seen: set[tuple] = set()

    for rel in relations:
        quote = rel.get("evidence_quote", "") or ""

        if vc.min_quote_len and len(quote.strip()) < vc.min_quote_len:
            continue

        if vc.require_evidence_substring and normalize(quote) not in body_norm:
            continue

        if vc.allowed_act_types is not None and rel.get("act_type") not in vc.allowed_act_types:
            continue

        if vc.normalize_passive_direction:
            rel = _maybe_swap_direction(rel)

        if vc.enforce_polarity_consistency:
            expected = _POLARITY_MAP.get(rel.get("act_type"))
            if expected is not None:
                rel = {**rel, "polarity": expected}

        # P-015: filtro de confianza ordinal
        if vc.min_confidence is not None:
            if _confidence_score(rel) < vc.min_confidence:
                continue

        # P-016: ambos actores deben aparecer en la evidence_quote
        if vc.require_both_in_quote:
            quote_norm = normalize(quote)
            frm = normalize(rel.get("from_entity", ""))
            to = normalize(rel.get("to_entity", ""))
            # matching por apellido (última palabra) como fallback
            frm_short = frm.split()[-1] if frm else ""
            to_short = to.split()[-1] if to else ""
            frm_ok = frm in quote_norm or (frm_short and frm_short in quote_norm)
            to_ok = to in quote_norm or (to_short and to_short in quote_norm)
            if not (frm_ok and to_ok):
                continue

        if vc.dedup:
            # P-008: clave por PAR (sin act_type). La rúbrica acredita cada par gold
            # una sola vez, así que una segunda relación del mismo (from, to) con otro
            # act_type es FP mecánico garantizado.
            key = (
                normalize(rel.get("from_entity", "")),
                normalize(rel.get("to_entity", "")),
            )
            if key in seen:
                continue
            seen.add(key)

        cleaned.append(rel)

    if vc.max_relations_per_article is not None:
        cleaned = cleaned[: vc.max_relations_per_article]

    return {"entities": parsed.get("entities", []), "relations": cleaned}
