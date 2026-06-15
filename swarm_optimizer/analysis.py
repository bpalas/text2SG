"""
Artefacto C: Analysis Tool determinista (pre-extracción, costo $0).
Transforma el roster por-artículo (union) + el body en un bloque estructurado
que se inyecta en el prompt. Mueve hechos por-artículo fuera del prompt frágil.
"""
from __future__ import annotations

import re

from swarm_optimizer.genome import SEED_ALLOWED_ACT_TYPES
from swarm_optimizer.rubric import normalize

# Vocabulario canónico de act_type (los permitidos en la semilla, sin co_occurs).
CANONICAL_ACT_TYPES = list(SEED_ALLOWED_ACT_TYPES)

# Equivalencias act_type no-canónico → canónico (auditoría gold 2026-06-10: ~20%
# del gold usa tipos no canónicos que castigan Act_acc sin afectar P_rel/R_rel).
ACT_TYPE_CANON = {
    "kill": "attacks", "kills": "attacks", "attack": "attacks", "ataca": "attacks",
    "defends": "endorses", "defend": "endorses", "supports": "endorses",
    "respalda": "endorses", "apoya": "endorses",
    "criticizes": "accuses", "criticize": "accuses", "critica": "accuses",
    "meet_with": "negotiates_with", "meets_with": "negotiates_with",
    "se_reune_con": "negotiates_with",
    "appoints": "endorses", "names": "endorses", "nombra": "endorses",
    "calls_for": "calls_on", "urges": "calls_on", "llama_a": "calls_on",
    "opposes": "competes_with", "se_opone": "competes_with",
}

# Palabras de rol → etiqueta normalizada (para los hints de rol, Task 3).
DEFAULT_ROLE_KEYWORDS = {
    "abogado": "abogado/defensa", "abogada": "abogado/defensa",
    "defensor": "abogado/defensa", "defensa": "abogado/defensa",
    "imputado": "imputado", "imputada": "imputado", "acusado": "imputado",
    "fiscal": "fiscal",
    "ministro": "ministro", "ministra": "ministro",
    "diputado": "diputado", "diputada": "diputado",
    "senador": "senador", "senadora": "senador",
    "presidente": "presidente", "presidenta": "presidente",
    "alcalde": "alcalde", "alcaldesa": "alcalde",
}

# Términos de dominio deportivo (gate de no-político, Task 4).
FOOTBALL_TERMS = ["gol", "partido", "futbol", "club", "seleccion", "delantero",
                  "arquero", "copa", "estadio", "entrenador"]


def canon_act_type(act: str | None) -> str:
    """Mapea un act_type a su forma canónica (idempotente si ya es canónico)."""
    a = (act or "").strip().lower()
    return ACT_TYPE_CANON.get(a, a)


def _real_actors(union: dict) -> list[tuple[str, dict]]:
    """Actores no-NIL del union, como [(uid, ent), ...]."""
    return [(uid, ent) for uid, ent in union.items() if ent.get("type") != "NIL"]


def _canon_name(ent: dict) -> str:
    names = ent.get("canonical_names") or ["?"]
    return names[0]


def _name_spans(names: list[str], text_norm: str) -> list[tuple[int, int]]:
    """Spans (start,end) de cualquiera de `names` en text_norm, con límite de palabra
    (evita que 'Boric' matchee dentro de 'Borico')."""
    spans = []
    for n in names:
        nn = normalize(n)
        if not nn:
            continue
        for m in re.finditer(rf"\b{re.escape(nn)}\b", text_norm):
            spans.append((m.start(), m.end()))
    return spans


def _count_mentions(names: list[str], text_norm: str) -> int:
    """Conteo de menciones físicas: fusiona spans solapados, así el apellido dentro
    del nombre completo ('Hermosilla' en 'Luis Hermosilla') cuenta una sola vez."""
    spans = sorted(_name_spans(names, text_norm))
    if not spans:
        return 0
    count, cur_end = 1, spans[0][1]
    for s, e in spans[1:]:
        if s < cur_end:                # solapado → misma mención física
            cur_end = max(cur_end, e)
        else:
            count += 1
            cur_end = e
    return count


def _actor_dossier(union: dict) -> str:
    lines = []
    for _uid, ent in _real_actors(union):
        aliases = ", ".join(ent.get("surfaces") or []) or "—"
        lines.append(f"  {_canon_name(ent)} (tipo: {ent.get('type', '?')}; alias: {aliases})")
    if not lines:
        return ""
    return "ACTORES (usa el nombre canónico, nunca el código U1/U2):\n" + "\n".join(lines)


def _alias_map(union: dict) -> str:
    # surface_norm -> {canonicals que lo reclaman}; un alias compartido por dos
    # actores (ej: 'Hermosilla') es AMBIGUO, no se mapea a uno solo.
    surf_to_canon: dict[str, set[str]] = {}
    display: dict[str, str] = {}
    for _uid, ent in _real_actors(union):
        canon = _canon_name(ent)
        for surf in ent.get("surfaces") or []:
            sn = normalize(surf)
            if not sn or sn == normalize(canon):
                continue
            surf_to_canon.setdefault(sn, set()).add(canon)
            display.setdefault(sn, surf)
    lines = []
    for sn, canons in surf_to_canon.items():
        if len(canons) == 1:
            lines.append(f"  '{display[sn]}' → {next(iter(canons))}")
        else:
            lines.append(f"  '{display[sn]}' → AMBIGUO ({' | '.join(sorted(canons))}), "
                         f"usa el contexto")
    if not lines:
        return ""
    return "MAPA DE ALIAS (normaliza menciones al canónico):\n" + "\n".join(lines)


def _act_type_canon_block() -> str:
    canon = ", ".join(CANONICAL_ACT_TYPES)
    mappings = "; ".join(f"{k}→{v}" for k, v in ACT_TYPE_CANON.items())
    return (f"ACT_TYPES CANÓNICOS (usa SOLO estos): {canon}.\n"
            f"Si dudás, mapea: {mappings}.")


def _role_hints(union: dict, body: str, role_keywords: dict | None, window: int | None = None) -> str:
    """Detecta el rol probable de cada actor por palabras-clave, con scoping POR ORACIÓN
    (no por ventana de chars): un rol solo se atribuye a actores presentes en la misma
    oración, y al match más específico (nombre más largo) cuando hay varios — así
    'imputado' no se filtra a 'Juan Pablo Hermosilla' desde otra oración. Marca
    AMBIGÜEDAD cuando dos actores comparten apellido. `window` se conserva por compat
    de firma/serialización pero ya no se usa."""
    role_keywords = role_keywords or DEFAULT_ROLE_KEYWORDS
    actors = _real_actors(union)
    if not actors:
        return ""

    # Apellidos compartidos → ambigüedad.
    surnames: dict[str, list[str]] = {}
    for _uid, ent in actors:
        sn = normalize(_canon_name(ent)).split()
        if sn:
            surnames.setdefault(sn[-1], []).append(_canon_name(ent))
    ambiguous = {s for s, names in surnames.items() if len(names) > 1}

    roles_by_canon: dict[str, set[str]] = {_canon_name(ent): set() for _uid, ent in actors}
    for sentence in re.split(r"[.!?]\s+", body):
        sent_norm = normalize(sentence)
        present = []   # (canon, especificidad = nº de tokens del mejor match)
        for _uid, ent in actors:
            names = (ent.get("surfaces") or []) + (ent.get("canonical_names") or [])
            best = 0
            for n in names:
                nn = normalize(n)
                if nn and re.search(rf"\b{re.escape(nn)}\b", sent_norm):
                    best = max(best, len(nn.split()))
            if best:
                present.append((_canon_name(ent), best))
        if not present:
            continue
        found = {role for kw, role in role_keywords.items()
                 if re.search(rf"\b{re.escape(normalize(kw))}\b", sent_norm)}
        if not found:
            continue
        max_spec = max(b for _, b in present)
        top = [c for c, b in present if b == max_spec]
        targets = top if len(top) == 1 else [c for c, _ in present]
        for c in targets:
            roles_by_canon[c] |= found

    lines = []
    for _uid, ent in actors:
        canon = _canon_name(ent)
        roles = roles_by_canon[canon]
        surname = normalize(canon).split()[-1] if normalize(canon).split() else ""
        flag = "  [AMBIGÜEDAD: comparte apellido con otro actor]" if surname in ambiguous else ""
        role_str = ", ".join(sorted(roles)) if roles else "rol no detectado"
        lines.append(f"  {canon}: {role_str}{flag}")

    return "ROLES DETECTADOS (heurística — desambigua atribución):\n" + "\n".join(lines)


def _direction_scaffold(body: str) -> str:
    """Detecta pasiva '<paciente> fue <participio> por <agente>' y emite la dirección
    correcta (from=agente, to=paciente). Usa el body crudo para preservar mayúsculas."""
    pat = re.compile(
        r"([A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*)*)"
        r"\s+fue\s+(?:\w+\s+){0,2}\w+\s+por\s+"   # participio con hasta 2 adverbios/modif.
        r"([A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*)*)"
    )
    hits = []
    for m in pat.finditer(body):
        patient, agent = m.group(1).strip(), m.group(2).strip()
        hits.append(f"  voz pasiva: from={agent}, to={patient}")
    if not hits:
        return ""
    return "DIRECCIÓN (corrige el sentido en pasiva):\n" + "\n".join(hits)


def _main_speaker(union: dict, body: str) -> str:
    body_norm = normalize(body)
    ranked = []
    for _uid, ent in _real_actors(union):
        names = (ent.get("surfaces") or []) + (ent.get("canonical_names") or [])
        count = _count_mentions(names, body_norm)
        ranked.append((count, _canon_name(ent)))
    if not ranked:
        return ""
    ranked.sort(reverse=True)
    top_count, top_name = ranked[0]
    if top_count == 0:
        return ""
    return (f"HABLANTE PRINCIPAL (más mencionado): {top_name} ({top_count} menciones) — "
            f"sujeto probable de varias relaciones; no lo omitas.")


def _comention_pairs(union: dict, body: str) -> str:
    actors = _real_actors(union)
    pairs: set[tuple[str, str]] = set()
    for sentence in re.split(r"[.!?]\s+", body):
        sent_norm = normalize(sentence)
        present = []
        for _uid, ent in actors:
            names = (ent.get("surfaces") or []) + (ent.get("canonical_names") or [])
            if _name_spans(names, sent_norm):     # límite de palabra, sin substring
                present.append(_canon_name(ent))
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                if present[a] == present[b]:       # actores con el mismo canónico (dup) → no es par
                    continue
                pairs.add(tuple(sorted((present[a], present[b]))))
    if not pairs:
        return ""
    lines = [f"  {a} ↔ {b}" for a, b in sorted(pairs)]
    return "PARES CO-MENCIONADOS (candidatos a relación, verifica el verbo):\n" + "\n".join(lines)


def _domain_gate(union: dict, body: str) -> str:
    body_norm = normalize(body)
    hits = [t for t in FOOTBALL_TERMS if t in body_norm]
    if len(hits) >= 2:
        return ("AVISO DE DOMINIO: el texto parece deportivo/no-político (términos: "
                + ", ".join(hits) + "). Sé MÁS estricto: solo relaciones políticas explícitas.")
    return ""


def build_analysis(union: dict, body: str, cfg) -> str:
    """Orquesta las secciones gateadas en un bloque único. Devuelve '' si el union
    está vacío o ninguna sección produce contenido."""
    if not union:
        return ""
    sections = []
    if cfg.emit_dossier:
        sections.append(_actor_dossier(union))
    if cfg.emit_alias_map:
        sections.append(_alias_map(union))
    if cfg.emit_role_hints:
        sections.append(_role_hints(union, body, cfg.role_keywords, cfg.role_window))
    if cfg.emit_direction_scaffold:
        sections.append(_direction_scaffold(body))
    if cfg.emit_main_speaker:
        sections.append(_main_speaker(union, body))
    if cfg.emit_comention_pairs:
        sections.append(_comention_pairs(union, body))
    if cfg.emit_act_type_canon:
        sections.append(_act_type_canon_block())
    if cfg.emit_domain_gate:
        sections.append(_domain_gate(union, body))

    sections = [s for s in sections if s]
    if not sections:
        return ""
    return "=== ANÁLISIS DE ACTORES ===\n" + "\n\n".join(sections) + "\n=== FIN ANÁLISIS ===\n"
