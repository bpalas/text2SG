"""Eje de subconjunto formal/informal para el loop multi-gradiente (Quality-Diversity).

El dataset sintético v1 tiene `registro` vacío, así que el registro se deriva del `medio`:
los tabloides / radio / sátira son informales (lenguaje coloquial), el resto formal
(prensa de referencia y agencias). Matching por substring normalizado para sobrevivir al
mojibake del parquet ("Radio B�o-B�o") y a variantes de superficie.
"""
from __future__ import annotations

from swarm_optimizer.rubric import normalize

# Substrings (ya normalizados: minúscula, sin diacríticos) que marcan un medio informal.
# "radio b" cubre "Radio Bío-Bío" aunque el parquet traiga el carácter de reemplazo U+FFFD.
INFORMAL_MEDIO_SUBSTR = ("la cuarta", "the clinic", "radio b")


def registro_of(medio: str) -> str:
    """'informal' si el medio es tabloide/radio/sátira; 'formal' en otro caso."""
    m = normalize(medio or "")
    return "informal" if any(s in m for s in INFORMAL_MEDIO_SUBSTR) else "formal"


def split_ids_by_registro(ids, medio_map: dict) -> dict[str, list[str]]:
    """{registro: [article_id, ...]} usando medio_map = {article_id: medio}."""
    out: dict[str, list[str]] = {"formal": [], "informal": []}
    for art_id in ids:
        out[registro_of(str(medio_map.get(art_id, "")))].append(art_id)
    return out
