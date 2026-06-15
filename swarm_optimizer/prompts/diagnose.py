"""Agente DIAGNOSE: lee FP/FN (con trayectoria) y emite causas de error."""
from __future__ import annotations

from swarm_optimizer.prompts._authored import STRUCT_DIAGNOSE, SYSTEM_DIAGNOSE
from swarm_optimizer.prompts.base import PromptSpec


def build_user_diagnose(fps: list[str], fns: list[str],
                        meta_review_text: str = "") -> str:
    """Contexto de la iteración: FP/FN (idealmente con trayectoria: artículo +
    evidence_quote + pred + gold) y, si existe, los patrones sistémicos vigentes."""
    review = (
        "\nPatrones sistémicos del meta-revisor (úsalos como contexto):\n"
        f"{meta_review_text}\n" if meta_review_text else ""
    )
    fp_txt = "\n".join(fps[:20]) or "(ninguno)"
    fn_txt = "\n".join(fns[:20]) or "(ninguno)"
    return (
        "Falsos positivos (relaciones emitidas que NO están en el gold) — con trayectoria:\n"
        f"{fp_txt}\n\n"
        "Falsos negativos (relaciones del gold que NO fueron emitidas) — con trayectoria:\n"
        f"{fn_txt}\n"
        f"{review}"
    )


DIAGNOSE_SPEC = PromptSpec(
    agent="diagnose",
    system=SYSTEM_DIAGNOSE,
    build_user=build_user_diagnose,
    required_structure=STRUCT_DIAGNOSE,
    required_context=("FP_TRAJ_SENTINEL", "FN_TRAJ_SENTINEL", "METAREV_CTX_SENTINEL"),
    probe={
        "fps": ["FP_TRAJ_SENTINEL"],
        "fns": ["FN_TRAJ_SENTINEL"],
        "meta_review_text": "METAREV_CTX_SENTINEL",
    },
)
