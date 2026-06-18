"""Agente META_REVIEW: detecta patrones sistémicos sobre errores agregados del top."""
from __future__ import annotations

from text2sg.prompts._authored import STRUCT_META_REVIEW, SYSTEM_META_REVIEW
from text2sg.prompts.base import PromptSpec


def build_user_meta_review(fps: list[str], fns: list[str]) -> str:
    """Errores AGREGADOS del top-3 de la población."""
    fp_txt = "\n".join(fps[:40]) or "(ninguno)"
    fn_txt = "\n".join(fns[:40]) or "(ninguno)"
    return (
        "Falsos positivos agregados (del top de la población):\n"
        f"{fp_txt}\n\n"
        "Falsos negativos agregados:\n"
        f"{fn_txt}\n"
    )


META_REVIEW_SPEC = PromptSpec(
    agent="meta_review",
    system=SYSTEM_META_REVIEW,
    build_user=build_user_meta_review,
    required_structure=STRUCT_META_REVIEW,
    required_context=("AGG_FP_SENTINEL", "AGG_FN_SENTINEL"),
    probe={"fps": ["AGG_FP_SENTINEL"], "fns": ["AGG_FN_SENTINEL"]},
)
