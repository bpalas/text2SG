"""Trazas de error para el operador reflexivo: qué relaciones reales se pierden (FN, el hueco de
recall) y qué se inventa (FP, disciplina), por tipo de texto. Es un feed CUALITATIVO para la
reflexión, no la métrica oficial:
- FP SUBCUENTA respecto de compute_metrics: una segunda predicción correcta del mismo par cuenta
  0 FP acá (compute_metrics la cuenta como 1 FP). Es decir, fp_total NO es cota superior.
- FN dedupea por par (igual que compute_metrics): dos relaciones gold sobre el mismo par colapsan
  a 1 FN, así que fn_by_dureza subcuenta el hueco real cuando el gold tiene multi-aristas por par.
"""
from __future__ import annotations

from swarm_optimizer.rubric import match_entity, canonicalize_union
from swarm_optimizer.subsets import registro_of


def _pred_pairs(relations, union):
    pairs = set()
    for rel in relations:
        f, fnil = match_entity(str(rel.get("from_entity") or ""), union)
        t, tnil = match_entity(str(rel.get("to_entity") or ""), union)
        if f and t and not fnil and not tnil:
            pairs.add((f, t))
    return pairs


def diagnostics(ids, preds, gold_df, union_map, articles_df, max_examples: int = 8) -> dict:
    pm = {p["article_id"]: p for p in preds}
    meta = {r.article_id: r for r in articles_df.itertuples()}
    fn, fp = [], []
    fn_by_dureza: dict[str, int] = {}
    fn_by_registro: dict[str, int] = {"formal": 0, "informal": 0}
    fp_by_registro: dict[str, int] = {"formal": 0, "informal": 0}
    fp_total = fp_distractor = 0
    for art_id in ids:
        row = meta.get(art_id)
        dureza = str(getattr(row, "dureza", "") or "") if row is not None else ""
        is_dis = bool(getattr(row, "es_distractor", False)) if row is not None else False
        registro = registro_of(str(getattr(row, "medio", "") or "") if row is not None else "")
        union, alias = canonicalize_union(union_map.get(art_id, {}))
        grows = gold_df[gold_df["article_id"] == art_id]
        gold_pairs = {(alias.get(r.u_from, r.u_from), alias.get(r.u_to, r.u_to)): r
                      for r in grows.itertuples()}
        rels = pm.get(art_id, {}).get("relations", [])
        ppairs = _pred_pairs(rels, union)
        for pair, r in gold_pairs.items():
            if pair not in ppairs:
                fn_by_dureza[dureza] = fn_by_dureza.get(dureza, 0) + 1
                fn_by_registro[registro] += 1
                if len(fn) < max_examples:
                    fn.append({"article_id": art_id, "dureza": dureza, "registro": registro,
                               "act_type": r.act_type,
                               "quote": (r.evidence_quote or "")[:160]})
        for rel in rels:
            f, fnil = match_entity(str(rel.get("from_entity") or ""), union)
            t, tnil = match_entity(str(rel.get("to_entity") or ""), union)
            if not (f and t and not fnil and not tnil and (f, t) in gold_pairs):
                fp_total += 1
                fp_by_registro[registro] += 1
                if is_dis:
                    fp_distractor += 1
                if len(fp) < max_examples:
                    fp.append({"article_id": art_id, "es_distractor": is_dis, "registro": registro,
                               "from_entity": rel.get("from_entity"),
                               "to_entity": rel.get("to_entity"),
                               "act_type": rel.get("act_type"),
                               "quote": (rel.get("evidence_quote") or "")[:120]})
    return {"fn": fn, "fp": fp, "fn_by_dureza": fn_by_dureza,
            "fn_by_registro": fn_by_registro, "fp_by_registro": fp_by_registro,
            "fp_total": fp_total, "fp_distractor": fp_distractor}
