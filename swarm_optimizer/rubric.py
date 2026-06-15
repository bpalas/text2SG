"""
Entity matching and metrics for the gold standard rubric.

Contains:
- normalize: lowercase + strip diacritics + strip spaces
- load_union: load YAML entity union for an article
- match_entity: resolve entity name to union_id via exact + fuzzy matching
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from swarm_optimizer.gold_version import gold_paths


def unions_dir() -> Path:
    return gold_paths()["unions_dir"]


def gold_relations_path() -> Path:
    return gold_paths()["relations"]


# ── helpers ──────────────────────────────────────────────────────────────── #


def normalize(text: str) -> str:
    """
    Lowercase + strip diacritics + strip extra spaces.

    Example:
        normalize("  García  ") -> "garcia"
    """
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def load_union(article_id: str) -> dict:
    """
    Carga el YAML de unión de entidades para un artículo.

    Args:
        article_id: article ID (first 16 chars used for filename)

    Returns:
        {union_id: {type, canonical_names, surfaces}} or {} if not found

    Example:
        union = load_union("05d9bed3286d25a957fcda...")
        # -> {"U1": {"type": "roster_actor", ...}, ...}
    """
    yaml_path = unions_dir() / f"{article_id[:16]}.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        ent["union_id"]: ent
        for ent in data.get("entities_union", [])
    }


def match_entity(name: str, union: dict) -> tuple[str | None, bool]:
    """
    Intenta resolver un nombre de entidad (texto libre) a un union_id.

    Estrategia:
    1. Exact match (normalized): busca name en todas las surfaces + canonical_names
    2. Fuzzy match (token_sort_ratio >= 85): si no hay exact match

    Args:
        name: entity name (text from extraction)
        union: {union_id: {type, canonical_names, surfaces}} from load_union()

    Returns:
        (union_id, is_nil)
        - union_id: matched union ID, or None if no match
        - is_nil: True if type == "NIL", False otherwise

    Example:
        uid, is_nil = match_entity("Gabriel Boric", union)
        # -> ("U1", False)
    """
    norm_name = normalize(name)

    # Paso 1: exact match
    for uid, ent in union.items():
        all_names = ent.get("surfaces", []) + ent.get("canonical_names", [])
        if norm_name in [normalize(n) for n in all_names]:
            return uid, ent.get("type") == "NIL"

    # Paso 2: fuzzy match (token_sort_ratio >= 85)
    best_score, best_uid, best_is_nil = 0, None, False
    for uid, ent in union.items():
        all_names = ent.get("surfaces", []) + ent.get("canonical_names", [])
        for n in all_names:
            score = fuzz.token_sort_ratio(norm_name, normalize(n))
            if score > best_score:
                best_score, best_uid, best_is_nil = score, uid, ent.get("type") == "NIL"

    if best_score >= 85:
        return best_uid, best_is_nil
    return None, False


def canonicalize_union(union: dict) -> tuple[dict, dict]:
    """Fusiona entradas duplicadas del union (mismo actor con dos union_ids).

    El censo 2026-06-10 (scripts/audit_unions.py) encontró 11% de uids duplicados
    (ej: 'Presidenta Bachelet' U6 vs 'Michelle Bachelet Jeria' U1) y relaciones gold
    ancladas al duplicado, lo que convierte extracciones correctas en FP+FN.

    Criterio de duplicado: el canonical normalizado de una entrada aparece entre los
    canonicals+surfaces de otra, o fuzzy(canonical, canonical) >= 90. Sobrevive la
    entrada con más surfaces (empate: el uid de número menor); la fusionada aporta
    sus canonicals/surfaces al sobreviviente.

    Returns:
        (clean_union, alias) — alias mapea uid_duplicado -> uid_sobreviviente.
    """
    items = list(union.items())
    alias: dict[str, str] = {}

    def _uid_num(uid: str) -> int:
        digits = "".join(c for c in uid if c.isdigit())
        return int(digits) if digits else 0

    for uid, ent in items:
        cn = normalize(ent.get("canonical_names", [""])[0] or "")
        if not cn:
            continue
        n_surf = len(ent.get("surfaces", []) or [])
        for uid2, ent2 in items:
            if uid2 == uid:
                continue
            names2 = [normalize(n) for n in (
                ent2.get("canonical_names", []) + (ent2.get("surfaces", []) or []))]
            c2 = normalize(ent2.get("canonical_names", [""])[0] or "")
            if cn in names2 or (c2 and fuzz.token_sort_ratio(cn, c2) >= 90):
                n_surf2 = len(ent2.get("surfaces", []) or [])
                # pierde el que tiene menos surfaces; empate: uid de número mayor
                if (n_surf, -_uid_num(uid)) < (n_surf2, -_uid_num(uid2)):
                    alias[uid] = uid2
                    break

    # resolver cadenas A->B->C
    def _resolve(uid: str) -> str:
        seen = set()
        while uid in alias and uid not in seen:
            seen.add(uid)
            uid = alias[uid]
        return uid

    alias = {uid: _resolve(uid) for uid in alias}

    clean: dict[str, dict] = {}
    for uid, ent in items:
        if uid in alias:
            continue
        clean[uid] = dict(ent)
    for uid, target in alias.items():
        if target not in clean:
            continue
        src = union[uid]
        tgt = clean[target]
        tgt["canonical_names"] = list(dict.fromkeys(
            tgt.get("canonical_names", []) + src.get("canonical_names", [])))
        tgt["surfaces"] = list(dict.fromkeys(
            (tgt.get("surfaces", []) or []) + (src.get("surfaces", []) or [])))

    return clean, alias


# ── metrics ──────────────────────────────────────────────────────────────── #

import pandas as pd


def compute_metrics(
    predictions: list[dict],
    article_ids: list[str],
    gold_df: pd.DataFrame,
    union_map: dict[str, dict],
) -> dict:
    """
    predictions: lista de {article_id, entities: [...], relations: [...]}
    article_ids: lista de IDs a evaluar (eval o test set)
    gold_df: gold parquet filtrado a esos artículos
    union_map: {article_id: union_dict} — de load_union() por cada artículo

    Retorna dict con F1_rel, Precision_rel, Recall_rel, Polarity_acc,
    Act_acc, F1_ent, Precision_ent, Recall_ent.
    """
    gold_filtered = gold_df[gold_df["article_id"].isin(article_ids)]
    pred_map = {p["article_id"]: p for p in predictions}

    tp_rel = fp_rel = fn_rel = 0
    tp_rel_u = fp_rel_u = fn_rel_u = 0   # undirected: par no ordenado {a,b}
    tp_ent = fp_ent = fn_ent = 0
    polarity_correct = polarity_total = 0
    act_correct = act_total = 0

    for art_id in article_ids:
        union, alias = canonicalize_union(union_map.get(art_id, {}))
        pred = pred_map.get(art_id, {"entities": [], "relations": []})
        gold_rels = gold_filtered[gold_filtered["article_id"] == art_id]

        # ── Entity metrics ────────────────────────────────────────────── #
        gold_uid_set = {
            uid for uid, ent in union.items()
            if ent.get("type") != "NIL"
        }
        matched_uids: set[str] = set()
        for ent in pred.get("entities", []):
            uid, is_nil = match_entity(ent["name"], union)
            if is_nil:
                pass  # NIL match → ignored in entity F1 (counts as FP in relations only)
            elif uid is None:
                fp_ent += 1
            else:
                matched_uids.add(uid)

        tp_ent += len(matched_uids & gold_uid_set)
        fp_ent += len(matched_uids - gold_uid_set)
        fn_ent += len(gold_uid_set - matched_uids)

        # ── Relation metrics ──────────────────────────────────────────── #
        # Los uids del gold pueden apuntar a un duplicado fusionado: remapear.
        gold_pairs = {
            (alias.get(row.u_from, row.u_from), alias.get(row.u_to, row.u_to)): row
            for row in gold_rels.itertuples()
        }

        # versión no ordenada del gold para la métrica undirected (el techo si la
        # dirección fuera perfecta): una predicción (a,b) machea gold (b,a).
        gold_pairs_u = {tuple(sorted(p)) for p in gold_pairs}

        matched_gold_pairs: set[tuple] = set()
        matched_gold_u: set[tuple] = set()
        for rel in pred.get("relations", []):
            # relación malformada (sin endpoints) cuenta como FP, no revienta
            from_uid, from_nil = match_entity(str(rel.get("from_entity") or ""), union)
            to_uid, to_nil = match_entity(str(rel.get("to_entity") or ""), union)

            # Relación con endpoint NIL o no-match → FP (directed y undirected)
            if from_uid is None or to_uid is None or from_nil or to_nil:
                fp_rel += 1
                fp_rel_u += 1
                continue

            pair = (from_uid, to_uid)
            if pair in gold_pairs and pair not in matched_gold_pairs:
                tp_rel += 1
                matched_gold_pairs.add(pair)
                gold_row = gold_pairs[pair]
                if rel.get("polarity") == gold_row.polarity:
                    polarity_correct += 1
                polarity_total += 1
                if rel.get("act_type") == gold_row.act_type:
                    act_correct += 1
                act_total += 1
            else:
                fp_rel += 1

            pair_u = tuple(sorted(pair))
            if pair_u in gold_pairs_u and pair_u not in matched_gold_u:
                tp_rel_u += 1
                matched_gold_u.add(pair_u)
            else:
                fp_rel_u += 1

        fn_rel += len(gold_pairs) - len(matched_gold_pairs)
        fn_rel_u += len(gold_pairs_u) - len(matched_gold_u)

    def f1(tp: int, fp: int, fn: int) -> float:
        if tp + fp == 0 and tp + fn == 0:
            return 1.0
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    return {
        "F1_rel": f1(tp_rel, fp_rel, fn_rel),
        "Precision_rel": tp_rel / (tp_rel + fp_rel) if tp_rel + fp_rel > 0 else 0.0,
        "Recall_rel": tp_rel / (tp_rel + fn_rel) if tp_rel + fn_rel > 0 else 0.0,
        # undirected (par no ordenado): el techo alcanzable arreglando solo la dirección.
        # El gap directed→undirected = cuánto cuesta hoy la dirección invertida.
        "F1_rel_undirected": f1(tp_rel_u, fp_rel_u, fn_rel_u),
        "Precision_rel_undirected": tp_rel_u / (tp_rel_u + fp_rel_u) if tp_rel_u + fp_rel_u > 0 else 0.0,
        "Recall_rel_undirected": tp_rel_u / (tp_rel_u + fn_rel_u) if tp_rel_u + fn_rel_u > 0 else 0.0,
        "Polarity_acc": polarity_correct / polarity_total if polarity_total > 0 else 0.0,
        "Act_acc": act_correct / act_total if act_total > 0 else 0.0,
        "F1_ent": f1(tp_ent, fp_ent, fn_ent),
        "Precision_ent": tp_ent / (tp_ent + fp_ent) if tp_ent + fp_ent > 0 else 0.0,
        "Recall_ent": tp_ent / (tp_ent + fn_ent) if tp_ent + fn_ent > 0 else 0.0,
    }


def load_union_map(article_ids: list[str]) -> dict[str, dict]:
    """Carga todos los union YAMLs para una lista de article_ids."""
    return {art_id: load_union(art_id) for art_id in article_ids}
