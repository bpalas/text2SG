"""Loop evolutivo: archivo open-ended + ELO (skirmish) + championship anclado."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from swarm_optimizer.archive import Archive
from swarm_optimizer.elo import update_pairwise
from swarm_optimizer.extractor import run_extraction
from swarm_optimizer.fitness import fitness
from swarm_optimizer.genome import Genome
from swarm_optimizer.meta_policy import MetaPolicy
from swarm_optimizer.mutate import (cross_pollinate, diagnose, fresh_genome,
                                    meta_review, propose)
from swarm_optimizer.rubric import compute_metrics, gold_relations_path, load_union_map
from swarm_optimizer.splits import gold_articles_path, load_splits, subsample


def skirmish_result(score_a: float, score_b: float) -> float:
    if score_a > score_b:
        return 1.0
    if score_a < score_b:
        return 0.0
    return 0.5


def should_stop(archive: Archive, iteration: int, max_iter: int, budget_tokens: int) -> bool:
    if iteration >= max_iter:
        return True
    if archive.total_tokens() >= budget_tokens:
        return True
    return False


def _format_errors(predictions, gold_df, union_map):
    """FP/FN con TRAYECTORIA (evidence_quote) para el diagnosticador (reflexión GEPA).
    La cita literal es la señal de trayectoria de mayor valor: dice qué afirma el
    modelo (FP) o qué texto sustenta el gold omitido (FN)."""
    from swarm_optimizer.rubric import match_entity
    fps, fns = [], []
    pred_map = {p["article_id"]: p for p in predictions}
    for art_id, pred in pred_map.items():
        union = union_map.get(art_id, {})
        gold_rels = gold_df[gold_df["article_id"] == art_id]
        has_gold = "u_from" in gold_df.columns
        gold_pairs = {(r.u_from, r.u_to) for r in gold_rels.itertuples()} if has_gold else set()
        gold_quote = {
            (r.u_from, r.u_to): (getattr(r, "evidence_quote", "") or "")
            for r in gold_rels.itertuples()
        } if has_gold else {}
        matched = set()
        for rel in pred.get("relations", []):
            fu, _ = match_entity(rel["from_entity"], union)
            tu, _ = match_entity(rel["to_entity"], union)
            pair = (fu, tu) if fu and tu else None
            if pair and pair in gold_pairs:
                matched.add(pair)
            elif fu and tu:
                q = (rel.get("evidence_quote") or "").strip().replace("\n", " ")[:120]
                fps.append(f"{rel['from_entity']} -{rel['act_type']}-> {rel['to_entity']} "
                           f"| cita: «{q}» ({art_id[:8]})")
        for pair in gold_pairs - matched:
            q = (gold_quote.get(pair, "") or "").strip().replace("\n", " ")[:120]
            fns.append(f"{pair[0]} -> {pair[1]} | cita gold: «{q}» ({art_id[:8]})")
    return fps, fns


def _evaluate(genome, ids, articles_df, gold_df, union_map, extract_fn, metrics_fn):
    preds, tokens = extract_fn(ids, articles_df, gold_df, union_map, genome)
    metrics = metrics_fn(preds, ids, gold_df, union_map)
    tokens_per_article = tokens / max(len(ids), 1)
    return preds, metrics, fitness(metrics, tokens_per_article, model=genome.model), tokens


def _sanity_check(child: "Genome", parent: "Genome") -> str | None:
    """Gate sintáctico gratis (#0): rechaza mutantes con invariantes rotos.
    Solo activo en prompts reales (>200 chars). Devuelve None si OK."""
    # Prompts cortos son de tests — no aplicar gate
    if len(parent.prompt_text.strip()) < 200:
        return None
    # El prompt debe contener el schema JSON para que Gemini sepa el formato
    if '{"entities"' not in child.prompt_text and '"relations"' not in child.prompt_text:
        return "schema JSON ausente en el prompt"
    # La arquitectura no debe cambiar (given_entities/one_pass/debate son invariantes)
    if child.architecture != parent.architecture:
        return f"arquitectura cambiada: {parent.architecture} -> {child.architecture}"
    # El prompt no puede quedar demasiado corto respecto al padre
    if len(child.prompt_text.strip()) < len(parent.prompt_text.strip()) * 0.3:
        return f"prompt demasiado recortado ({len(child.prompt_text)} vs {len(parent.prompt_text)} chars)"
    # El cap nunca debe subir más de 2x el del padre (explosión de relaciones)
    parent_cap = parent.validation.max_relations_per_article or 999
    child_cap = child.validation.max_relations_per_article or 999
    if child_cap > parent_cap * 2:
        return f"cap explosivo: {parent_cap} -> {child_cap}"
    return None


def run_loop(
    max_iter: int = 80,
    budget_usd: float = 8.0,
    subsample_k: int = 12,
    championship_every: int = 5,
    cross_every: int = 7,
    gate_k: int = 4,
    gate_epsilon: float = 0.05,
    use_meta_policy: bool = False,
    meta_policy_path: Path | None = None,
    evo_model: str | None = None,
    archive_path: Path | None = None,
    best_path: Path | None = None,
    seed_genome: Genome | None = None,
    seed_genomes: list[Genome] | None = None,
    articles_df: pd.DataFrame | None = None,
    gold_df: pd.DataFrame | None = None,
    splits: dict | None = None,
    client=None,
    extract_fn=run_extraction,
    metrics_fn=compute_metrics,
    verbose: bool = True,
) -> Genome:
    """Loop evolutivo con las extensiones del informe 2026-06-09:
    - Cascada de evaluación (AlphaEvolve): gate K=gate_k con margen ε antes del skirmish.
    - Multi-seed (tarea 1.4): varias semillas compiten desde el inicio.
    - Meta-policy (bandit Thompson, §2.2): reemplaza el calendario fijo si use_meta_policy.
    - Memoria retrospectiva v1 + Meta-review agent inyectados en los prompts de mutación.
    """
    archive_path = archive_path or (Path(__file__).parent.parent / "results/swarm/history.jsonl")
    best_path = best_path or (Path(__file__).parent.parent / "results/swarm/best_config.json")

    if articles_df is None:
        articles_df = pd.read_parquet(gold_articles_path())
    if gold_df is None:
        gold_df = pd.read_parquet(gold_relations_path())
    if splits is None:
        splits = load_splits()
    if client is None:
        # GeminiClient (no genai.Client crudo): habilita rol `system` para los
        # meta-agentes. El extractor crea su propio genai.Client vía _client().
        from swarm_optimizer.llm_backends import GeminiClient
        client = GeminiClient(api_key=os.environ.get("GEMINI_API_KEY", ""))

    eval_ids, test_ids = splits["eval"], splits["test"]
    # Carga los unions reales cuando se usa la rúbrica real (los necesita para matchear
    # entidades). Con un metrics_fn falso (tests) se omite el I/O.
    union_map = load_union_map(eval_ids + test_ids) if metrics_fn is compute_metrics else {}
    budget_tokens = int(budget_usd / 0.15 * 1_000_000)
    rng = np.random.default_rng(42)

    archive = Archive(archive_path)
    seeds = seed_genomes or [seed_genome or Genome.from_seed()]
    evo = evo_model or seeds[0].model  # modelo del "Evolution AI" (diagnose/propose/cross)
    policy = MetaPolicy(meta_policy_path) if use_meta_policy else None

    # sembrar todas las semillas + championship inicial para anclar al campeón
    champion_id, best_s0 = None, -float("inf")
    for seed in seeds:
        sid = archive.add(seed, mutation_type="seed")
        _, m0, s0, t0 = _evaluate(seed, eval_ids, articles_df, gold_df,
                                  union_map, extract_fn, metrics_fn)
        archive.record_championship(sid, s0, {**m0, "tokens": t0})
        if s0 > best_s0:
            champion_id, best_s0 = sid, s0
        if verbose and m0.get("Recall_rel", 1) == 0 and m0.get("Precision_rel", 1) == 0:
            print(f"  [WARN] seed={sid} extrajo 0 relaciones en eval completo.",
                  file=sys.stderr)
            print(f"         Ejecuta '--probe' para diagnosticar antes de continuar.",
                  file=sys.stderr)

    current_meta_review = ""          # síntesis transversal vigente (AI Co-Scientist)
    window_arms: list[tuple[str, str]] = []   # acciones desde el último championship
    prev_champ_score = best_s0
    consecutive_noops = 0
    _NOOP_WARN = 3                    # avisa cada vez que se acumulan N noops seguidos

    # Captura Ctrl+C al final de la iteración actual (no pierde el archivo)
    import signal as _signal
    _stop = [False]

    def _on_sigint(sig, frame):
        _stop[0] = True
        if verbose:
            print("\n[SIGINT] Ctrl+C — terminando iteración actual y guardando...",
                  file=sys.stderr)

    _old_sigint = _signal.getsignal(_signal.SIGINT)
    _signal.signal(_signal.SIGINT, _on_sigint)

    for iteration in range(max_iter):
        if _stop[0] or should_stop(archive, iteration, max_iter, budget_tokens):
            break

        # 1. meta-agente (o calendario fijo) elige operador + sesgo de padre
        arm = None
        if policy is not None:
            ops = ["diff_a", "diff_b"]
            if len(archive.all()) >= 2:
                ops += ["cross", "fresh"]
            arm = policy.choose(rng, available_ops=ops)
            operator, bias = arm
        else:
            do_cross = (iteration > 0 and iteration % cross_every == 0
                        and len(archive.all()) >= 2)
            operator, bias = ("cross" if do_cross else "diff"), "explore"

        if bias == "exploit":
            parent_id = archive.top_by_elo(1)[0].id
        else:
            parent_id = archive.select_parent(rng)
        parent = archive.get(parent_id).genome

        # 2. mutar según el operador elegido
        if operator == "cross":
            # cross ya NO es ciego: recibe las métricas de cada padre + los patrones
            # sistémicos vigentes para fusionar con justificación métrica.
            top2 = archive.top_by_elo(2)
            child, mtype, touched = cross_pollinate(
                top2[0].genome, top2[1].genome, client, model=evo,
                parent1_metrics=top2[0].metrics, parent2_metrics=top2[1].metrics,
                diagnosis=current_meta_review)
            diagnosis = None
        elif operator == "fresh":
            # fresh ya NO es ciego: diseña contra los patrones sistémicos del meta-revisor.
            top2 = archive.top_by_elo(2)
            child, mtype, touched = fresh_genome(
                top2[0].genome, top2[1].genome, client, model=evo,
                meta_review_text=current_meta_review)
            diagnosis = None
        else:
            sub = subsample(eval_ids, subsample_k, seed=1000 + iteration)
            preds, _, _, _ = _evaluate(parent, sub, articles_df, gold_df,
                                       union_map, extract_fn, metrics_fn)
            fps, fns = _format_errors(preds, gold_df, union_map)
            diagnosis = diagnose(fps, fns, client, model=evo,
                                 meta_review_text=current_meta_review)
            memory = archive.memory_snippets(parent_id)   # memoria retrospectiva v1
            force = {"diff_a": "A", "diff_b": "B"}.get(operator)
            child, mtype, touched = propose(parent, diagnosis, client, model=evo,
                                            memory=memory, force_artifact=force)

        if mtype == "noop":
            consecutive_noops += 1
            if verbose and consecutive_noops % _NOOP_WARN == 0:
                print(f"  [WARN] {consecutive_noops} noops consecutivos — "
                      f"el LLM no produce diffs válidos. "
                      f"¿El prompt de propose está bien formado?", file=sys.stderr)
            continue
        consecutive_noops = 0

        # gate sintáctico gratis (#0): rechaza invariantes rotos antes de gastar tokens
        sanity_fail = _sanity_check(child, parent)
        if sanity_fail:
            if verbose:
                print(f"[iter {iteration}] {mtype} SANITY-FAIL: {sanity_fail}", file=sys.stderr)
            continue
        child_id = archive.add(child, parent_id=parent_id, mutation_type=mtype,
                               artifact_touched=touched, diagnosis=diagnosis)
        actual_arm = (mtype, bias)    # crédito a lo realmente ejecutado

        # 3a. gate 1 de la cascada (AlphaEvolve): K barato, descarta mutantes claramente malos
        if gate_k and gate_k < subsample_k:
            sub_g = subsample(eval_ids, gate_k, seed=3000 + iteration)
            _, _, g_child, _ = _evaluate(child, sub_g, articles_df, gold_df,
                                         union_map, extract_fn, metrics_fn)
            _, _, g_champ, _ = _evaluate(archive.get(champion_id).genome, sub_g,
                                         articles_df, gold_df, union_map,
                                         extract_fn, metrics_fn)
            if g_child < g_champ - gate_epsilon:
                # descartado en gate 1: pierde ELO, registra delta y recompensa; no skirmish
                new_child_elo, new_champ_elo = update_pairwise(
                    archive.get(child_id).elo, archive.get(champion_id).elo, 0.0)
                archive.record_elo(child_id, new_child_elo)
                archive.record_elo(champion_id, new_champ_elo)
                archive.record_delta(child_id, g_child - g_champ)
                if policy is not None:
                    policy.update_from_delta(actual_arm, g_child - g_champ)
                    window_arms.append(actual_arm)
                if verbose:
                    print(f"[iter {iteration}] {mtype} child={child_id} GATE1-DESCARTADO "
                          f"g_child={g_child:.3f} g_champ={g_champ:.3f}")
                continue

        # 3b. skirmish completo sobre submuestreo fresco (child vs campeón actual)
        sub = subsample(eval_ids, subsample_k, seed=2000 + iteration)
        _, _, s_child, _ = _evaluate(child, sub, articles_df, gold_df,
                                     union_map, extract_fn, metrics_fn)
        _, _, s_champ, _ = _evaluate(archive.get(champion_id).genome, sub,
                                     articles_df, gold_df, union_map, extract_fn, metrics_fn)
        res = skirmish_result(s_child, s_champ)
        new_child_elo, new_champ_elo = update_pairwise(
            archive.get(child_id).elo, archive.get(champion_id).elo, res)
        archive.record_elo(child_id, new_child_elo)
        archive.record_elo(champion_id, new_champ_elo)
        archive.record_delta(child_id, s_child - s_champ)   # memoria v2: ¿el cambio ayudó?
        if policy is not None:
            policy.update_from_delta(actual_arm, s_child - s_champ)
            window_arms.append(actual_arm)

        if verbose:
            print(f"[iter {iteration}] {mtype} child={child_id} "
                  f"s_child={s_child:.3f} s_champ={s_champ:.3f} res={res}")

        # 4. championship cada M: re-evalúa top-T sobre eval fijo, corona
        if iteration > 0 and iteration % championship_every == 0:
            agg_fps, agg_fns = [], []
            for entry in archive.top_by_elo(3):
                preds_c, m, s, t = _evaluate(entry.genome, eval_ids, articles_df, gold_df,
                                             union_map, extract_fn, metrics_fn)
                # Goodhart: chequeo contra test (persistido para report_run.py)
                _, mt, st, _ = _evaluate(entry.genome, test_ids, articles_df, gold_df,
                                         union_map, extract_fn, metrics_fn)
                archive.record_championship(
                    entry.id, s, {**m, "tokens": t, "test_score": st, "eval_score": s})
                if s - st > 0.10 and verbose:
                    print(f"  [GOODHART] entry={entry.id} eval={s:.3f} test={st:.3f}",
                          file=sys.stderr)
                fps_c, fns_c = _format_errors(preds_c, gold_df, union_map)
                agg_fps += fps_c
                agg_fns += fns_c
            champ = archive.champion()
            if champ:
                champion_id = champ.id
                # crédito retrospectivo del championship al meta-agente (§2.2)
                if policy is not None and champ.championship_score is not None:
                    policy.credit_championship(
                        window_arms, champ.championship_score - prev_champ_score)
                    prev_champ_score = champ.championship_score
                window_arms = []
            # Meta-review agent (AI Co-Scientist): síntesis de patrones transversales
            if agg_fps or agg_fns:
                review = meta_review(agg_fps, agg_fns, client, model=evo)
                if review:
                    current_meta_review = review
                    archive.record_meta_review(champion_id, review)

    _signal.signal(_signal.SIGINT, _old_sigint)  # restaurar handler original

    # championship final sobre el top para asegurar un campeón con championship_score
    for entry in archive.top_by_elo(3):
        if entry.championship_score is None:
            _, m, s, t = _evaluate(entry.genome, eval_ids, articles_df, gold_df,
                                   union_map, extract_fn, metrics_fn)
            archive.record_championship(entry.id, s, {**m, "tokens": t})

    best = archive.champion()
    best_genome = best.genome if best else archive.get(champion_id).genome
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(best_genome.to_json(), encoding="utf-8")
    if verbose:
        print(f"Campeón: score={best.championship_score if best else 'n/a'} -> {best_path}")
    return best_genome
