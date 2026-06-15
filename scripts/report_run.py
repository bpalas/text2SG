"""Reporte MD de una corrida del swarm optimizer (tarea 1.3 del informe 2026-06-09).

Lee history.jsonl y emite un resumen Markdown con:
- campeón actual y sus métricas
- histograma de recall_rel de la población (vigilancia del piso de recall)
- tasa de éxito por mutation_type (% fitness_delta > 0)
- gap Goodhart (eval_score − test_score) por entrada con championship
- tasa de descarte en gate 1 (cascada)

Uso:
    python scripts/report_run.py [ruta/history.jsonl] [-o reporte.md]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_HISTORY = Path(__file__).parent.parent / "results/swarm/history.jsonl"
RECALL_BINS = [(0.0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30),
               (0.30, 0.50), (0.50, 1.01)]


def load_entries(path: Path) -> dict[int, dict]:
    """Última línea por id gana (mismo contrato que Archive._reload)."""
    entries: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            entries[rec["id"]] = rec
    return entries


def recall_histogram(entries: dict[int, dict]) -> list[str]:
    counts = defaultdict(int)
    for e in entries.values():
        rr = (e.get("metrics") or {}).get("Recall_rel")
        if rr is None:
            continue
        for lo, hi in RECALL_BINS:
            if lo <= rr < hi:
                counts[(lo, hi)] += 1
                break
    lines = []
    for lo, hi in RECALL_BINS:
        n = counts[(lo, hi)]
        bar = "█" * n
        lines.append(f"| [{lo:.2f}, {hi:.2f}) | {n} | {bar} |")
    return lines


def operator_stats(entries: dict[int, dict]) -> list[str]:
    by_op: dict[str, list[float]] = defaultdict(list)
    for e in entries.values():
        if e.get("mutation_type") in (None, "seed"):
            continue
        d = e.get("fitness_delta")
        if d is not None:
            by_op[e["mutation_type"]].append(d)
    lines = []
    for op, deltas in sorted(by_op.items()):
        n = len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        mean = sum(deltas) / n
        lines.append(f"| {op} | {n} | {wins} ({100*wins/n:.0f}%) | {mean:+.4f} |")
    return lines


def goodhart_rows(entries: dict[int, dict]) -> list[str]:
    rows = []
    for e in sorted(entries.values(), key=lambda x: x["id"]):
        m = e.get("metrics") or {}
        ev, ts = m.get("eval_score"), m.get("test_score")
        if ev is None or ts is None:
            continue
        gap = ev - ts
        flag = " ⚠️" if gap > 0.10 else ""
        rows.append(f"| {e['id']} | {ev:.3f} | {ts:.3f} | {gap:+.3f}{flag} |")
    return rows


def build_report(path: Path) -> str:
    entries = load_entries(path)
    n = len(entries)
    muts = [e for e in entries.values() if e.get("mutation_type") not in (None, "seed")]
    champ = max(
        (e for e in entries.values() if e.get("championship_score") is not None),
        key=lambda e: e["championship_score"], default=None,
    )

    lines = [f"# Reporte de corrida — `{path.name}`", ""]
    lines += [f"- **Entradas en el archivo:** {n} ({len(muts)} mutaciones no-seed)"]
    total_tokens = sum(int((e.get("metrics") or {}).get("tokens", 0)) for e in entries.values())
    lines += [f"- **Tokens totales registrados:** {total_tokens:,}", ""]

    if champ:
        m = champ.get("metrics") or {}
        lines += ["## Campeón", ""]
        lines += [f"- **id:** {champ['id']} | **mutation_type:** {champ.get('mutation_type')} "
                  f"| **ELO:** {champ.get('elo', 0):.0f}"]
        lines += [f"- **championship_score:** {champ['championship_score']:.4f}"]
        lines += [f"- **Precision_rel:** {m.get('Precision_rel', float('nan')):.3f} | "
                  f"**Recall_rel:** {m.get('Recall_rel', float('nan')):.3f} | "
                  f"**Precision_ent:** {m.get('Precision_ent', float('nan')):.3f} | "
                  f"**Polarity_acc:** {m.get('Polarity_acc', float('nan')):.3f}", ""]

    lines += ["## Histograma de Recall_rel (vigilancia del piso 0.15)", ""]
    lines += ["| Rango | n | |", "|---|---|---|"] + recall_histogram(entries) + [""]

    lines += ["## Tasa de éxito por operador (fitness_delta > 0)", ""]
    op_rows = operator_stats(entries)
    if op_rows:
        lines += ["| Operador | n | Éxitos | Delta medio |", "|---|---|---|---|"] + op_rows
    else:
        lines += ["(sin mutaciones con fitness_delta registrado)"]
    lines += [""]

    lines += ["## Gap Goodhart (eval − test) por championship", ""]
    g_rows = goodhart_rows(entries)
    if g_rows:
        lines += ["| id | eval | test | gap |", "|---|---|---|---|"] + g_rows
    else:
        lines += ["(sin test_score persistido — corre el loop actualizado)"]
    lines += [""]

    reviews = [e for e in entries.values() if e.get("meta_review")]
    if reviews:
        lines += ["## Último meta-review", "", "```",
                  reviews[-1]["meta_review"].strip(), "```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reporte MD de una corrida del swarm")
    ap.add_argument("history", nargs="?", default=str(DEFAULT_HISTORY),
                    help="Ruta a history.jsonl")
    ap.add_argument("-o", "--output", default=None, help="Archivo MD de salida (default: stdout)")
    args = ap.parse_args()

    path = Path(args.history)
    if not path.exists():
        raise SystemExit(f"No existe: {path}")
    report = build_report(path)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Reporte escrito en {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
