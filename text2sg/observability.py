"""Traza estructurada por corrida (sin dependencias externas).

Un RunLogger acumula eventos en memoria y, si está habilitado, los escribe
línea-a-línea (JSONL) a results/runs/<run_id>.jsonl. Cada evento de tipo "call"
registra un rol del pipeline (ner / extractor / verifier): backend, modelo,
tokens, latencia, estado y un dict de detalle libre. Un evento "summary" cierra
la corrida con los totales.

Habilitado=False -> no-op (no toca el disco). Útil como default en librería/tests.
"""
from __future__ import annotations

import json
import os
import time


class RunLogger:
    def __init__(self, run_id: str, out_dir: str = "results/runs",
                 enabled: bool = True, clock=time.time):
        self.run_id = run_id
        self.enabled = enabled
        self.clock = clock
        self.events: list[dict] = []
        self.path: str | None = None
        if enabled:
            os.makedirs(out_dir, exist_ok=True)
            self.path = os.path.join(out_dir, f"{run_id}.jsonl")

    def event(self, role: str, backend: str, model: str, status: str,
              tokens: int = 0, latency_s: float = 0.0,
              detail: dict | None = None) -> dict:
        rec = {
            "run_id": self.run_id,
            "ts": self.clock(),
            "kind": "call",
            "role": role,
            "backend": backend,
            "model": model,
            "status": status,
            "tokens": tokens,
            "latency_s": round(latency_s, 3),
            "detail": detail or {},
        }
        self.events.append(rec)
        self._append(rec)
        return rec

    def summary(self, mode: str, n_relations: int, n_entities: int,
                total_tokens: int) -> dict:
        # NOTE: summary records intentionally have no `latency_s` field — they
        # aggregate a whole run, not a single call, so per-call latency is N/A.
        rec = {
            "run_id": self.run_id,
            "ts": self.clock(),
            "kind": "summary",
            "mode": mode,
            "n_relations": n_relations,
            "n_entities": n_entities,
            "total_tokens": total_tokens,
            "n_calls": sum(1 for e in self.events if e.get("kind") == "call"),
        }
        self.events.append(rec)
        self._append(rec)
        return rec

    def _append(self, rec: dict) -> None:
        if not self.enabled or not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def format_trace(events: list[dict]) -> str:
    """Tabla legible de una traza para imprimir a stderr."""
    if not events:
        return "[trace] (sin eventos)"
    calls = [e for e in events if e.get("kind") == "call"]
    summary = next((e for e in events if e.get("kind") == "summary"), None)

    lines = ["", "─── trace ───────────────────────────────────────────────"]
    header = f"  {'role':<10} {'backend:model':<34} {'status':<7} {'tok':>6} {'s':>6}"
    lines.append(header)
    for e in calls:
        bm = f"{e.get('backend', '?')}:{e.get('model', '?')}"
        lines.append(
            f"  {e.get('role', '?'):<10} {bm:<34} "
            f"{e.get('status', '?'):<7} {e.get('tokens', 0):>6} "
            f"{e.get('latency_s', 0.0):>6.2f}"
        )
        detail = e.get("detail") or {}
        if detail:
            kv = "  ".join(f"{k}={v}" for k, v in detail.items())
            lines.append(f"             └ {kv}")
    if summary:
        lines.append("  " + "─" * 56)
        lines.append(
            f"  summary    mode={summary.get('mode', '?')}  "
            f"relations={summary.get('n_relations', 0)}  "
            f"entities={summary.get('n_entities', 0)}  "
            f"tokens={summary.get('total_tokens', 0)}  "
            f"calls={summary.get('n_calls', 0)}"
        )
    lines.append("─────────────────────────────────────────────────────────")
    return "\n".join(lines)
