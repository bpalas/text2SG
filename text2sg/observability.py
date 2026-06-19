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
    """Tabla legible de una traza para imprimir a stderr. Placeholder mínimo —
    se completa en la Task 2."""
    return ""
