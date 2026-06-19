# Observabilidad de los TRACKs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persistir y mostrar una traza estructurada por corrida (JSONL) que registre cada llamada LLM de los tres roles (ner / extractor / verifier) en ambos modos (given_entities, end2end), para poder verificar en dev que cada TRACK funciona con cada backend.

**Architecture:** Un `RunLogger` sin dependencias (stdlib `json`/`os`/`time`) que acumula eventos en memoria y los escribe línea-a-línea a `results/runs/<run_id>.jsonl`. El pipeline (`extract_text`) lo recibe como parámetro opcional e instrumenta cada rol con tiempo, tokens y resultado. El CLI crea el logger, expone flags `--log-dir`/`--no-log`, e imprime una tabla-traza a stderr al terminar. Logger ausente o deshabilitado → no-op, así los tests existentes y el uso como librería no cambian.

**Tech Stack:** Python 3.10+, stdlib only, pytest (con `tmp_path` para aislar la escritura de archivos).

---

## File Structure

- **Create** `text2sg/observability.py` — `RunLogger` (eventos en memoria + escritura JSONL) y `format_trace(events) -> str` (tabla legible para stderr). Responsabilidad única: registrar y formatear la traza de una corrida.
- **Modify** `text2sg/pipeline.py` — `extract_text` acepta `logger=None`; instrumenta ner/extractor/verifier y emite un evento de resumen. Sin logger usa un `RunLogger` deshabilitado (no-op).
- **Modify** `text2sg/__main__.py` — flags `--log-dir` (default `results/runs`) y `--no-log`; crea el `RunLogger` con un `run_id` por timestamp; imprime la traza con `format_trace` y la ruta del archivo.
- **Modify** `.gitignore` — ignorar `results/runs/` (las trazas son voluminosas y regenerables).
- **Create** `tests/test_observability.py` — tests offline del `RunLogger` y `format_trace`.
- **Modify** `tests/test_pipeline.py` — un test nuevo que pasa un logger y verifica los eventos emitidos.
- **Modify** `README.md` — nota breve sobre `--log-dir` y el formato de la traza.

---

## Task 1: `RunLogger` — eventos en memoria + escritura JSONL

**Files:**
- Create: `text2sg/observability.py`
- Test: `tests/test_observability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_observability.py
"""Tests offline de observabilidad — no requieren API key, Ollama ni red."""
from __future__ import annotations

import json

from text2sg.observability import RunLogger, format_trace


class TestRunLogger:
    def test_event_recorded_in_memory(self):
        log = RunLogger(run_id="r1", enabled=False)
        rec = log.event("extractor", "gemini", "gemini-2.0-flash-lite",
                        status="ok", tokens=150, latency_s=0.42,
                        detail={"n_relations": 3})
        assert rec["kind"] == "call"
        assert rec["role"] == "extractor"
        assert rec["backend"] == "gemini"
        assert rec["model"] == "gemini-2.0-flash-lite"
        assert rec["status"] == "ok"
        assert rec["tokens"] == 150
        assert rec["latency_s"] == 0.42
        assert rec["detail"] == {"n_relations": 3}
        assert log.events == [rec]

    def test_disabled_logger_writes_no_file(self, tmp_path):
        log = RunLogger(run_id="r1", out_dir=str(tmp_path), enabled=False)
        log.event("ner", "ollama", "qwen2.5:7b", status="ok")
        assert list(tmp_path.iterdir()) == []
        assert log.path is None

    def test_enabled_logger_appends_jsonl(self, tmp_path):
        log = RunLogger(run_id="r1", out_dir=str(tmp_path), enabled=True)
        log.event("ner", "ollama", "qwen2.5:7b", status="ok", tokens=10)
        log.event("extractor", "ollama", "qwen2.5:7b", status="empty", tokens=20)
        lines = log.path.read_text(encoding="utf-8").strip().splitlines() \
            if hasattr(log.path, "read_text") else \
            open(log.path, encoding="utf-8").read().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["role"] == "ner"
        assert first["run_id"] == "r1"

    def test_summary_event(self, tmp_path):
        log = RunLogger(run_id="r1", out_dir=str(tmp_path), enabled=True)
        log.event("extractor", "gemini", "m", status="ok", tokens=100)
        rec = log.summary(mode="given_entities", n_relations=2,
                          n_entities=0, total_tokens=100)
        assert rec["kind"] == "summary"
        assert rec["mode"] == "given_entities"
        assert rec["n_relations"] == 2
        assert rec["total_tokens"] == 100
        assert rec["n_calls"] == 1

    def test_clock_injection(self):
        log = RunLogger(run_id="r1", enabled=False, clock=lambda: 1234.0)
        rec = log.event("extractor", "gemini", "m", status="ok")
        assert rec["ts"] == 1234.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_observability.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'text2sg.observability'`

- [ ] **Step 3: Write minimal implementation**

```python
# text2sg/observability.py
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
```

> Nota sobre el test `test_enabled_logger_appends_jsonl`: `log.path` es un `str`, así que el `hasattr(..., "read_text")` será falso y se usará la rama `open(...)`. La rama `read_text` solo aplicaría si `path` fuese un `pathlib.Path`; se deja por robustez.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_observability.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add text2sg/observability.py tests/test_observability.py
git commit -m "feat(obs): RunLogger — structured per-run JSONL trace

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `format_trace` — tabla legible para stderr

**Files:**
- Modify: `text2sg/observability.py`
- Test: `tests/test_observability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_observability.py  (añadir al final)
class TestFormatTrace:
    def test_empty_events(self):
        assert "(sin eventos)" in format_trace([])

    def test_renders_calls_and_summary(self):
        events = [
            {"kind": "call", "role": "ner", "backend": "ollama",
             "model": "qwen2.5:7b", "status": "ok", "tokens": 100,
             "latency_s": 0.5, "detail": {"n_actors": 3}},
            {"kind": "call", "role": "extractor", "backend": "gemini",
             "model": "gemini-2.0-flash-lite", "status": "empty", "tokens": 200,
             "latency_s": 1.2, "detail": {"n_relations": 0}},
            {"kind": "summary", "mode": "end2end", "n_relations": 0,
             "n_entities": 3, "total_tokens": 300, "n_calls": 2},
        ]
        out = format_trace(events)
        assert "ner" in out
        assert "extractor" in out
        assert "ollama:qwen2.5:7b" in out
        assert "gemini:gemini-2.0-flash-lite" in out
        assert "ok" in out and "empty" in out
        assert "300" in out          # total tokens en el summary
        assert "end2end" in out      # mode en el summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_observability.py::TestFormatTrace -q`
Expected: FAIL — `assert "(sin eventos)" in ""` (placeholder devuelve "")

- [ ] **Step 3: Write minimal implementation**

Reemplazar la función `format_trace` placeholder en `text2sg/observability.py` por:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_observability.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add text2sg/observability.py tests/test_observability.py
git commit -m "feat(obs): format_trace — human-readable run trace table

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Instrumentar `extract_text` en el pipeline

**Files:**
- Modify: `text2sg/pipeline.py:18-21` (imports) y `text2sg/pipeline.py:115-179` (`extract_text`)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py  (añadir dentro de class TestExtractText)
    def test_logger_records_events(self):
        from text2sg.observability import RunLogger
        genome = self._genome()
        ext_agent = AgentDef("ollama", "qwen2.5:7b")
        config = PipelineConfig(mode="given_entities", extractor=ext_agent)
        ext_client = _make_mock_client()
        log = RunLogger(run_id="t", enabled=False)

        with patch.object(ext_agent, "make_client", return_value=ext_client):
            extract_text("Texto.", genome, config,
                         actors=["Actor A"], logger=log)

        kinds = [e["kind"] for e in log.events]
        assert "call" in kinds
        assert "summary" in kinds
        ext_calls = [e for e in log.events
                     if e["kind"] == "call" and e["role"] == "extractor"]
        assert len(ext_calls) == 1
        assert ext_calls[0]["backend"] == "ollama"
        assert ext_calls[0]["model"] == "qwen2.5:7b"
        assert ext_calls[0]["tokens"] > 0

    def test_end2end_logger_records_ner_and_extractor(self):
        from text2sg.observability import RunLogger
        genome = self._genome()
        ext_agent = AgentDef("ollama", "qwen2.5:7b")
        config = PipelineConfig(mode="end2end", extractor=ext_agent)

        ner_resp = MagicMock()
        ner_resp.text = '{"actors": [{"uid": "U1", "canonical_name": "Boric", "type": "roster_actor"}]}'
        ner_resp.usage_metadata.prompt_token_count = 80
        ner_resp.usage_metadata.candidates_token_count = 20
        ext_resp = MagicMock()
        ext_resp.text = '{"entities": [], "relations": []}'
        ext_resp.usage_metadata.prompt_token_count = 200
        ext_resp.usage_metadata.candidates_token_count = 30
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [ner_resp, ext_resp]
        log = RunLogger(run_id="t", enabled=False)

        with patch.object(ext_agent, "make_client", return_value=mock_client):
            extract_text("Boric anunció medidas.", genome, config, logger=log)

        roles = [e["role"] for e in log.events if e["kind"] == "call"]
        assert "ner" in roles
        assert "extractor" in roles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::TestExtractText::test_logger_records_events -q`
Expected: FAIL — `extract_text() got an unexpected keyword argument 'logger'`

- [ ] **Step 3: Write minimal implementation**

En `text2sg/pipeline.py`, cambiar el bloque de imports del tope (líneas 18-21):

```python
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
```

Reemplazar la firma y cuerpo de `extract_text` (actual líneas 115-179) por:

```python
def extract_text(
    text: str,
    genome,
    config: PipelineConfig,
    actors: list[str] | None = None,
    article_id: str = "article",
    logger=None,
) -> dict:
    """Extract political relations from a single text string.

    Args:
        text:       the article body (Spanish)
        genome:     Genome object — prompt_text + ValidationConfig + AnalysisConfig
        config:     PipelineConfig — which model handles each role
        actors:     known actor names for given_entities mode (ignored in end2end)
        article_id: identifier included in the result
        logger:     optional RunLogger for per-call observability. None -> no-op.

    Returns:
        {
            "article_id": str,
            "relations": [...],
            "entities":  [...],
            "tokens":    {"ner": int, "extractor": int, "verifier": int, "total": int},
        }
    """
    from text2sg.extractor import extract_entities, extract_article, verify_relations
    from text2sg.observability import RunLogger

    if logger is None:
        logger = RunLogger(run_id="adhoc", enabled=False)

    token_counts: dict[str, int] = {"ner": 0, "extractor": 0, "verifier": 0}

    # ── 1. NER pass (end2end only) ────────────────────────────────────────── #
    if config.mode == "end2end":
        ner_agent = config.ner  # already defaulted to extractor in __post_init__
        ner_client = ner_agent.make_client()
        t0 = time.time()
        union, ner_tokens = extract_entities(text, ner_agent.model, ner_client)
        logger.event(
            "ner", ner_agent.backend, ner_agent.model,
            status="ok" if union else "empty",
            tokens=ner_tokens, latency_s=time.time() - t0,
            detail={"n_actors": len(union)},
        )
        token_counts["ner"] = ner_tokens
    else:
        union = _actors_to_union(actors or [])

    # ── 2. Extraction pass ────────────────────────────────────────────────── #
    g = copy.copy(genome)
    g.model = config.extractor.model
    g.verify = False          # we run verify separately with its own client

    ext_client = config.extractor.make_client()
    t0 = time.time()
    result = extract_article(
        article_id, text, union, g,
        few_shot_examples=[],
        client=ext_client,
    )
    ext_tokens = result.get("tokens", 0)
    rels = result.get("relations", [])
    logger.event(
        "extractor", config.extractor.backend, config.extractor.model,
        status="ok" if rels else "empty",
        tokens=ext_tokens, latency_s=time.time() - t0,
        detail={"n_relations": len(rels),
                "n_entities": len(result.get("entities", []))},
    )
    token_counts["extractor"] = ext_tokens

    # ── 3. Optional agentic verify pass ──────────────────────────────────── #
    if config.verifier is not None:
        ver_client = config.verifier.make_client()
        t0 = time.time()
        verified_rels, ver_tokens = verify_relations(
            result.get("relations", []), text, config.verifier.model, ver_client,
        )
        result["relations"] = verified_rels
        token_counts["verifier"] = ver_tokens
        logger.event(
            "verifier", config.verifier.backend, config.verifier.model,
            status="ok", tokens=ver_tokens, latency_s=time.time() - t0,
            detail={"n_relations_out": len(verified_rels)},
        )

    token_counts["total"] = sum(token_counts.values())
    result["tokens"] = token_counts
    logger.summary(
        mode=config.mode,
        n_relations=len(result.get("relations", [])),
        n_entities=len(result.get("entities", [])),
        total_tokens=token_counts["total"],
    )
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: PASS (todos los tests previos de pipeline + los 2 nuevos)

- [ ] **Step 5: Commit**

```bash
git add text2sg/pipeline.py tests/test_pipeline.py
git commit -m "feat(obs): instrument extract_text with per-role RunLogger events

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Flags `--log-dir` / `--no-log` y traza en el CLI

**Files:**
- Modify: `text2sg/__main__.py:45-76` (args de `run`) y `text2sg/__main__.py:93-155` (`_cmd_run`)
- Test: `tests/test_pipeline.py` (test de integración del CLI con cliente mockeado)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py  (añadir al final del archivo, fuera de las clases)
class TestCliRunWritesTrace:
    def test_run_writes_jsonl_trace(self, tmp_path, monkeypatch, capsys):
        import json as _json
        from unittest.mock import MagicMock, patch
        import text2sg.__main__ as cli

        # cliente LLM mockeado: devuelve una relación válida
        payload = _json.dumps({"entities": [], "relations": [
            {"from_entity": "A", "to_entity": "B", "act_type": "endorses",
             "polarity": "positive", "issue": "x",
             "evidence_quote": "A respaldó a B"}]})
        resp = MagicMock()
        resp.text = payload
        resp.usage_metadata.prompt_token_count = 10
        resp.usage_metadata.candidates_token_count = 5
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = resp

        log_dir = tmp_path / "runs"
        argv = ["prog", "run",
                "--extractor", "ollama:qwen2.5:7b",
                "--actors", "A", "B",
                "--text", "A respaldó a B",
                "--log-dir", str(log_dir)]
        monkeypatch.setattr(cli.sys, "argv", argv)

        with patch("text2sg.pipeline.AgentDef.make_client",
                   return_value=mock_client):
            cli.main()

        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        kinds = [_json.loads(ln)["kind"] for ln in lines]
        assert "call" in kinds and "summary" in kinds
        # la traza se imprime a stderr
        err = capsys.readouterr().err
        assert "trace" in err

    def test_no_log_skips_file(self, tmp_path, monkeypatch):
        import json as _json
        from unittest.mock import MagicMock, patch
        import text2sg.__main__ as cli

        payload = _json.dumps({"entities": [], "relations": []})
        resp = MagicMock()
        resp.text = payload
        resp.usage_metadata.prompt_token_count = 10
        resp.usage_metadata.candidates_token_count = 5
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = resp

        log_dir = tmp_path / "runs"
        argv = ["prog", "run",
                "--extractor", "ollama:qwen2.5:7b",
                "--actors", "A",
                "--text", "texto",
                "--log-dir", str(log_dir),
                "--no-log"]
        monkeypatch.setattr(cli.sys, "argv", argv)

        with patch("text2sg.pipeline.AgentDef.make_client",
                   return_value=mock_client):
            cli.main()

        assert not log_dir.exists() or list(log_dir.glob("*.jsonl")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::TestCliRunWritesTrace -q`
Expected: FAIL — `unrecognized arguments: --log-dir` (SystemExit del argparse)

- [ ] **Step 3: Write minimal implementation**

En `text2sg/__main__.py`, añadir dos argumentos al parser de `run` (justo después del bloque `--output`, antes del comentario `# -- models --`):

```python
    rp.add_argument(
        "--log-dir", metavar="DIR", default="results/runs",
        help="Directory for per-run JSONL traces (default: results/runs)",
    )
    rp.add_argument(
        "--no-log", action="store_true",
        help="Disable writing the run trace to disk",
    )
```

En `_cmd_run`, cambiar los imports del tope de la función:

```python
def _cmd_run(args: argparse.Namespace) -> None:
    import time
    from text2sg.genome import Genome
    from text2sg.observability import RunLogger, format_trace
    from text2sg.pipeline import PipelineConfig, extract_text
```

Después de construir `config` y antes del bloque `# -- log plan --`, crear el logger:

```python
    # -- run logger -- #
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    logger = RunLogger(
        run_id=run_id,
        out_dir=args.log_dir,
        enabled=not args.no_log,
    )
```

Añadir `import os` al tope del módulo (junto a `import argparse`, `import json`, `import sys`).

Pasar `logger=logger` a la llamada `extract_text`:

```python
    result = extract_text(
        text, genome, config,
        actors=args.actors,
        article_id=args.file or "stdin",
        logger=logger,
    )
```

Después de la salida (`_pretty_print` / json), imprimir la traza y la ruta a stderr:

```python
    # -- trace -- #
    print(format_trace(logger.events), file=sys.stderr)
    if logger.path:
        print(f"[text2sg] trace saved to {logger.path}", file=sys.stderr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py::TestCliRunWritesTrace -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add text2sg/__main__.py tests/test_pipeline.py
git commit -m "feat(obs): --log-dir/--no-log flags + print run trace to stderr

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Ignorar trazas en git + nota en README

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Ignorar el directorio de trazas**

Añadir a `.gitignore` debajo del bloque "Resultados de corridas evolutivas":

```
# Trazas de observabilidad por corrida (regenerables)
results/runs/
```

- [ ] **Step 2: Documentar en README**

En `README.md`, después de la sección "Quick start" (antes de "## Genome"), añadir:

```markdown
## Observability

Every CLI run writes a structured JSONL trace — one event per LLM call
(role, backend, model, tokens, latency, status) plus a summary — so you can
verify each track works with each backend:

```bash
python -m text2sg run \
    --mode end2end \
    --ner       ollama:qwen2.5:7b \
    --extractor gemini:gemini-2.0-flash-lite \
    --file      articulo.txt
# ...prints a trace table to stderr and saves results/runs/<timestamp>.jsonl
```

Disable with `--no-log`, or change the location with `--log-dir DIR`.
```

- [ ] **Step 3: Verify full suite still green**

Run: `python -m pytest tests/ -q`
Expected: PASS (todos los tests previos + los nuevos de observabilidad)

- [ ] **Step 4: Commit**

```bash
git add .gitignore README.md
git commit -m "docs(obs): gitignore run traces + README observability section

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notas de diseño y alcance

- **Lo que esto SÍ hace:** registra y muestra qué hizo cada rol (ner/extractor/verifier), con qué backend/modelo, cuántos tokens, cuánto tardó y si devolvió algo (`status="empty"` cuando el rol no produjo salida). Permite confirmar en dev que ambos TRACKs (given_entities / end2end) funcionan con cada backend.
- **Lo que esto NO hace (queda para el plan de robustez):** no cambia el manejo de errores. Hoy `extract_entities` y `verify_relations` se tragan las excepciones y devuelven vacío; esta capa lo *observa* como `status="empty"`, pero el retry/backoff y el manejo de rate limits de Claude (429, `retry-after`) son un subsistema aparte.
- **Sin dependencias nuevas:** stdlib only, coherente con `dependencies = []` del `pyproject.toml`.
- **Compatibilidad:** `extract_text(..., logger=None)` es retrocompatible; los tests existentes y el uso como librería no cambian.
```
