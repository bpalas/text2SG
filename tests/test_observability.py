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
        with open(log.path, encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
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
