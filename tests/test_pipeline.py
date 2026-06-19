"""Tests offline del pipeline — no requieren API key ni Ollama."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from text2sg.pipeline import AgentDef, PipelineConfig, _actors_to_union, extract_text
from text2sg.genome import Genome


# ── AgentDef ──────────────────────────────────────────────────────────────── #

class TestAgentDef:
    def test_from_str_gemini(self):
        a = AgentDef.from_str("gemini:gemini-2.0-flash-lite")
        assert a.backend == "gemini"
        assert a.model == "gemini-2.0-flash-lite"

    def test_from_str_ollama_with_colon_in_model(self):
        # model names like qwen2.5:7b have a colon — must split on FIRST colon only
        a = AgentDef.from_str("ollama:qwen2.5:7b")
        assert a.backend == "ollama"
        assert a.model == "qwen2.5:7b"

    def test_from_str_anthropic(self):
        a = AgentDef.from_str("anthropic:claude-haiku-4-5")
        assert a.backend == "anthropic"
        assert a.model == "claude-haiku-4-5"

    def test_from_str_missing_colon_raises(self):
        with pytest.raises(ValueError, match="backend:model"):
            AgentDef.from_str("ollama")

    def test_from_str_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            AgentDef.from_str("cohere:command-r")

    def test_str_roundtrip(self):
        spec = "ollama:qwen2.5:7b"
        assert str(AgentDef.from_str(spec)) == spec


# ── PipelineConfig ─────────────────────────────────────────────────────────── #

class TestPipelineConfig:
    def test_requires_extractor(self):
        with pytest.raises((ValueError, TypeError)):
            PipelineConfig()

    def test_defaults_ner_to_extractor_in_end2end(self):
        ext = AgentDef("ollama", "qwen2.5:7b")
        cfg = PipelineConfig(mode="end2end", extractor=ext)
        assert cfg.ner is ext

    def test_explicit_ner_kept(self):
        ext = AgentDef("ollama", "qwen2.5:7b")
        ner = AgentDef("gemini", "gemini-2.0-flash-lite")
        cfg = PipelineConfig(mode="end2end", extractor=ext, ner=ner)
        assert cfg.ner is ner

    def test_given_entities_no_ner_default(self):
        ext = AgentDef("ollama", "qwen2.5:7b")
        cfg = PipelineConfig(mode="given_entities", extractor=ext)
        assert cfg.ner is None   # not needed

    def test_from_cli_args(self):
        cfg = PipelineConfig.from_cli_args(
            mode="end2end",
            extractor="ollama:qwen2.5:7b",
            ner="gemini:gemini-2.0-flash-lite",
            verifier="anthropic:claude-haiku-4-5",
        )
        assert cfg.mode == "end2end"
        assert cfg.extractor.model == "qwen2.5:7b"
        assert cfg.ner.model == "gemini-2.0-flash-lite"
        assert cfg.verifier.model == "claude-haiku-4-5"

    def test_from_cli_args_no_verifier(self):
        cfg = PipelineConfig.from_cli_args("given_entities", "ollama:qwen2.5:7b")
        assert cfg.verifier is None


# ── _actors_to_union ──────────────────────────────────────────────────────── #

class TestActorsToUnion:
    def test_basic(self):
        union = _actors_to_union(["Gabriel Boric", "Camila Vallejo"])
        assert "U1" in union
        assert union["U1"]["canonical_names"] == ["Gabriel Boric"]
        assert union["U2"]["canonical_names"] == ["Camila Vallejo"]
        assert all(v["type"] == "roster_actor" for v in union.values())

    def test_empty(self):
        assert _actors_to_union([]) == {}


# ── extract_text (mocked LLM) ─────────────────────────────────────────────── #

def _make_mock_client(relations=None):
    """Build a mock LLM client that returns a valid JSON extraction."""
    if relations is None:
        relations = [
            {
                "from_entity": "Gabriel Boric",
                "to_entity": "Camila Vallejo",
                "act_type": "endorses",
                "polarity": "positive",
                "issue": "political_coalitions",
                "evidence_quote": "Boric respaldó las propuestas de Vallejo",
            }
        ]
    import json
    payload = json.dumps({"entities": [], "relations": relations})
    mock_resp = MagicMock()
    mock_resp.text = payload
    mock_resp.usage_metadata.prompt_token_count = 100
    mock_resp.usage_metadata.candidates_token_count = 50
    client = MagicMock()
    client.models.generate_content.return_value = mock_resp
    return client


class TestExtractText:
    def _genome(self):
        return Genome.from_seed()

    def test_given_entities_returns_relations(self):
        genome = self._genome()
        config = PipelineConfig(
            mode="given_entities",
            extractor=AgentDef("ollama", "qwen2.5:7b"),
        )
        mock_client = _make_mock_client()

        with patch.object(config.extractor, "make_client", return_value=mock_client):
            result = extract_text(
                "Boric respaldó las propuestas de Vallejo.",
                genome, config,
                actors=["Gabriel Boric", "Camila Vallejo"],
            )

        assert isinstance(result["relations"], list)
        assert len(result["relations"]) >= 1
        assert result["tokens"]["extractor"] > 0
        assert result["tokens"]["ner"] == 0   # not end2end

    def test_end2end_calls_ner_first(self):
        genome = self._genome()
        ext_agent = AgentDef("ollama", "qwen2.5:7b")
        config = PipelineConfig(mode="end2end", extractor=ext_agent)

        ner_payload = '{"actors": [{"uid": "U1", "canonical_name": "Boric", "type": "roster_actor"}]}'
        ext_payload = '{"entities": [], "relations": []}'

        ner_resp = MagicMock()
        ner_resp.text = ner_payload
        ner_resp.usage_metadata.prompt_token_count = 80
        ner_resp.usage_metadata.candidates_token_count = 20

        ext_resp = MagicMock()
        ext_resp.text = ext_payload
        ext_resp.usage_metadata.prompt_token_count = 200
        ext_resp.usage_metadata.candidates_token_count = 30

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [ner_resp, ext_resp]

        with patch.object(ext_agent, "make_client", return_value=mock_client):
            result = extract_text("Boric anunció medidas.", genome, config)

        assert result["tokens"]["ner"] > 0
        assert result["tokens"]["extractor"] > 0
        # NER and extractor each called generate_content once
        assert mock_client.models.generate_content.call_count == 2

    def test_verifier_called_when_configured(self):
        genome = self._genome()
        ext_agent = AgentDef("ollama", "qwen2.5:7b")
        ver_agent = AgentDef("anthropic", "claude-haiku-4-5")
        config = PipelineConfig(
            mode="given_entities",
            extractor=ext_agent,
            verifier=ver_agent,
        )

        ext_client = _make_mock_client()
        ver_resp = MagicMock()
        ver_resp.text = '["keep"]'   # verify_relations parses its own format
        ver_resp.usage_metadata.prompt_token_count = 50
        ver_resp.usage_metadata.candidates_token_count = 10
        ver_client = MagicMock()
        ver_client.models.generate_content.return_value = ver_resp

        with (
            patch.object(ext_agent, "make_client", return_value=ext_client),
            patch.object(ver_agent, "make_client", return_value=ver_client),
        ):
            result = extract_text(
                "Boric respaldó las propuestas de Vallejo.",
                genome, config,
                actors=["Gabriel Boric", "Camila Vallejo"],
            )

        assert ver_client.models.generate_content.called
        assert result["tokens"]["verifier"] > 0

    def test_no_verifier_skips_verify_call(self):
        genome = self._genome()
        ext_agent = AgentDef("ollama", "qwen2.5:7b")
        config = PipelineConfig(mode="given_entities", extractor=ext_agent)
        ext_client = _make_mock_client()

        with patch.object(ext_agent, "make_client", return_value=ext_client):
            result = extract_text("Texto.", genome, config, actors=["Actor A"])

        assert result["tokens"]["verifier"] == 0

    def test_token_total_is_sum(self):
        genome = self._genome()
        ext_agent = AgentDef("ollama", "qwen2.5:7b")
        config = PipelineConfig(mode="given_entities", extractor=ext_agent)
        ext_client = _make_mock_client()

        with patch.object(ext_agent, "make_client", return_value=ext_client):
            result = extract_text("Texto.", genome, config, actors=["Actor A"])

        tok = result["tokens"]
        assert tok["total"] == tok["ner"] + tok["extractor"] + tok["verifier"]

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

    def test_logger_writes_jsonl_through_pipeline(self, tmp_path):
        import json as _json
        from text2sg.observability import RunLogger
        genome = self._genome()
        ext_agent = AgentDef("ollama", "qwen2.5:7b")
        config = PipelineConfig(mode="given_entities", extractor=ext_agent)
        ext_client = _make_mock_client()
        log = RunLogger(run_id="t", out_dir=str(tmp_path), enabled=True)

        with patch.object(ext_agent, "make_client", return_value=ext_client):
            extract_text("Texto.", genome, config, actors=["Actor A"], logger=log)

        with open(log.path, encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
        kinds = [_json.loads(ln)["kind"] for ln in lines]
        assert "call" in kinds
        assert "summary" in kinds


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
