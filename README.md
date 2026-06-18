# text2sg — Political Relation Extraction as Signed Graphs

![tests](https://github.com/bpalas/text2SG/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-blue)
![PyPI](https://img.shields.io/pypi/v/text2sg)

Turn Spanish-language news articles into **structured signed graphs** of political relations using any LLM backend (Gemini, Claude, GPT, or local Ollama).

```
"Boric respaldó las propuestas de Vallejo,           ┌─────────────┐
 mientras que Kast las calificó de 'peligrosas'."    │  endorses + │
                                              Boric ──┤             ├──▶ Vallejo
                                                      └─────────────┘
                                                      ┌─────────────┐
                                                      │  attacks  − │
                                              Kast ───┤             ├──▶ Vallejo
                                                      └─────────────┘
```

Each extracted relation has: `from_entity`, `to_entity`, `act_type`, `polarity`, `evidence_quote`, `confidence`.

---

## Why signed graphs?

Most political NLP stops at sentiment or named entities. Signed graphs encode **who does what to whom** — the minimal structure needed to detect alliances, oppositions, and polarization dynamics over time. This tool was built to feed a longitudinal network analysis of Chilean political elites across 2.1M news articles.

---

## Install

```bash
pip install -e ".[gemini]"     # Gemini (default)
pip install -e ".[claude]"     # Claude
pip install -e ".[openai]"     # GPT
pip install -e ".[ollama]"     # local (Ollama)
pip install -e ".[all]"        # all backends + dev deps
```

Set your API key:
```bash
export GEMINI_API_KEY=...      # or ANTHROPIC_API_KEY / OPENAI_API_KEY
```

---

## Quick start

```python
from text2sg.genome import Genome
from text2sg.extractor import run_extraction
from text2sg.llm_backends import GeminiClient

# Load the champion genome (prompt + validation config + analysis config)
genome = Genome.from_json("genomes/id15_champion.json")

article = {
    "body": "Boric respaldó las propuestas de Vallejo, mientras que Kast las calificó de peligrosas.",
    "unions": ["Gabriel Boric", "Camila Vallejo", "José Antonio Kast"],  # known actors
}

client = GeminiClient()
result = run_extraction(article, genome, client, model="gemini-2.0-flash-lite")

for rel in result["relations"]:
    print(f"{rel['from_entity']} --[{rel['act_type']}]--> {rel['to_entity']}")
# Gabriel Boric --[endorses]--> Camila Vallejo
# José Antonio Kast --[attacks]--> Camila Vallejo
```

**No actors pre-computed? Use end-to-end mode** (NER + extraction in one call):

```python
result = run_extraction(article_without_unions, genome, client,
                        model="gemini-2.0-flash-lite", end2end=True)
```

---

## Genome: the three-artifact design

A genome encodes the full extraction strategy as three independently mutable artifacts:

| Artifact | What it controls | Cost |
|----------|-----------------|------|
| **A** — `prompt_text` | The LLM extraction prompt | per-call |
| **B** — `ValidationConfig` | Deterministic post-processing filters | $0 |
| **C** — `AnalysisConfig` | Pre-extraction scaffolding (actor dossier, alias map, …) | $0 |

```json
{
  "prompt_text": "Extract political relations...",
  "validation": {
    "require_evidence_substring": true,
    "min_quote_len": 8,
    "allowed_act_types": ["endorses", "attacks", "allies_with", ...]
  },
  "analysis": {
    "actor_dossier": true,
    "alias_map": true,
    "direction_scaffold": false
  }
}
```

Artifacts B and C cost nothing to evaluate — changes to them are screened before spending any API budget. This design enables the [evolutionary optimizer](https://github.com/bpalas/text2graph-evolve) to iterate cheaply.

---

## Benchmark — Synthetic Oracle v2

Evaluated on 287 articles, 914 gold relations (truth planted by `claude-opus-4-8`, difficulty 1–10, stratified by domain × register).

| Model | Precision | Recall | F0.5 |
|-------|-----------|--------|------|
| gemini-2.0-flash-lite (id15) | 0.940 | 0.884 | 0.928 |
| gemini-2.0-flash-lite (id13) | 0.928 | 0.901 | 0.922 |
| gemini-2.0-flash-lite (id18) | 0.927 | 0.905 | 0.921 |

> **F0.5** weights precision 2× over recall — right for downstream graph analysis where false edges corrupt community structure more than missed edges.

The benchmark uses a **fully synthetic oracle** (no real corpus required): articles are generated with planted ground truth, so evaluation is 100% honest and reproducible without any data-license restrictions.

---

## Supported relation types

| `act_type` | Polarity | Example |
|-----------|---------|---------|
| `endorses` | + | "Boric respaldó la propuesta" |
| `attacks` | − | "Kast calificó de peligrosas" |
| `allies_with` | + | "firmaron un acuerdo conjunto" |
| `calls_on` | neutral | "exigió al ministro que" |
| `distances_from` | − | "se desmarcó de la postura" |
| `questions` | − | "cuestionó la decisión de" |
| `negotiates_with` | neutral | "negoció con la oposición" |
| `competes_with` | − | "compite directamente contra" |
| `accuses` | − | "acusó de corrupción a" |

---

## Multi-backend support

```python
from text2sg.llm_backends import GeminiClient, AnthropicClient, OpenAIClient, OllamaClient

# Gemini (default, cheapest at scale)
client = GeminiClient()

# Claude — strong on formal political text
client = AnthropicClient()

# GPT
client = OpenAIClient()

# Local — zero API cost, useful for development
client = OllamaClient()   # requires: ollama serve && ollama pull qwen2.5:7b
```

All backends expose the same interface — swap without changing extraction code.

---

## Project structure

```
text2sg/
├── extractor.py      — build_prompt + LLM call + parsing + validation
├── genome.py         — Genome dataclass (A + B + C) with JSON serialization
├── validation.py     — Artifact B: deterministic post-processing ($0)
├── analysis.py       — Artifact C: pre-extraction scaffolding ($0)
├── rubric.py         — precision/recall metrics (directed + undirected)
├── llm_backends.py   — Gemini / Claude / GPT / Ollama (same interface)
├── config.py         — base config + seed prompt
└── prompts/          — system prompts for meta-agents (diagnose/propose/cross/fresh)
tests/                — 100+ tests, all run offline without API key
```

---

## How the champion genome was found

The champion genome was discovered by [text2graph-evolve](https://github.com/bpalas/text2graph-evolve), a multi-agent evolutionary optimizer that runs on top of this package. The optimizer treats the genome as a three-artifact individual, evaluates mutations against the synthetic oracle, and manages a Pareto front across precision/recall gradients using a panel of LLM meta-agents (diagnose, propose, cross-pollinate, fresh-design).

---

## Citation

If you use this tool in academic work:

```bibtex
@software{text2sg2026,
  author  = {Palacios, Benjamin},
  title   = {text2sg: Political Relation Extraction as Signed Graphs},
  year    = {2026},
  url     = {https://github.com/bpalas/text2SG}
}
```

---

## License

MIT — see [LICENSE](LICENSE). The synthetic oracle and champion genomes are included. Real corpus data (IMFD) is not redistributed.
