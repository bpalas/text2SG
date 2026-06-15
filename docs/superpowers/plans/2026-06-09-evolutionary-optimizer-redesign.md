# Rediseño Evolutivo del Optimizer — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar el hill-climbing greedy del `swarm_optimizer` por un sistema evolutivo estilo RoboPhD: archivo open-ended con ELO + championship, genoma de 2 artefactos (prompt + validación determinista), fitness F0.5 + piso de recall + costo, y mutación por diff/cross-pollination.

**Architecture:** Módulos puros y testeables (genome, validation, fitness, elo, archive) por debajo; orquestación (mutate, extractor, loop) por encima. Frontera nítida entre lógica de búsqueda determinista y llamadas a Gemini. Se construye bottom-up: cada módulo puro primero (TDD sin API), luego la orquestación con cliente LLM mockeado.

**Tech Stack:** Python 3.12, dataclasses, pytest, google-genai (Gemini), rapidfuzz, pandas. Sin dependencias nuevas.

**Spec:** `docs/superpowers/specs/2026-06-09-text2graph-evolve-roadmap-design.md`

---

## Estructura de archivos

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `swarm_optimizer/genome.py` | crear | `Genome` + `ValidationConfig` + seed. Modelo del candidato (2 artefactos). |
| `swarm_optimizer/validation.py` | crear | Artefacto B determinista. `apply_validation(parsed, body, union, vcfg)`. Puro. |
| `swarm_optimizer/fitness.py` | crear | `f_beta`, `fitness` (F0.5 + piso recall + costo). Puro. |
| `swarm_optimizer/elo.py` | crear | `expected_score`, `update_pairwise`, `sample_parent`. Puro. |
| `swarm_optimizer/archive.py` | reescribir | Archivo open-ended: ELO, linaje, `add`/`select_parent`/`top_by_elo`/`champion`. |
| `swarm_optimizer/mutate.py` | crear | `apply_diff`, `parse_search_replace`, `apply_validation_patch` (puros) + `diagnose`/`propose`/`cross_pollinate` (LLM). |
| `swarm_optimizer/extractor.py` | modificar | Integrar validación + verificación agéntica opcional. |
| `swarm_optimizer/splits.py` | modificar | Añadir `subsample(eval_ids, k, seed)`. |
| `swarm_optimizer/loop.py` | reescribir | Orquesta skirmish/championship/archivo/budget/parada. |
| `swarm_optimizer/run.py` | modificar | Entry point al nuevo `run_loop`. |
| `swarm_optimizer/rubric.py` | sin cambios | Regresión: `test_rubric.py` queda verde. |
| `swarm_optimizer/config.py` | sin cambios | Se conserva para compatibilidad/migración. |

Tests nuevos/reescritos: `test_genome.py`, `test_validation.py`, `test_fitness.py`, `test_elo.py`, `test_archive.py` (reescrito), `test_mutate.py`, `test_extractor.py`, `test_splits.py` (ampliado), `test_loop.py` (reescrito).

---

## Task 1: Modelo del genoma (`genome.py`)

**Files:**
- Create: `swarm_optimizer/genome.py`
- Test: `swarm_optimizer/tests/test_genome.py`

- [ ] **Step 1: Write the failing tests**

```python
# swarm_optimizer/tests/test_genome.py
from swarm_optimizer.genome import Genome, ValidationConfig


def test_seed_has_prompt_and_default_validation():
    g = Genome.from_seed()
    assert "extractor de relaciones" in g.prompt_text.lower()
    assert g.architecture == "one_pass"
    assert g.verify is False
    assert isinstance(g.validation, ValidationConfig)
    assert g.validation.require_evidence_substring is True


def test_roundtrip_json_preserves_validation():
    g = Genome.from_seed()
    g.validation.min_quote_len = 12
    g.verify = True
    restored = Genome.from_json(g.to_json())
    assert restored.validation.min_quote_len == 12
    assert restored.verify is True
    assert restored.prompt_text == g.prompt_text


def test_validation_defaults():
    v = ValidationConfig()
    assert v.dedup is True
    assert v.enforce_polarity_consistency is False
    assert v.allowed_act_types is None
    assert v.max_relations_per_article is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_genome.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_optimizer.genome'`

- [ ] **Step 3: Write minimal implementation**

```python
# swarm_optimizer/genome.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

from swarm_optimizer.config import SEED_PROMPT


@dataclass
class ValidationConfig:
    """Artefacto B: post-proceso determinista (costo $0)."""
    require_evidence_substring: bool = True
    min_quote_len: int = 8
    normalize_passive_direction: bool = True
    dedup: bool = True
    enforce_polarity_consistency: bool = False
    allowed_act_types: list[str] | None = None
    max_relations_per_article: int | None = None


@dataclass
class Genome:
    """Artefacto A (prompt) + flags + Artefacto B (validation)."""
    prompt_text: str
    few_shots: list[str] = field(default_factory=list)
    architecture: str = "one_pass"          # "one_pass" | "given_entities"
    model: str = "gemini-2.5-flash"
    verify: bool = False                     # verificación agéntica en inferencia
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        data = dict(data)
        vc = data.pop("validation", {}) or {}
        return cls(validation=ValidationConfig(**vc), **data)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_seed(cls) -> "Genome":
        return cls(prompt_text=SEED_PROMPT)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_genome.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/genome.py swarm_optimizer/tests/test_genome.py
git commit -m "feat: genome con 2 artefactos (prompt + ValidationConfig)"
```

---

## Task 2: Capa de validación determinista (`validation.py`)

**Files:**
- Create: `swarm_optimizer/validation.py`
- Test: `swarm_optimizer/tests/test_validation.py`

Esta es la palanca de precisión a costo cero. `apply_validation` recibe el output parseado y devuelve uno limpio.

- [ ] **Step 1: Write the failing tests**

```python
# swarm_optimizer/tests/test_validation.py
from swarm_optimizer.genome import ValidationConfig
from swarm_optimizer.validation import apply_validation

BODY = "Boric fue criticado por Matthei durante la sesión. Kast respaldó la moción."


def _rel(frm, to, act="attacks", pol="negative", quote="x"):
    return {"from_entity": frm, "to_entity": to, "act_type": act,
            "polarity": pol, "issue": "x", "evidence_quote": quote}


def test_substring_filter_drops_unquoted_relations():
    vc = ValidationConfig(require_evidence_substring=True, min_quote_len=0,
                          normalize_passive_direction=False)
    parsed = {"entities": [], "relations": [
        _rel("Matthei", "Boric", quote="criticado por Matthei"),   # substring real
        _rel("Kast", "Boric", quote="frase inventada que no existe"),  # no substring
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1
    assert out["relations"][0]["from_entity"] == "Matthei"


def test_min_quote_len_drops_short_quotes():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=10,
                          normalize_passive_direction=False)
    parsed = {"entities": [], "relations": [_rel("A", "B", quote="corta")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"] == []


def test_passive_direction_swaps_from_to():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=True)
    # El modelo puso la dirección al revés: patient antes de 'por', agent después
    parsed = {"entities": [], "relations": [
        _rel("Boric", "Matthei", quote="Boric fue criticado por Matthei")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["from_entity"] == "Matthei"
    assert out["relations"][0]["to_entity"] == "Boric"


def test_dedup_collapses_identical_triples():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False, dedup=True)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", quote="q1"),
        _rel("A", "B", "attacks", quote="q2"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1


def test_allowed_act_types_filters():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False,
                          allowed_act_types=["endorses"])
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", quote="q"),
        _rel("A", "B", "endorses", quote="q"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1
    assert out["relations"][0]["act_type"] == "endorses"


def test_max_relations_caps_output():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False, max_relations_per_article=1)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", quote="q"),
        _rel("C", "D", "endorses", quote="q"),
    ]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert len(out["relations"]) == 1


def test_polarity_consistency_corrects_polarity():
    vc = ValidationConfig(require_evidence_substring=False, min_quote_len=0,
                          normalize_passive_direction=False,
                          enforce_polarity_consistency=True)
    parsed = {"entities": [], "relations": [
        _rel("A", "B", "attacks", pol="positive", quote="q")]}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["relations"][0]["polarity"] == "negative"


def test_entities_passthrough_untouched():
    vc = ValidationConfig()
    parsed = {"entities": [{"name": "Boric", "type": "roster_actor"}], "relations": []}
    out = apply_validation(parsed, BODY, {}, vc)
    assert out["entities"] == [{"name": "Boric", "type": "roster_actor"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_validation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_optimizer.validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# swarm_optimizer/validation.py
"""
Artefacto B: capa de validación determinista (post-proceso puro, costo $0).
Limpia el output del LLM antes de scoring. Sube precisión sin gastar tokens.
"""
from __future__ import annotations

from swarm_optimizer.genome import ValidationConfig
from swarm_optimizer.rubric import normalize

# act_type -> polaridad esperada (coherencia)
_POLARITY_MAP = {
    "attacks": "negative",
    "accuses": "negative",
    "questions": "negative",
    "competes_with": "negative",
    "distances_from": "negative",
    "endorses": "positive",
    "allies_with": "positive",
    "calls_on": "neutral",
    "negotiates_with": "neutral",
    "co_occurs": "neutral",
}


def _maybe_swap_direction(rel: dict) -> dict:
    """En pasiva 'X fue criticado por Y', el agente (true from) va DESPUÉS de 'por'.
    Si el modelo puso from=X (antes de 'por') y to=Y (después), corrige el swap."""
    quote = normalize(rel.get("evidence_quote", ""))
    if " por " not in quote:
        return rel
    f = normalize(rel.get("from_entity", ""))
    t = normalize(rel.get("to_entity", ""))
    if not f or not t:
        return rel
    idx = quote.find(" por ")
    before, after = quote[:idx], quote[idx + 5:]
    if f in before and t in after:
        return {**rel, "from_entity": rel["to_entity"], "to_entity": rel["from_entity"]}
    return rel


def apply_validation(parsed: dict, body: str, union: dict, vc: ValidationConfig) -> dict:
    """parsed = {'entities': [...], 'relations': [...]} -> versión limpia."""
    body_norm = normalize(body)
    relations = list(parsed.get("relations", []))
    cleaned: list[dict] = []
    seen: set[tuple] = set()

    for rel in relations:
        quote = rel.get("evidence_quote", "") or ""

        if vc.min_quote_len and len(quote.strip()) < vc.min_quote_len:
            continue

        if vc.require_evidence_substring and normalize(quote) not in body_norm:
            continue

        if vc.allowed_act_types is not None and rel.get("act_type") not in vc.allowed_act_types:
            continue

        if vc.normalize_passive_direction:
            rel = _maybe_swap_direction(rel)

        if vc.enforce_polarity_consistency:
            expected = _POLARITY_MAP.get(rel.get("act_type"))
            if expected is not None:
                rel = {**rel, "polarity": expected}

        if vc.dedup:
            key = (
                normalize(rel.get("from_entity", "")),
                normalize(rel.get("to_entity", "")),
                rel.get("act_type"),
            )
            if key in seen:
                continue
            seen.add(key)

        cleaned.append(rel)

    if vc.max_relations_per_article is not None:
        cleaned = cleaned[: vc.max_relations_per_article]

    return {"entities": parsed.get("entities", []), "relations": cleaned}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_validation.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/validation.py swarm_optimizer/tests/test_validation.py
git commit -m "feat: capa de validación determinista (artefacto B)"
```

---

## Task 3: Fitness F0.5 + piso de recall + costo (`fitness.py`)

**Files:**
- Create: `swarm_optimizer/fitness.py`
- Test: `swarm_optimizer/tests/test_fitness.py`

- [ ] **Step 1: Write the failing tests**

```python
# swarm_optimizer/tests/test_fitness.py
from swarm_optimizer.fitness import f_beta, fitness, RECALL_FLOOR


def test_f_beta_weights_precision_over_recall():
    # F0.5: con precisión alta y recall bajo, F0.5 > F1
    p, r = 0.8, 0.2
    f05 = f_beta(p, r, beta=0.5)
    f1 = f_beta(p, r, beta=1.0)
    assert f05 > f1


def test_f_beta_zero_when_both_zero():
    assert f_beta(0.0, 0.0, 0.5) == 0.0


def _metrics(prec_rel, rec_rel, prec_ent=0.8, rec_ent=0.8, pol=0.8, act=0.7):
    return {
        "Precision_rel": prec_rel, "Recall_rel": rec_rel,
        "Precision_ent": prec_ent, "Recall_ent": rec_ent,
        "Polarity_acc": pol, "Act_acc": act,
    }


def test_recall_floor_disqualifies():
    below = fitness(_metrics(0.95, RECALL_FLOOR - 0.01), tokens_per_article=1000)
    above = fitness(_metrics(0.95, RECALL_FLOOR + 0.10), tokens_per_article=1000)
    assert below < 0          # penalización fuerte
    assert above > below


def test_cost_penalty_prefers_cheaper():
    cheap = fitness(_metrics(0.7, 0.5), tokens_per_article=500)
    pricey = fitness(_metrics(0.7, 0.5), tokens_per_article=8000)
    assert cheap > pricey


def test_higher_precision_scores_higher():
    lo = fitness(_metrics(0.4, 0.5), tokens_per_article=1000)
    hi = fitness(_metrics(0.9, 0.5), tokens_per_article=1000)
    assert hi > lo


def test_pro_penalized_more_than_flash_same_tokens():
    flash = fitness(_metrics(0.7, 0.5), tokens_per_article=3000, model="gemini-2.5-flash")
    pro = fitness(_metrics(0.7, 0.5), tokens_per_article=3000, model="gemini-2.5-pro")
    assert flash > pro      # mismo conteo de tokens, pero Pro cuesta más -> menor fitness
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_fitness.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_optimizer.fitness'`

- [ ] **Step 3: Write minimal implementation**

```python
# swarm_optimizer/fitness.py
"""Fitness precision-weighted (F0.5) + piso de recall + penalización de costo."""
from __future__ import annotations

DEFAULT_FITNESS_WEIGHTS = {"rel": 0.45, "ent": 0.25, "pol": 0.15, "act": 0.15}
RECALL_FLOOR = 0.15          # recall_rel mínimo para ser campeón
COST_LAMBDA = 0.10           # peso de la penalización de costo
COST_REF_TOKENS = 4000       # tokens/artículo de referencia (modelo barato) para normalizar
COST_CAP = 2.0               # tope del costo normalizado (deja gradiente por encima de Flash)
DISQUALIFY_PENALTY = 1.0     # se resta si recall_rel < piso

# Multiplicador de precio relativo POR TOKEN respecto al modelo barato (Flash = 1.0).
# Hace que el cost penalty mida COSTO (precio), no solo cantidad de tokens. Aprox., tunable.
PRICE_MULT = {
    "gemini-2.5-flash": 1.0,
    "gemini-2.5-pro": 16.0,
}
DEFAULT_PRICE_MULT = 1.0


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def fitness(
    metrics: dict,
    tokens_per_article: float,
    model: str = "gemini-2.5-flash",
    weights: dict | None = None,
    recall_floor: float = RECALL_FLOOR,
    cost_lambda: float = COST_LAMBDA,
) -> float:
    w = weights or DEFAULT_FITNESS_WEIGHTS
    f05_rel = f_beta(metrics.get("Precision_rel", 0.0), metrics.get("Recall_rel", 0.0), 0.5)
    f05_ent = f_beta(metrics.get("Precision_ent", 0.0), metrics.get("Recall_ent", 0.0), 0.5)

    quality = (
        w["rel"] * f05_rel
        + w["ent"] * f05_ent
        + w["pol"] * metrics.get("Polarity_acc", 0.0)
        + w["act"] * metrics.get("Act_acc", 0.0)
    )

    # Costo = tokens × precio relativo del modelo (price-aware), normalizado y con tope.
    price = PRICE_MULT.get(model, DEFAULT_PRICE_MULT)
    cost_norm = min(tokens_per_article * price / COST_REF_TOKENS, COST_CAP)
    score = quality - cost_lambda * cost_norm

    if metrics.get("Recall_rel", 0.0) < recall_floor:
        score -= DISQUALIFY_PENALTY

    return score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_fitness.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/fitness.py swarm_optimizer/tests/test_fitness.py
git commit -m "feat: fitness F0.5 + piso de recall + penalización de costo"
```

---

## Task 4: ELO + selección de padres (`elo.py`)

**Files:**
- Create: `swarm_optimizer/elo.py`
- Test: `swarm_optimizer/tests/test_elo.py`

- [ ] **Step 1: Write the failing tests**

```python
# swarm_optimizer/tests/test_elo.py
import numpy as np
from swarm_optimizer.elo import expected_score, update_pairwise, sample_parent


def test_expected_score_symmetric():
    assert abs(expected_score(1000, 1000) - 0.5) < 1e-9


def test_winner_gains_loser_loses():
    new_a, new_b = update_pairwise(1000, 1000, score_a=1.0, k=32)
    assert new_a > 1000
    assert new_b < 1000
    assert abs((new_a - 1000) - (1000 - new_b)) < 1e-9   # zero-sum


def test_upset_moves_more_than_expected_win():
    # un under-rated (800) que vence a un favorito (1200) gana más que al revés
    underdog_gain = update_pairwise(800, 1200, 1.0)[0] - 800
    favorite_gain = update_pairwise(1200, 800, 1.0)[0] - 1200
    assert underdog_gain > favorite_gain


def test_sample_parent_prefers_high_elo_low_children():
    rng = np.random.default_rng(0)
    entries = [
        {"elo": 1300, "children": 0},   # fuerte, inexplorado -> favorito
        {"elo": 900, "children": 10},   # débil, sobre-explorado
    ]
    picks = [sample_parent(entries, rng) for _ in range(400)]
    assert picks.count(0) > picks.count(1)


def test_sample_parent_returns_valid_index():
    rng = np.random.default_rng(1)
    entries = [{"elo": 1000, "children": 0}]
    assert sample_parent(entries, rng) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_elo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_optimizer.elo'`

- [ ] **Step 3: Write minimal implementation**

```python
# swarm_optimizer/elo.py
"""Ratings ELO para selección evolutiva + muestreo de padres (DGM)."""
from __future__ import annotations

import math

K_DEFAULT = 32
ELO_BASE = 1000.0
SIGMOID_SCALE = 200.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_pairwise(rating_a: float, rating_b: float, score_a: float, k: float = K_DEFAULT):
    """score_a in {1.0 win, 0.5 tie, 0.0 loss}. Devuelve (new_a, new_b)."""
    ea = expected_score(rating_a, rating_b)
    delta = k * (score_a - ea)
    return rating_a + delta, rating_b - delta


def _weight(entry: dict) -> float:
    elo = entry.get("elo", ELO_BASE)
    children = entry.get("children", 0)
    pref = 1.0 / (1.0 + math.exp(-(elo - ELO_BASE) / SIGMOID_SCALE))  # sigmoid(ELO)
    return pref / (1.0 + children)                                    # 1/(1+#hijos)


def sample_parent(entries: list[dict], rng) -> int:
    """Muestrea índice de padre ∝ sigmoid(ELO) y ∝ 1/(1+#hijos). rng = np Generator."""
    weights = [_weight(e) for e in entries]
    total = sum(weights)
    if total <= 0:
        return int(rng.integers(len(entries)))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(entries) - 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_elo.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/elo.py swarm_optimizer/tests/test_elo.py
git commit -m "feat: ELO + selección de padres (sigmoid ELO / 1+hijos)"
```

---

## Task 5: Archivo open-ended (`archive.py` reescrito)

**Files:**
- Modify (rewrite): `swarm_optimizer/archive.py`
- Modify (rewrite): `swarm_optimizer/tests/test_archive.py`

El `Archive` actual (historial + `best()`) se reemplaza por un archivo open-ended con ELO y linaje. **Los tests antiguos de archive se reescriben** (usaban `Config`); ahora usan `Genome`.

- [ ] **Step 1: Write the failing tests (reemplazar el archivo completo)**

```python
# swarm_optimizer/tests/test_archive.py
import tempfile
from pathlib import Path

import numpy as np

from swarm_optimizer.archive import Archive
from swarm_optimizer.genome import Genome


def _g(p="v"):
    return Genome.from_seed() if p == "seed" else Genome(prompt_text=p)


def test_add_returns_incrementing_ids_and_counts_children():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        root = arc.add(_g("root"), mutation_type="seed")
        child = arc.add(_g("child"), parent_id=root, mutation_type="diff_a",
                        artifact_touched="A")
        assert root == 0 and child == 1
        assert arc.get(root).children == 1
        assert arc.get(child).parent_id == root
        assert arc.get(child).artifact_touched == "A"


def test_record_elo_and_championship():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        i = arc.add(_g("a"))
        arc.record_elo(i, 1240.0)
        arc.record_championship(i, score=0.6, metrics={"Recall_rel": 0.4})
        assert arc.get(i).elo == 1240.0
        assert arc.get(i).championship_score == 0.6


def test_champion_picks_best_championship_score():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        a = arc.add(_g("a")); b = arc.add(_g("b"))
        arc.record_championship(a, 0.5, {})
        arc.record_championship(b, 0.7, {})
        assert arc.champion().id == b


def test_champion_none_when_no_championship_run():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        arc.add(_g("a"))
        assert arc.champion() is None


def test_top_by_elo_orders_desc():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        a = arc.add(_g("a")); b = arc.add(_g("b")); c = arc.add(_g("c"))
        arc.record_elo(a, 1100); arc.record_elo(b, 1300); arc.record_elo(c, 900)
        top = arc.top_by_elo(2)
        assert [e.id for e in top] == [b, a]


def test_select_parent_returns_existing_id():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        arc.add(_g("a")); arc.add(_g("b"))
        rng = np.random.default_rng(0)
        pid = arc.select_parent(rng)
        assert pid in (0, 1)


def test_persists_and_reloads():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "arch.jsonl"
        arc = Archive(path)
        i = arc.add(_g("persisted"))
        arc.record_elo(i, 1234.0)
        reloaded = Archive(path)
        assert reloaded.get(i).genome.prompt_text == "persisted"
        assert reloaded.get(i).elo == 1234.0


def test_total_tokens_sums_championship_metrics():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "arch.jsonl")
        a = arc.add(_g("a"))
        arc.record_championship(a, 0.5, {"tokens": 1000})
        assert arc.total_tokens() == 1000


def test_diagnosis_and_delta_persist():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "arch.jsonl"
        arc = Archive(path)
        i = arc.add(_g("a"), diagnosis="faltan relaciones")
        arc.record_delta(i, 0.12)
        reloaded = Archive(path)
        assert reloaded.get(i).diagnosis == "faltan relaciones"
        assert reloaded.get(i).fitness_delta == 0.12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_archive.py -q`
Expected: FAIL (nuevos métodos `add`/`get`/`record_elo`/… no existen)

- [ ] **Step 3: Write minimal implementation (reescribir el archivo completo)**

```python
# swarm_optimizer/archive.py
"""Archivo evolutivo open-ended (DGM): conserva todos los genomas con ELO + linaje."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from swarm_optimizer.elo import ELO_BASE, sample_parent
from swarm_optimizer.genome import Genome


@dataclass
class ArchiveEntry:
    id: int
    genome: Genome
    elo: float = ELO_BASE
    children: int = 0
    parent_id: int | None = None
    mutation_type: str | None = None        # "seed"|"diff_a"|"diff_b"|"cross"
    artifact_touched: str | None = None      # "A"|"B"|None
    championship_score: float | None = None
    metrics: dict = field(default_factory=dict)
    diagnosis: str | None = None          # diagnóstico que originó esta mutación (memoria v2)
    fitness_delta: float | None = None    # margen vs campeón en el skirmish (memoria v2)


class Archive:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[int, ArchiveEntry] = {}
        self._next_id = 0
        if self.path.exists():
            self._reload()

    # ── construcción ──────────────────────────────────────────────── #
    def add(self, genome: Genome, parent_id: int | None = None,
            mutation_type: str = "seed", artifact_touched: str | None = None,
            elo: float = ELO_BASE, diagnosis: str | None = None) -> int:
        eid = self._next_id
        self._next_id += 1
        entry = ArchiveEntry(
            id=eid, genome=genome, elo=elo, parent_id=parent_id,
            mutation_type=mutation_type, artifact_touched=artifact_touched,
            diagnosis=diagnosis,
        )
        self._entries[eid] = entry
        if parent_id is not None and parent_id in self._entries:
            self._entries[parent_id].children += 1
        self._persist(entry)
        return eid

    # ── updates ───────────────────────────────────────────────────── #
    def record_elo(self, eid: int, new_elo: float) -> None:
        self._entries[eid].elo = new_elo
        self._persist(self._entries[eid])

    def record_championship(self, eid: int, score: float, metrics: dict) -> None:
        e = self._entries[eid]
        e.championship_score = score
        e.metrics = metrics
        self._persist(e)

    def record_delta(self, eid: int, fitness_delta: float) -> None:
        self._entries[eid].fitness_delta = fitness_delta
        self._persist(self._entries[eid])

    # ── consultas ─────────────────────────────────────────────────── #
    def get(self, eid: int) -> ArchiveEntry:
        return self._entries[eid]

    def all(self) -> list[ArchiveEntry]:
        return list(self._entries.values())

    def select_parent(self, rng) -> int:
        entries = self.all()
        idx = sample_parent(
            [{"elo": e.elo, "children": e.children} for e in entries], rng
        )
        return entries[idx].id

    def top_by_elo(self, n: int) -> list[ArchiveEntry]:
        return sorted(self.all(), key=lambda e: e.elo, reverse=True)[:n]

    def champion(self) -> ArchiveEntry | None:
        scored = [e for e in self.all() if e.championship_score is not None]
        if not scored:
            return None
        return max(scored, key=lambda e: e.championship_score)

    def total_tokens(self) -> int:
        return sum(int(e.metrics.get("tokens", 0)) for e in self.all())

    # ── persistencia ──────────────────────────────────────────────── #
    def _persist(self, entry: ArchiveEntry) -> None:
        rec = {
            "id": entry.id,
            "genome": entry.genome.to_dict(),
            "elo": entry.elo,
            "children": entry.children,
            "parent_id": entry.parent_id,
            "mutation_type": entry.mutation_type,
            "artifact_touched": entry.artifact_touched,
            "championship_score": entry.championship_score,
            "metrics": entry.metrics,
            "diagnosis": entry.diagnosis,
            "fitness_delta": entry.fitness_delta,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _reload(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            entry = ArchiveEntry(
                id=rec["id"],
                genome=Genome.from_dict(rec["genome"]),
                elo=rec["elo"],
                children=rec["children"],
                parent_id=rec["parent_id"],
                mutation_type=rec["mutation_type"],
                artifact_touched=rec["artifact_touched"],
                championship_score=rec["championship_score"],
                metrics=rec.get("metrics", {}),
                diagnosis=rec.get("diagnosis"),
                fitness_delta=rec.get("fitness_delta"),
            )
            self._entries[entry.id] = entry      # última línea gana (updates)
            self._next_id = max(self._next_id, entry.id + 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_archive.py -q`
Expected: PASS (9 passed)

Nota: `_reload` reconstruye el estado tomando la última línea por id (los updates de ELO/championship se persisten como líneas nuevas). `children` se persiste en cada `_persist`, por lo que el reload toma el conteo final.

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/archive.py swarm_optimizer/tests/test_archive.py
git commit -m "feat: archivo open-ended con ELO + linaje (reemplaza greedy)"
```

---

## Task 6: Operadores de mutación (`mutate.py`)

**Files:**
- Create: `swarm_optimizer/mutate.py`
- Test: `swarm_optimizer/tests/test_mutate.py`

Separa lo puro (aplicar diffs, parsear, aplicar patch a ValidationConfig) de lo que llama al LLM (`diagnose`/`propose`/`cross_pollinate`). El LLM se mockea con un cliente falso.

- [ ] **Step 1: Write the failing tests**

```python
# swarm_optimizer/tests/test_mutate.py
from swarm_optimizer.genome import Genome, ValidationConfig
from swarm_optimizer.mutate import (
    apply_diff, parse_search_replace, apply_validation_patch, propose,
)


class FakeResp:
    def __init__(self, text): self.text = text


class FakeModels:
    def __init__(self, text): self._text = text
    def generate_content(self, model, contents): return FakeResp(self._text)


class FakeClient:
    def __init__(self, text): self.models = FakeModels(text)


def test_apply_diff_replaces_once():
    out, ok = apply_diff("hola mundo mundo", "mundo", "tierra")
    assert ok is True
    assert out == "hola tierra mundo"


def test_apply_diff_noop_when_search_absent():
    out, ok = apply_diff("abc", "zzz", "x")
    assert ok is False
    assert out == "abc"


def test_apply_diff_tolerates_whitespace_differences():
    # el LLM re-indentó: search con espacios simples, texto con múltiples
    out, ok = apply_diff("Regla    X   aquí", "Regla X aquí", "Regla Y")
    assert ok is True
    assert out == "Regla Y"


def test_parse_search_replace_extracts_blocks():
    text = (
        "blah\n<<<<<<< SEARCH\nfoo bar\n=======\nfoo BAZ\n>>>>>>> REPLACE\ntrailing"
    )
    res = parse_search_replace(text)
    assert res == ("foo bar", "foo BAZ")


def test_parse_search_replace_none_when_malformed():
    assert parse_search_replace("no markers here") is None


def test_apply_validation_patch_updates_fields():
    vc = ValidationConfig(min_quote_len=8)
    patched = apply_validation_patch(vc, {"min_quote_len": 15, "dedup": False})
    assert patched.min_quote_len == 15
    assert patched.dedup is False
    assert patched.require_evidence_substring == vc.require_evidence_substring


def test_apply_validation_patch_ignores_unknown_keys():
    vc = ValidationConfig()
    patched = apply_validation_patch(vc, {"bogus_key": 1, "min_quote_len": 3})
    assert patched.min_quote_len == 3
    assert not hasattr(patched, "bogus_key")


def test_propose_artifact_a_applies_diff_to_prompt():
    g = Genome(prompt_text="Eres un extractor. Regla X.")
    client = FakeClient(
        '{"artifact": "A", "diff": '
        '"<<<<<<< SEARCH\\nRegla X.\\n=======\\nRegla X mejorada.\\n>>>>>>> REPLACE"}'
    )
    child, mtype, touched = propose(g, "diagnóstico", client)
    assert "Regla X mejorada." in child.prompt_text
    assert mtype == "diff_a" and touched == "A"


def test_propose_artifact_b_patches_validation():
    g = Genome(prompt_text="p")
    client = FakeClient('{"artifact": "B", "patch": {"min_quote_len": 20}}')
    child, mtype, touched = propose(g, "diagnóstico", client)
    assert child.validation.min_quote_len == 20
    assert mtype == "diff_b" and touched == "B"


def test_propose_invalid_json_returns_noop_clone():
    g = Genome(prompt_text="p")
    client = FakeClient("esto no es json")
    child, mtype, touched = propose(g, "diag", client)
    assert child.prompt_text == "p"
    assert mtype == "noop" and touched is None


def test_propose_diff_that_does_not_apply_is_noop():
    g = Genome(prompt_text="contenido real")
    client = FakeClient(
        '{"artifact": "A", "diff": '
        '"<<<<<<< SEARCH\\nNO EXISTE\\n=======\\nx\\n>>>>>>> REPLACE"}'
    )
    child, mtype, touched = propose(g, "diag", client)
    assert child.prompt_text == "contenido real"
    assert mtype == "noop"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_mutate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_optimizer.mutate'`

- [ ] **Step 3: Write minimal implementation**

```python
# swarm_optimizer/mutate.py
"""El 'Evolution AI': diagnóstico + mutación por diff/patch + cross-pollination.

Funciones puras (apply_diff, parse_search_replace, apply_validation_patch) testeables
sin API; las que llaman a Gemini reciben un `client` inyectable.
"""
from __future__ import annotations

import dataclasses
import json
import re
from copy import deepcopy

from swarm_optimizer.genome import Genome, ValidationConfig

_SR_RE = re.compile(
    r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)

_DIAGNOSE_PROMPT = """\
Eres un diagnosticador de errores de extracción de relaciones políticas.

Falsos positivos (relaciones emitidas que NO están en el gold):
{fps}

Falsos negativos (relaciones en el gold que NO fueron emitidas):
{fns}

Identifica 3-5 causas concretas y agrupadas de error.
Responde SOLO con una lista numerada de causas, sin introducción ni cierre.
"""

_PROPOSE_PROMPT = """\
Eres un optimizador de un extractor de relaciones políticas. El extractor tiene dos
artefactos:
- Artefacto A: el PROMPT de extracción (texto).
- Artefacto B: un config de validación determinista con estos campos:
  require_evidence_substring(bool), min_quote_len(int), normalize_passive_direction(bool),
  dedup(bool), enforce_polarity_consistency(bool), allowed_act_types(list|null),
  max_relations_per_article(int|null).

Diagnóstico de errores:
{diagnosis}

PROMPT actual (artefacto A):
{prompt}

ValidationConfig actual (artefacto B):
{validation}

Propone UN SOLO cambio pequeño que ataque la causa más importante. Elige UN artefacto.
- Si eliges A: devuelve un diff en formato SEARCH/REPLACE.
- Si eliges B: devuelve un patch con los campos a cambiar.

Responde ÚNICAMENTE con JSON, sin markdown:
- Para A: {{"artifact": "A", "diff": "<<<<<<< SEARCH\\n...\\n=======\\n...\\n>>>>>>> REPLACE"}}
- Para B: {{"artifact": "B", "patch": {{"campo": valor}}}}
"""

_CROSS_PROMPT = """\
Eres un optimizador. Tienes dos extractores top que destacan por motivos distintos.
Combina sus fortalezas en un PROMPT nuevo (artefacto A) que tome lo mejor de ambos.

PROMPT del padre 1:
{p1}

PROMPT del padre 2:
{p2}

Devuelve ÚNICAMENTE el nuevo prompt completo, sin markdown ni explicaciones.
"""


# ── puros ─────────────────────────────────────────────────────────── #
def apply_diff(text: str, search: str, replace: str) -> tuple[str, bool]:
    # 1. exact match (rápido y determinista)
    if search in text:
        return text.replace(search, replace, 1), True
    # 2. tolerante a whitespace: tokens de search separados por \s+. Atrapa la
    #    re-indentación del LLM SIN el riesgo de corrupción semántica de un fuzzy
    #    por similitud (un noop es más seguro que un mal-apply plausible).
    tokens = search.split()
    if not tokens:
        return text, False
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in tokens))
    m = pattern.search(text)
    if m:
        return text[: m.start()] + replace + text[m.end():], True
    return text, False


def parse_search_replace(text: str) -> tuple[str, str] | None:
    m = _SR_RE.search(text or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def apply_validation_patch(vc: ValidationConfig, patch: dict) -> ValidationConfig:
    valid = {f.name for f in dataclasses.fields(ValidationConfig)}
    updates = {k: v for k, v in patch.items() if k in valid}
    return dataclasses.replace(vc, **updates)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── con LLM (client inyectable) ───────────────────────────────────── #
def diagnose(fps: list[str], fns: list[str], client) -> str:
    prompt = _DIAGNOSE_PROMPT.format(
        fps="\n".join(fps[:20]) or "(ninguno)",
        fns="\n".join(fns[:20]) or "(ninguno)",
    )
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return resp.text or ""
    except Exception:
        return "(diagnóstico no disponible)"


def propose(genome: Genome, diagnosis: str, client) -> tuple[Genome, str, str | None]:
    """Devuelve (hijo, mutation_type, artifact_touched).
    mutation_type ∈ {'diff_a','diff_b','noop'}."""
    prompt = _PROPOSE_PROMPT.format(
        diagnosis=diagnosis,
        prompt=genome.prompt_text,
        validation=json.dumps(dataclasses.asdict(genome.validation), ensure_ascii=False),
    )
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        data = json.loads(_strip_fences(resp.text or ""))
    except Exception:
        return deepcopy(genome), "noop", None

    child = deepcopy(genome)
    artifact = data.get("artifact")

    if artifact == "A":
        sr = parse_search_replace(data.get("diff", ""))
        if not sr:
            return child, "noop", None
        new_text, ok = apply_diff(child.prompt_text, sr[0], sr[1])
        if not ok:
            return child, "noop", None
        child.prompt_text = new_text
        return child, "diff_a", "A"

    if artifact == "B":
        patch = data.get("patch", {})
        if not isinstance(patch, dict) or not patch:
            return child, "noop", None
        child.validation = apply_validation_patch(child.validation, patch)
        return child, "diff_b", "B"

    return child, "noop", None


def cross_pollinate(parent1: Genome, parent2: Genome, client) -> tuple[Genome, str, str]:
    """Combina los prompts de dos padres top en un hijo (artefacto A)."""
    prompt = _CROSS_PROMPT.format(p1=parent1.prompt_text, p2=parent2.prompt_text)
    child = deepcopy(parent1)
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        new_prompt = _strip_fences(resp.text or "")
        if new_prompt:
            child.prompt_text = new_prompt
    except Exception:
        pass
    return child, "cross", "A"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_mutate.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/mutate.py swarm_optimizer/tests/test_mutate.py
git commit -m "feat: mutación por diff SEARCH/REPLACE + patch B + cross-pollination"
```

---

## Task 7: Submuestreo por iteración (`splits.py`)

**Files:**
- Modify: `swarm_optimizer/splits.py` (añadir función)
- Modify: `swarm_optimizer/tests/test_splits.py` (añadir tests)

- [ ] **Step 1: Write the failing tests (añadir al final del archivo de tests)**

```python
# añadir a swarm_optimizer/tests/test_splits.py
from swarm_optimizer.splits import subsample


def test_subsample_size_and_membership():
    pool = [f"a{i}" for i in range(20)]
    s = subsample(pool, k=5, seed=0)
    assert len(s) == 5
    assert all(x in pool for x in s)
    assert len(set(s)) == 5


def test_subsample_deterministic_with_seed():
    pool = [f"a{i}" for i in range(20)]
    assert subsample(pool, 5, seed=42) == subsample(pool, 5, seed=42)


def test_subsample_varies_with_seed():
    pool = [f"a{i}" for i in range(20)]
    assert subsample(pool, 5, seed=1) != subsample(pool, 5, seed=2)


def test_subsample_caps_at_pool_size():
    pool = ["a", "b", "c"]
    assert sorted(subsample(pool, 10, seed=0)) == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_splits.py -q`
Expected: FAIL with `ImportError: cannot import name 'subsample'`

- [ ] **Step 3: Write minimal implementation (añadir a `splits.py`)**

```python
# añadir a swarm_optimizer/splits.py (al final, deja los imports np existentes)

def subsample(pool: list[str], k: int, seed: int) -> list[str]:
    """Submuestreo aleatorio sin reemplazo de k artículos del pool (RoboPhD undersampling).
    Determinista dado el seed. Si k >= len(pool), devuelve el pool completo (orden barajado)."""
    rng = np.random.default_rng(seed)
    arr = np.array(pool, dtype=object)
    rng.shuffle(arr)
    n = min(k, len(pool))
    return arr[:n].tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_splits.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/splits.py swarm_optimizer/tests/test_splits.py
git commit -m "feat: submuestreo de artículos por iteración (undersampling)"
```

---

## Task 8: Integrar validación + verificación en `extractor.py`

**Files:**
- Modify: `swarm_optimizer/extractor.py`
- Create: `swarm_optimizer/tests/test_extractor.py`

`build_prompt`/`run_extraction` ya funcionan con cualquier objeto que tenga `prompt_text`/`few_shots`/`architecture`/`model` — `Genome` los tiene. Añadimos: (a) aplicar `validation.apply_validation` tras parsear; (b) pasada de verificación agéntica opcional gateada por `genome.verify`.

- [ ] **Step 1: Write the failing tests**

```python
# swarm_optimizer/tests/test_extractor.py
from swarm_optimizer.genome import Genome, ValidationConfig
from swarm_optimizer.extractor import extract_article, verify_relations

BODY = "Boric fue criticado por Matthei. Kast respaldó la moción del gobierno."


class FakeResp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = type("U", (), {"prompt_token_count": 10,
                                             "candidates_token_count": 5})()


class FakeModels:
    def __init__(self, texts): self._texts = list(texts); self._i = 0
    def generate_content(self, model, contents):
        t = self._texts[min(self._i, len(self._texts) - 1)]; self._i += 1
        return FakeResp(t)


class FakeClient:
    def __init__(self, *texts): self.models = FakeModels(texts)


def test_extract_article_applies_validation_substring_filter():
    g = Genome(prompt_text="p",
               validation=ValidationConfig(require_evidence_substring=True,
                                           min_quote_len=0,
                                           normalize_passive_direction=False))
    out = (
        '{"entities": [], "relations": ['
        '{"from_entity":"Matthei","to_entity":"Boric","act_type":"attacks",'
        '"polarity":"negative","issue":"x","evidence_quote":"criticado por Matthei"},'
        '{"from_entity":"X","to_entity":"Y","act_type":"attacks",'
        '"polarity":"negative","issue":"x","evidence_quote":"cita inexistente zzz"}]}'
    )
    client = FakeClient(out)
    res = extract_article("a1", BODY, {}, g, [], client)
    assert len(res["relations"]) == 1
    assert res["relations"][0]["from_entity"] == "Matthei"


def test_verify_relations_drops_unsupported():
    # la verificación devuelve solo la relación soportada
    verified_json = (
        '{"relations": [{"from_entity":"Kast","to_entity":"gobierno",'
        '"act_type":"endorses","polarity":"positive","issue":"x",'
        '"evidence_quote":"Kast respaldó la moción"}]}'
    )
    client = FakeClient(verified_json)
    rels = [
        {"from_entity": "Kast", "to_entity": "gobierno", "act_type": "endorses",
         "polarity": "positive", "issue": "x", "evidence_quote": "Kast respaldó la moción"},
        {"from_entity": "A", "to_entity": "B", "act_type": "attacks",
         "polarity": "negative", "issue": "x", "evidence_quote": "no soportada"},
    ]
    out = verify_relations(rels, BODY, "gemini-2.5-flash", client)
    assert len(out) == 1
    assert out[0]["from_entity"] == "Kast"


def test_verify_relations_failsafe_on_bad_output():
    client = FakeClient("no es json")
    rels = [{"from_entity": "A", "to_entity": "B", "act_type": "attacks",
             "polarity": "negative", "issue": "x", "evidence_quote": "q"}]
    out = verify_relations(rels, BODY, "gemini-2.5-flash", client)
    assert out == rels       # fail-safe: conserva las originales


def test_verify_flag_off_skips_verification():
    g = Genome(prompt_text="p", verify=False,
               validation=ValidationConfig(require_evidence_substring=False,
                                           min_quote_len=0,
                                           normalize_passive_direction=False))
    out = ('{"entities": [], "relations": [{"from_entity":"A","to_entity":"B",'
           '"act_type":"attacks","polarity":"negative","issue":"x","evidence_quote":"q"}]}')
    client = FakeClient(out)   # un solo texto -> si verificara, fallaría con otro formato
    res = extract_article("a1", BODY, {}, g, [], client)
    assert len(res["relations"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_extractor.py -q`
Expected: FAIL with `ImportError: cannot import name 'verify_relations'`

- [ ] **Step 3: Modify `extractor.py`**

Añadir el import al inicio (junto a los otros imports):

```python
from swarm_optimizer.validation import apply_validation
```

Añadir esta función (antes de `extract_article`):

```python
_VERIFY_PROMPT = """\
Revisa estas relaciones extraídas de un artículo. Elimina las que NO estén soportadas
literalmente por el texto y corrige dirección/polaridad cuando el texto lo indique.

ARTÍCULO:
{body}

RELACIONES (JSON):
{relations}

Devuelve ÚNICAMENTE el JSON corregido: {{"relations": [...]}}, sin markdown.
"""


def verify_relations(relations: list[dict], body: str, model: str, client,
                     temperatures=(0.0, 0.2)) -> list[dict]:
    """Verificación agéntica (RoboPhD). Fail-safe: ante error conserva las originales."""
    if not relations:
        return relations
    prompt = _VERIFY_PROMPT.format(
        body=body[:4000],
        relations=json.dumps(relations, ensure_ascii=False),
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        parsed = parse_llm_output(response.text or "")
        verified = parsed.get("relations", [])
        return verified if verified else relations
    except Exception:
        return relations
```

Modificar `extract_article`: tras `parsed = parse_llm_output(text)` y antes de construir el dict de retorno, insertar el bloque de validación + verificación. Reemplazar el cuerpo del `try` de `extract_article` para que quede así:

```python
    prompt = build_prompt(config, body, union, few_shot_examples)

    for attempt in range(retry):
        try:
            response = client.models.generate_content(
                model=config.model,
                contents=prompt,
            )
            text = response.text or ""
            parsed = parse_llm_output(text)

            # Verificación agéntica opcional (gateada por flag del genoma)
            if getattr(config, "verify", False):
                parsed["relations"] = verify_relations(
                    parsed.get("relations", []), body, config.model, client
                )

            # Validación determinista (artefacto B)
            vc = getattr(config, "validation", None)
            if vc is not None:
                parsed = apply_validation(parsed, body, union, vc)

            usage = getattr(response, "usage_metadata", None)
            token_count = 0
            if usage:
                token_count = (
                    getattr(usage, "prompt_token_count", 0)
                    + getattr(usage, "candidates_token_count", 0)
                )
            return {
                "article_id": article_id,
                "entities": parsed["entities"],
                "relations": parsed["relations"],
                "tokens": token_count,
            }
        except Exception as e:
            import sys
            print(f"[extractor] attempt {attempt+1}/{retry} failed: {e}", file=sys.stderr)
            if attempt == retry - 1:
                return {"article_id": article_id, "entities": [], "relations": [], "tokens": 0}
            time.sleep(2 ** attempt)

    return {"article_id": article_id, "entities": [], "relations": [], "tokens": 0}
```

(El resto de `extractor.py` —`build_prompt`, `parse_llm_output`, `run_extraction`, `_load_few_shot_examples`— no cambia. `run_extraction` sigue recibiendo un objeto con `.model/.few_shots/.architecture`, que `Genome` cumple.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_extractor.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/extractor.py swarm_optimizer/tests/test_extractor.py
git commit -m "feat: extractor aplica validación B + verificación agéntica opcional"
```

---

## Task 9: Reescribir el loop evolutivo (`loop.py`)

**Files:**
- Modify (rewrite): `swarm_optimizer/loop.py`
- Modify (rewrite): `swarm_optimizer/tests/test_loop.py`

El loop orquesta: seleccionar padre → diagnosticar → mutar → skirmish (ELO sobre submuestreo) → cada M, championship sobre `eval` fijo. Para testear sin parquet ni API, `run_loop` acepta dependencias inyectables (`articles_df`, `gold_df`, `splits`, `client`, `extract_fn`).

- [ ] **Step 1: Write the failing tests (reemplazar el archivo completo)**

```python
# swarm_optimizer/tests/test_loop.py
import tempfile
from pathlib import Path

import pandas as pd

from swarm_optimizer.genome import Genome
from swarm_optimizer.loop import skirmish_result, should_stop, run_loop
from swarm_optimizer.archive import Archive


def test_skirmish_result_maps_scores_to_winloss():
    # a mejor que b -> 1.0 ; empate -> 0.5 ; a peor -> 0.0
    assert skirmish_result(0.7, 0.5) == 1.0
    assert skirmish_result(0.5, 0.5) == 0.5
    assert skirmish_result(0.3, 0.5) == 0.0


def test_should_stop_on_max_iter():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "a.jsonl")
        assert should_stop(arc, iteration=10, max_iter=10, budget_tokens=10**9) is True


def test_should_stop_on_budget():
    with tempfile.TemporaryDirectory() as d:
        arc = Archive(Path(d) / "a.jsonl")
        i = arc.add(Genome.from_seed())
        arc.record_championship(i, 0.5, {"tokens": 2_000_000})
        assert should_stop(arc, iteration=1, max_iter=80, budget_tokens=1_000_000) is True


def _fake_extract_fn(article_ids, articles_df, gold_df, union_map, genome):
    """Extractor determinista falso: una relación correcta por artículo si el prompt
    contiene 'BUENO', ninguna si no. Devuelve (predictions, tokens)."""
    preds = []
    good = "BUENO" in genome.prompt_text
    for aid in article_ids:
        rels = [{"from_entity": "A", "to_entity": "B", "act_type": "attacks",
                 "polarity": "negative", "issue": "x", "evidence_quote": "q"}] if good else []
        preds.append({"article_id": aid, "entities": [], "relations": rels, "tokens": 100})
    return preds, 100 * len(article_ids)


def _fake_metrics_fn(preds, article_ids, gold_df, union_map):
    """Métricas falsas: precision/recall altos si hay relaciones, 0 si no."""
    has = any(p["relations"] for p in preds)
    if has:
        return {"Precision_rel": 0.9, "Recall_rel": 0.6, "Precision_ent": 0.8,
                "Recall_ent": 0.8, "Polarity_acc": 0.9, "Act_acc": 0.8}
    return {"Precision_rel": 0.0, "Recall_rel": 0.0, "Precision_ent": 0.0,
            "Recall_ent": 0.0, "Polarity_acc": 0.0, "Act_acc": 0.0}


class FakeResp:
    def __init__(self, text): self.text = text


class FakeModels:
    # diagnose -> texto; propose -> diff que inserta 'BUENO' en el prompt
    def generate_content(self, model, contents):
        if "diagnosticador" in contents:
            return FakeResp("1. faltan relaciones")
        return FakeResp(
            '{"artifact": "A", "diff": '
            '"<<<<<<< SEARCH\\nSEED\\n=======\\nSEED BUENO\\n>>>>>>> REPLACE"}'
        )


class FakeClient:
    def __init__(self): self.models = FakeModels()


def test_run_loop_improves_and_crowns_champion():
    with tempfile.TemporaryDirectory() as d:
        articles_df = pd.DataFrame({"article_id": ["x1", "x2", "x3", "x4"],
                                    "body": ["b"] * 4})
        gold_df = pd.DataFrame({"article_id": ["x1"]})
        splits = {"eval": ["x1", "x2", "x3"], "test": ["x4"]}
        seed = Genome(prompt_text="SEED")    # malo (sin 'BUENO')

        champ = run_loop(
            max_iter=4,
            budget_usd=999.0,
            subsample_k=2,
            championship_every=2,
            cross_every=99,
            archive_path=Path(d) / "arch.jsonl",
            best_path=Path(d) / "best.json",
            seed_genome=seed,
            articles_df=articles_df,
            gold_df=gold_df,
            splits=splits,
            client=FakeClient(),
            extract_fn=_fake_extract_fn,
            metrics_fn=_fake_metrics_fn,
            verbose=False,
        )
        # tras mutar, el hijo lleva 'BUENO' y debería ganar el championship
        assert "BUENO" in champ.prompt_text
        assert (Path(d) / "best.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest swarm_optimizer/tests/test_loop.py -q`
Expected: FAIL (`skirmish_result`/nueva `run_loop` no existen)

- [ ] **Step 3: Write minimal implementation (reescribir el archivo completo)**

```python
# swarm_optimizer/loop.py
"""Loop evolutivo: archivo open-ended + ELO (skirmish) + championship anclado."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai

from swarm_optimizer.archive import Archive
from swarm_optimizer.elo import update_pairwise
from swarm_optimizer.extractor import run_extraction
from swarm_optimizer.fitness import fitness
from swarm_optimizer.genome import Genome
from swarm_optimizer.mutate import cross_pollinate, diagnose, propose
from swarm_optimizer.rubric import compute_metrics, load_union_map
from swarm_optimizer.splits import load_splits, subsample

GOLD_ARTICLES = Path(__file__).parent.parent.parent / "gold_standard_v5/data/pilot_gold_articles.parquet"
GOLD_PARQUET = Path(__file__).parent.parent.parent / "gold_standard_v5/data/pilot_gold_final.parquet"


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
    """FP/FN legibles para el diagnosticador (reusa match_entity)."""
    from swarm_optimizer.rubric import match_entity
    fps, fns = [], []
    pred_map = {p["article_id"]: p for p in predictions}
    for art_id, pred in pred_map.items():
        union = union_map.get(art_id, {})
        gold_rels = gold_df[gold_df["article_id"] == art_id]
        gold_pairs = {(r.u_from, r.u_to) for r in gold_rels.itertuples()} \
            if "u_from" in gold_df.columns else set()
        matched = set()
        for rel in pred.get("relations", []):
            fu, _ = match_entity(rel["from_entity"], union)
            tu, _ = match_entity(rel["to_entity"], union)
            pair = (fu, tu) if fu and tu else None
            if pair and pair in gold_pairs:
                matched.add(pair)
            elif fu and tu:
                fps.append(f"{rel['from_entity']} -{rel['act_type']}-> {rel['to_entity']} ({art_id[:8]})")
        for pair in gold_pairs - matched:
            fns.append(f"{pair[0]} -> {pair[1]} ({art_id[:8]})")
    return fps, fns


def _evaluate(genome, ids, articles_df, gold_df, union_map, extract_fn, metrics_fn):
    preds, tokens = extract_fn(ids, articles_df, gold_df, union_map, genome)
    metrics = metrics_fn(preds, ids, gold_df, union_map)
    tokens_per_article = tokens / max(len(ids), 1)
    return preds, metrics, fitness(metrics, tokens_per_article, model=genome.model), tokens


def run_loop(
    max_iter: int = 80,
    budget_usd: float = 8.0,
    subsample_k: int = 12,
    championship_every: int = 5,
    cross_every: int = 7,
    archive_path: Path | None = None,
    best_path: Path | None = None,
    seed_genome: Genome | None = None,
    articles_df: pd.DataFrame | None = None,
    gold_df: pd.DataFrame | None = None,
    splits: dict | None = None,
    client=None,
    extract_fn=run_extraction,
    metrics_fn=compute_metrics,
    verbose: bool = True,
) -> Genome:
    archive_path = archive_path or (Path(__file__).parent.parent / "results/swarm/history.jsonl")
    best_path = best_path or (Path(__file__).parent.parent / "results/swarm/best_config.json")

    if articles_df is None:
        articles_df = pd.read_parquet(GOLD_ARTICLES)
    if gold_df is None:
        gold_df = pd.read_parquet(GOLD_PARQUET)
    if splits is None:
        splits = load_splits()
    if client is None:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

    eval_ids, test_ids = splits["eval"], splits["test"]
    union_map = load_union_map(eval_ids + test_ids) if extract_fn is run_extraction else {}
    budget_tokens = int(budget_usd / 0.15 * 1_000_000)
    rng = np.random.default_rng(42)

    archive = Archive(archive_path)
    seed = seed_genome or Genome.from_seed()
    champion_id = archive.add(seed, mutation_type="seed")
    # championship inicial para anclar al campeón
    _, m0, s0, t0 = _evaluate(archive.get(champion_id).genome, eval_ids,
                              articles_df, gold_df, union_map, extract_fn, metrics_fn)
    archive.record_championship(champion_id, s0, {**m0, "tokens": t0})

    for iteration in range(max_iter):
        if should_stop(archive, iteration, max_iter, budget_tokens):
            break

        # 1. seleccionar padre
        parent_id = archive.select_parent(rng)
        parent = archive.get(parent_id).genome

        # 2. mutar (cross-pollination cada N; si no, diff guiado por diagnóstico)
        if iteration > 0 and iteration % cross_every == 0 and len(archive.all()) >= 2:
            top2 = archive.top_by_elo(2)
            child, mtype, touched = cross_pollinate(top2[0].genome, top2[1].genome, client)
            diagnosis = None
        else:
            sub = subsample(eval_ids, subsample_k, seed=1000 + iteration)
            preds, _, _, _ = _evaluate(parent, sub, articles_df, gold_df,
                                       union_map, extract_fn, metrics_fn)
            fps, fns = _format_errors(preds, gold_df, union_map)
            diagnosis = diagnose(fps, fns, client)
            child, mtype, touched = propose(parent, diagnosis, client)

        if mtype == "noop":
            continue
        child_id = archive.add(child, parent_id=parent_id, mutation_type=mtype,
                               artifact_touched=touched, diagnosis=diagnosis)

        # 3. skirmish sobre submuestreo fresco (child vs campeón actual)
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

        if verbose:
            print(f"[iter {iteration}] {mtype} child={child_id} "
                  f"s_child={s_child:.3f} s_champ={s_champ:.3f} res={res}")

        # 4. championship cada M: re-evalúa top-T sobre eval fijo, corona
        if iteration > 0 and iteration % championship_every == 0:
            for entry in archive.top_by_elo(3):
                _, m, s, t = _evaluate(entry.genome, eval_ids, articles_df, gold_df,
                                       union_map, extract_fn, metrics_fn)
                archive.record_championship(entry.id, s, {**m, "tokens": t})
                # Goodhart: chequeo contra test
                _, mt, st, _ = _evaluate(entry.genome, test_ids, articles_df, gold_df,
                                         union_map, extract_fn, metrics_fn)
                if s - st > 0.10 and verbose:
                    print(f"  [GOODHART] entry={entry.id} eval={s:.3f} test={st:.3f}",
                          file=sys.stderr)
            champ = archive.champion()
            if champ:
                champion_id = champ.id

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest swarm_optimizer/tests/test_loop.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/loop.py swarm_optimizer/tests/test_loop.py
git commit -m "feat: loop evolutivo con archivo open-ended + ELO + championship"
```

---

## Task 10: Actualizar entry point y verificación final (`run.py`)

**Files:**
- Modify: `swarm_optimizer/run.py`

- [ ] **Step 1: Read the current `run.py`**

Run: `cat swarm_optimizer/run.py` para ver los argumentos actuales que pasa a `run_loop`.

- [ ] **Step 2: Update `run.py` to match the new `run_loop` signature**

Reemplazar la llamada a `run_loop(...)` para que use los nuevos parámetros. El cuerpo esperado:

```python
# swarm_optimizer/run.py
from __future__ import annotations

from swarm_optimizer.loop import run_loop


def main() -> None:
    run_loop(
        max_iter=80,
        budget_usd=8.0,
        subsample_k=12,
        championship_every=5,
        cross_every=7,
        verbose=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest swarm_optimizer/tests/ -q`
Expected: PASS (todos los módulos verdes, incluido `test_rubric.py` sin tocar)

- [ ] **Step 4: Verify imports resolve and run.py is importable (sin API key real)**

Run: `python -c "import swarm_optimizer.run; import swarm_optimizer.loop; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/run.py
git commit -m "chore: run.py usa el nuevo run_loop evolutivo"
```

---

## Verificación final (manual, opcional, con GEMINI_API_KEY)

Smoke real antes de una corrida larga (cuesta unos centavos):

```bash
# Requiere gold_standard_v5/data/ y GEMINI_API_KEY
python -c "from swarm_optimizer.loop import run_loop; run_loop(max_iter=3, budget_usd=0.50, subsample_k=6, championship_every=2, verbose=True)"
```

Esperado: el loop corre 3 iteraciones, el archivo crece (`results/swarm/history.jsonl`),
y se escribe `results/swarm/best_config.json` con un genoma campeón. Revisar que
`Precision_rel`/`Precision_ent` del campeón ≥ baseline (0.20 / 0.81) en el championship.

---

## Notas de ejecución

- **Orden:** las tareas son secuenciales por dependencias (genome → validation/fitness/elo → archive → mutate → splits → extractor → loop → run). No reordenar.
- **DRY/YAGNI:** MAP-Elites, Retrospective Memory, MCGS y co-evolución sin labels NO se implementan aquí (puertas de upgrade en el spec §8).
- **TDD:** cada tarea es test-first. No implementar sin ver el test fallar primero.
- **Regresión:** `test_rubric.py` y `test_config.py` no se tocan y deben quedar verdes.
- **Desviación del spec (plateau-stop):** el spec §5 listaba "plateau" como guarda de parada (heredada del sistema greedy). En el nuevo diseño open-ended (DGM) se **omite intencionalmente**: DGM tolera mesetas y "dips" temporales porque habilitan breakthroughs futuros vía stepping stones. Las guardas duras son `max_iter` y `budget` (en `should_stop`). Si en la práctica se quiere cortar por estancamiento, agregar después un check sobre la trayectoria de `championship_score` del campeón (no incluido aquí — YAGNI).
- **Enmiendas v1.1 (post-review):** (1) `fitness` es **price-aware** — el costo se pondera por `PRICE_MULT[model]` (Flash=1.0), de modo que "menor costo" mide precio y no solo tokens; habilita usar Pro a futuro de forma segura. (2) `apply_diff` hace exact-match **tolerante a whitespace** (no fuzzy por similitud: un `noop` es más seguro que un mal-apply plausible). (3) El archivo persiste `diagnosis` + `fitness_delta` por entrada para habilitar una Retrospective Memory v2 sin costo extra ahora.
