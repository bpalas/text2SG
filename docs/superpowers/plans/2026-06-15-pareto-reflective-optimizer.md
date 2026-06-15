# Optimizer Pareto-reflexivo (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar la selección campeón-único/ELO por un frente de Pareto sobre (precisión, recall) por tier de modelo, alimentado por reflexión Opus sobre trazas de error, sobre el harness sintético confiable.

**Architecture:** Tres unidades nuevas: `pareto.py` (archivo de Pareto + dominancia, sin ELO), `diagnostics.py` (FN/FP por tipo para la reflexión), y `synth_evolve.py` (CLI glue: `add`/`pick`/`frontier`). El operador reflexivo es un dispatch de agente Opus orquestado por el controller (runbook), no código Python. Evaluador = `do_score`; executor = `synth_run_model` (API) o swarm cero-API.

**Tech Stack:** Python 3, dataclasses, pytest, pandas; reutiliza `rubric.match_entity/canonicalize_union/compute_metrics`, `synth_agent_eval.do_score`.

---

## File Structure
- **Create** `swarm_optimizer/pareto.py` — `ParetoEntry`, `_dominates`, `ParetoArchive`. Responsabilidad única: dominancia + frente + persistencia.
- **Create** `swarm_optimizer/diagnostics.py` — `diagnostics(...)`: extrae FN/FP por tipo.
- **Create** `scripts/synth_evolve.py` — CLI `add`/`pick`/`frontier` (glue fino sobre lo anterior + do_score).
- **Create** `swarm_optimizer/tests/test_pareto.py` — tests de pareto + diagnostics.
- **Generado en runtime:** `results/synthetic/pareto/<model>.json` (archivo del frente por tier).

---

### Task 1: `pareto.py` — archivo de Pareto (sin ELO)

**Files:**
- Create: `swarm_optimizer/pareto.py`
- Test: `swarm_optimizer/tests/test_pareto.py`

- [ ] **Step 1: Write the failing test**

Crear `swarm_optimizer/tests/test_pareto.py`:

```python
from swarm_optimizer.pareto import ParetoArchive, _dominates, ParetoEntry


def _e(P, R, exp=0, _id=0):
    return ParetoEntry(id=_id, genome={}, P=P, R=R, expansions=exp)


def test_dominates_strict_and_ties():
    assert _dominates(_e(0.9, 0.9), _e(0.8, 0.8))      # mejor en ambos
    assert _dominates(_e(0.9, 0.8), _e(0.8, 0.8))      # mejor en uno, igual en otro
    assert not _dominates(_e(0.8, 0.8), _e(0.8, 0.8))  # empate exacto: no domina
    assert not _dominates(_e(0.9, 0.7), _e(0.8, 0.8))  # tradeoff: ninguno domina


def test_add_and_frontier_excludes_dominated():
    a = ParetoArchive()
    a.add({}, 0.93, 0.90)   # balanceado
    a.add({}, 0.94, 0.83)   # max P
    a.add({}, 0.91, 0.89)   # dominado por el balanceado
    front = {(round(e.P, 2), round(e.R, 2)) for e in a.frontier()}
    assert (0.93, 0.90) in front and (0.94, 0.83) in front
    assert (0.91, 0.89) not in front     # dominado, fuera del frente
    assert len(a.all()) == 3             # pero sigue en el archivo


def test_pick_to_expand_least_expanded_then_lowest_id():
    a = ParetoArchive()
    e0 = a.add({}, 0.93, 0.90)   # id 0, expansions 0
    e1 = a.add({}, 0.94, 0.83)   # id 1, expansions 0
    a.mark_expanded(e0.id)       # e0 ahora expansions 1
    assert a.pick_to_expand().id == e1.id   # el menos expandido
    a.mark_expanded(e1.id)
    assert a.pick_to_expand().id == e0.id    # empate 1-1 -> menor id


def test_json_roundtrip():
    a = ParetoArchive()
    a.add({"prompt_text": "x"}, 0.9, 0.8, parent_id=None)
    a.add({"prompt_text": "y"}, 0.95, 0.7, parent_id=0)
    b = ParetoArchive.from_json(a.to_json())
    assert len(b.all()) == 2
    assert b.add({}, 0.5, 0.5).id == 2     # next_id se preserva
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_pareto.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swarm_optimizer.pareto'`.

- [ ] **Step 3: Create `swarm_optimizer/pareto.py`**

```python
"""Archivo de Pareto sobre (precisión, recall) — reemplaza el campeón-único/ELO.
Un archivo por tier de modelo. Determinista; sin dependencias del ELO."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class ParetoEntry:
    id: int
    genome: dict
    P: float
    R: float
    parent_id: int | None = None
    expansions: int = 0
    preds_path: str | None = None


def _dominates(a: ParetoEntry, b: ParetoEntry) -> bool:
    """a domina b si a >= b en P y R, y es estrictamente mejor en al menos uno."""
    return a.P >= b.P and a.R >= b.R and (a.P > b.P or a.R > b.R)


class ParetoArchive:
    def __init__(self, entries: list[ParetoEntry] | None = None, next_id: int = 0):
        self._entries: list[ParetoEntry] = list(entries or [])
        self._next_id = next_id

    def add(self, genome: dict, P: float, R: float, parent_id: int | None = None,
            preds_path: str | None = None) -> ParetoEntry:
        e = ParetoEntry(id=self._next_id, genome=genome, P=P, R=R,
                        parent_id=parent_id, preds_path=preds_path)
        self._next_id += 1
        self._entries.append(e)
        return e

    def all(self) -> list[ParetoEntry]:
        return list(self._entries)

    def frontier(self) -> list[ParetoEntry]:
        return [e for e in self._entries
                if not any(_dominates(o, e) for o in self._entries if o.id != e.id)]

    def pick_to_expand(self) -> ParetoEntry | None:
        front = self.frontier()
        if not front:
            return None
        return min(front, key=lambda e: (e.expansions, e.id))   # menos expandido, desempate por id

    def mark_expanded(self, entry_id: int) -> None:
        for e in self._entries:
            if e.id == entry_id:
                e.expansions += 1
                return

    def to_json(self) -> str:
        return json.dumps({"next_id": self._next_id,
                           "entries": [asdict(e) for e in self._entries]},
                          ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "ParetoArchive":
        d = json.loads(s)
        entries = [ParetoEntry(**e) for e in d.get("entries", [])]
        return cls(entries=entries, next_id=d.get("next_id", len(entries)))

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ParetoArchive":
        p = Path(path)
        return cls.from_json(p.read_text(encoding="utf-8")) if p.exists() else cls()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_pareto.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/pareto.py swarm_optimizer/tests/test_pareto.py
git commit -m "feat: pareto.py — archivo de Pareto (P,R) por tier, sin ELO"
```

---

### Task 2: `diagnostics.py` — trazas de error para la reflexión

**Files:**
- Create: `swarm_optimizer/diagnostics.py`
- Test: `swarm_optimizer/tests/test_pareto.py` (append)

- [ ] **Step 1: Write the failing test**

APPEND a `swarm_optimizer/tests/test_pareto.py`:

```python
import pandas as pd
from swarm_optimizer.diagnostics import diagnostics


def test_diagnostics_fn_and_fp_distractor():
    articles = pd.DataFrame([
        {"article_id": "A1", "es_distractor": False, "dureza": "oblicua"},
        {"article_id": "D1", "es_distractor": True, "dureza": "mixta"},
    ])
    gold = pd.DataFrame([
        {"article_id": "A1", "u_from": "U1", "u_to": "U2", "act_type": "endorses",
         "evidence_quote": "valoró el compromiso de B"},
    ])
    unions = {
        "A1": {"U1": {"union_id": "U1", "type": "roster_actor",
                      "canonical_names": ["Actor A"], "surfaces": ["A"]},
               "U2": {"union_id": "U2", "type": "roster_actor",
                      "canonical_names": ["Actor B"], "surfaces": ["B"]}},
        "D1": {"U1": {"union_id": "U1", "type": "roster_actor",
                      "canonical_names": ["Actor C"], "surfaces": ["C"]},
               "U2": {"union_id": "U2", "type": "roster_actor",
                      "canonical_names": ["Actor D"], "surfaces": ["D"]}},
    }
    preds = [
        {"article_id": "A1", "entities": [], "relations": []},   # se pierde la gold -> FN
        {"article_id": "D1", "entities": [], "relations": [      # inventa en distractor -> FP
            {"from_entity": "Actor C", "to_entity": "Actor D", "act_type": "endorses",
             "evidence_quote": "x"}]},
    ]
    d = diagnostics(["A1", "D1"], preds, gold, unions, articles)
    assert len(d["fn"]) == 1 and d["fn"][0]["dureza"] == "oblicua"
    assert d["fn_by_dureza"].get("oblicua") == 1
    assert d["fp_distractor"] == 1 and d["fp_total"] >= 1
    assert d["fp"][0]["es_distractor"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_pareto.py::test_diagnostics_fn_and_fp_distractor -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swarm_optimizer.diagnostics'`.

- [ ] **Step 3: Create `swarm_optimizer/diagnostics.py`**

```python
"""Trazas de error para el operador reflexivo: qué relaciones reales se pierden (FN, el hueco de
recall) y qué se inventa (FP, disciplina), por tipo de texto. Conteos de FP son aproximados (no
deduplican pares como compute_metrics); sirven como feed cualitativo para la reflexión, no como
métrica oficial."""
from __future__ import annotations

from swarm_optimizer.rubric import match_entity, canonicalize_union


def _pred_pairs(relations, union):
    pairs = set()
    for rel in relations:
        f, fnil = match_entity(str(rel.get("from_entity") or ""), union)
        t, tnil = match_entity(str(rel.get("to_entity") or ""), union)
        if f and t and not fnil and not tnil:
            pairs.add((f, t))
    return pairs


def diagnostics(ids, preds, gold_df, union_map, articles_df, max_examples: int = 8) -> dict:
    pm = {p["article_id"]: p for p in preds}
    meta = {r.article_id: r for r in articles_df.itertuples()}
    fn, fp = [], []
    fn_by_dureza: dict[str, int] = {}
    fp_total = fp_distractor = 0
    for art_id in ids:
        row = meta.get(art_id)
        dureza = str(getattr(row, "dureza", "") or "") if row is not None else ""
        is_dis = bool(getattr(row, "es_distractor", False)) if row is not None else False
        union, alias = canonicalize_union(union_map.get(art_id, {}))
        grows = gold_df[gold_df["article_id"] == art_id]
        gold_pairs = {(alias.get(r.u_from, r.u_from), alias.get(r.u_to, r.u_to)): r
                      for r in grows.itertuples()}
        rels = pm.get(art_id, {}).get("relations", [])
        ppairs = _pred_pairs(rels, union)
        for pair, r in gold_pairs.items():
            if pair not in ppairs:
                fn_by_dureza[dureza] = fn_by_dureza.get(dureza, 0) + 1
                if len(fn) < max_examples:
                    fn.append({"article_id": art_id, "dureza": dureza,
                               "act_type": r.act_type,
                               "quote": (r.evidence_quote or "")[:160]})
        for rel in rels:
            f, fnil = match_entity(str(rel.get("from_entity") or ""), union)
            t, tnil = match_entity(str(rel.get("to_entity") or ""), union)
            if not (f and t and not fnil and not tnil and (f, t) in gold_pairs):
                fp_total += 1
                if is_dis:
                    fp_distractor += 1
                if len(fp) < max_examples:
                    fp.append({"article_id": art_id, "es_distractor": is_dis,
                               "from_entity": rel.get("from_entity"),
                               "to_entity": rel.get("to_entity"),
                               "act_type": rel.get("act_type"),
                               "quote": (rel.get("evidence_quote") or "")[:120]})
    return {"fn": fn, "fp": fp, "fn_by_dureza": fn_by_dureza,
            "fp_total": fp_total, "fp_distractor": fp_distractor}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_pareto.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/diagnostics.py swarm_optimizer/tests/test_pareto.py
git commit -m "feat: diagnostics.py — FN/FP por tipo para el operador reflexivo"
```

---

### Task 3: `synth_evolve.py` — CLI del loop (add / pick / frontier)

**Files:**
- Create: `scripts/synth_evolve.py`

**Nota:** glue fino sobre pareto/diagnostics/do_score. El operador reflexivo (Opus) y el executor
son orquestados por el controller (runbook). Se valida por smoke en Task 4, no unit test.

- [ ] **Step 1: Create `scripts/synth_evolve.py`**

```python
"""CLI del loop Pareto-reflexivo. Las partes deterministas; la reflexión (Opus) y la extracción
las orquesta el controller.

  add      --model M --genome G.json --preds P.json --dataset v1 --split train [--parent ID]
           -> puntúa P.json con do_score, agrega al frente results/synthetic/pareto/<M>.json
  pick     --model M --dataset v1 --split train
           -> elige el miembro del frente menos expandido; vuelca su genoma + sus diagnósticos
              (FN/FP) a archivos para que el agente Opus proponga la mutación; marca expansión
  frontier --model M
           -> imprime el frente actual (id, P, R) en JSON
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from swarm_optimizer.genome import Genome
from swarm_optimizer.pareto import ParetoArchive
from swarm_optimizer.diagnostics import diagnostics
from swarm_optimizer.synth_data import (load_synth_articles, load_synth_gold,
                                        load_synth_unions, load_synth_split)
from synth_agent_eval import do_score, _load_preds   # noqa: E402

PARETO_DIR = Path("results/synthetic/pareto")


def _arch_path(model: str) -> Path:
    return PARETO_DIR / f"{model.replace('.', '_').replace('/', '_')}.json"


def _ids(dataset: str, split: str | None, arts):
    return load_synth_split(dataset)[split] if split else list(dict.fromkeys(arts["article_id"]))


def cmd_add(args) -> None:
    arts = load_synth_articles(args.dataset)
    gold = load_synth_gold(args.dataset)
    unions = load_synth_unions(args.dataset)
    ids = _ids(args.dataset, args.split, arts)
    genome = Genome.from_json(Path(args.genome).read_text(encoding="utf-8"))
    preds = _load_preds(Path(args.preds))
    m, _ = do_score(genome, ids, arts, gold, unions, preds)
    arch = ParetoArchive.load(_arch_path(args.model))
    e = arch.add(genome.to_dict(), round(m["Precision_rel"], 4), round(m["Recall_rel"], 4),
                 parent_id=args.parent, preds_path=str(args.preds))
    arch.save(_arch_path(args.model))
    on_front = any(x.id == e.id for x in arch.frontier())
    print(json.dumps({"id": e.id, "P": e.P, "R": e.R, "on_frontier": on_front,
                      "frontier_size": len(arch.frontier())}, ensure_ascii=False))


def cmd_pick(args) -> None:
    arts = load_synth_articles(args.dataset)
    gold = load_synth_gold(args.dataset)
    unions = load_synth_unions(args.dataset)
    ids = _ids(args.dataset, args.split, arts)
    arch = ParetoArchive.load(_arch_path(args.model))
    e = arch.pick_to_expand()
    if e is None:
        print("frente vacío: agregá baselines con 'add' primero"); return
    out = _arch_path(args.model).parent
    (out / f"{args.model}_expand_genome.json").write_text(
        json.dumps(e.genome, ensure_ascii=False, indent=1), encoding="utf-8")
    diag = {"fn_by_dureza": {}, "fn": [], "fp": [], "fp_total": 0, "fp_distractor": 0}
    if e.preds_path and Path(e.preds_path).exists():
        diag = diagnostics(ids, _load_preds(Path(e.preds_path)), gold, unions, arts)
    (out / f"{args.model}_expand_diag.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8")
    arch.mark_expanded(e.id)
    arch.save(_arch_path(args.model))
    print(json.dumps({"picked_id": e.id, "P": e.P, "R": e.R,
                      "genome": str(out / f"{args.model}_expand_genome.json"),
                      "diag": str(out / f"{args.model}_expand_diag.json"),
                      "fn_by_dureza": diag["fn_by_dureza"], "fp_distractor": diag["fp_distractor"]},
                     ensure_ascii=False))


def cmd_frontier(args) -> None:
    arch = ParetoArchive.load(_arch_path(args.model))
    front = sorted(arch.frontier(), key=lambda e: e.R)
    print(json.dumps([{"id": e.id, "P": e.P, "R": e.R, "expansions": e.expansions} for e in front],
                     ensure_ascii=False, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("add")
    pa.add_argument("--model", required=True); pa.add_argument("--genome", required=True)
    pa.add_argument("--preds", required=True); pa.add_argument("--dataset", default="v1")
    pa.add_argument("--split", default=None, choices=["train", "test"])
    pa.add_argument("--parent", type=int, default=None); pa.set_defaults(func=cmd_add)
    pp = sub.add_parser("pick")
    pp.add_argument("--model", required=True); pp.add_argument("--dataset", default="v1")
    pp.add_argument("--split", default=None, choices=["train", "test"]); pp.set_defaults(func=cmd_pick)
    pf = sub.add_parser("frontier")
    pf.add_argument("--model", required=True); pf.set_defaults(func=cmd_frontier)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke (sin API) — sembrar el frente con 2 baselines ya medidos y pickear**

Run:
```bash
python scripts/synth_evolve.py add --model haiku --genome results/synthetic/genomes/seed.json \
  --preds results/synthetic/runs/v1_seed/preds_haiku.json --dataset v1 --split train
python scripts/synth_evolve.py add --model haiku --genome results/synthetic/genomes/analysis.json \
  --preds results/synthetic/runs/v1_analysis/preds_haiku.json --dataset v1 --split train
python scripts/synth_evolve.py frontier --model haiku
python scripts/synth_evolve.py pick --model haiku --dataset v1 --split train
```
Expected: `add` imprime P/R y `on_frontier`; `frontier` lista los no-dominados; `pick` imprime el
miembro elegido + `fn_by_dureza` + escribe `*_expand_genome.json` y `*_expand_diag.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/synth_evolve.py
git commit -m "feat: synth_evolve — CLI add/pick/frontier del loop Pareto-reflexivo"
```

---

### Task 4: Verificación + runbook del operador reflexivo

**Files:**
- Modify: `docs/synthetic-eval-loop.md`

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest swarm_optimizer/tests/ -q`
Expected: PASS — la suite existente (158) + los 5 nuevos de `test_pareto.py`.

- [ ] **Step 2: Confirmar el frente sembrado**

Run: `python scripts/synth_evolve.py frontier --model haiku`
Expected: JSON con los baselines no-dominados (seed/analysis con sus P/R Haiku).

- [ ] **Step 3: Documentar el loop reflexivo en el runbook**

APPEND a `docs/synthetic-eval-loop.md`:

```markdown
## Loop Pareto-reflexivo (synth_evolve)

Selección por frente de Pareto (P,R) por tier, sin ELO. Una ronda:
1. `python scripts/synth_evolve.py pick --model M --dataset v1 --split train`
   → escribe `<M>_expand_genome.json` (miembro a expandir) y `<M>_expand_diag.json` (FN/FP).
2. **Reflexión (Opus):** despachar un subagente Opus con esos dos archivos → propone un genoma
   nuevo (edita prompt A y/o flags de AnalysisConfig C, condicionado al modelo) → guardarlo en
   `cand.json`. (Es el Evolution Agent; orquestado por el controller, no automático.)
3. **Extraer + puntuar:** correr `cand.json` con el executor →
   - API: `python scripts/synth_run_model.py --genome cand.json --model M --dataset v1 --split train`
   - cero-API: `dump → swarm → collect → score` (ver arriba) → `preds.json`.
4. `python scripts/synth_evolve.py add --model M --genome cand.json --preds preds.json \
      --dataset v1 --split train --parent <picked_id>`
   → entra al frente si no es dominado.
5. Repetir 2-3 rondas. `frontier --model M` para ver el estado; regenerar el gráfico de Pareto.
Árbitro: confirmar los mejores del frente en `--split test` y en el test real.
```

- [ ] **Step 4: Commit**

```bash
git add docs/synthetic-eval-loop.md
git commit -m "docs: runbook del loop Pareto-reflexivo (pick -> Opus -> extraer -> add)"
```

---

## Notas
- El operador reflexivo NO es Python (es dispatch de agente). El driver hace lo determinista
  (pick/diag/add); el controller orquesta la reflexión + extracción con checkpoints.
- Costo como tier: un archivo de frente por modelo (`results/synthetic/pareto/<model>.json`).
- B (ValidationConfig) se barre gratis re-puntuando, sin nueva extracción.
</content>
