# Synthetic Subagent Eval + Improve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluar cualquier genoma sobre el oráculo sintético (verdad plantada, 100% control) usando un swarm de subagentes Claude (cero API), y un loop de mejora estilo RoboPhD que itera el genoma para subir el puntaje.

**Architecture:** Patrón `dump → (Workflow de subagentes extrae) → score` reusando el de `agent_eval.py`, re-apuntado a `results/synthetic/<dataset>/`. Las partes deterministas (dump/score, validación B) son Python ($0); la extracción masiva es un Workflow (no-FS, no-Python: prompts entran por `args`, extracciones salen por return, el controller escribe `preds.json`). El loop de rondas lo orquesta el controller.

**Tech Stack:** Python 3, pandas/parquet, pytest, el Workflow tool (JS), subagentes Claude (Agent tool). Reutiliza `build_prompt`, `apply_validation`, `compute_metrics`, `build_analysis`.

---

## File Structure

- **Create** `swarm_optimizer/synth_data.py` — loaders del dataset sintético + split estratificado. Pura, testeable.
- **Create** `scripts/synth_agent_eval.py` — CLI `dump`/`score` sobre el sintético (funciones `do_dump`/`do_score` testeables + glue CLI).
- **Create** `scripts/make_synth_genomes.py` — escribe los 3 genomas baseline.
- **Create** `scripts/synth_extract_workflow.js` — el Workflow de extracción (fan-out por lotes).
- **Create** `swarm_optimizer/tests/test_synth_eval.py` — tests de loaders/split/dump/score.
- **Generado en runtime:** `results/synthetic/v1/split.json`, `results/synthetic/genomes/*.json`, `results/synthetic/runs/<id>/{prompts.json,preds.json}`.

`synth_data.py` vive en el paquete (como `splits.py`) para que los tests lo importen limpio. Los loaders aceptan un `base` opcional para que los tests usen `tmp_path` (hermético).

---

### Task 1: `synth_data.py` — loaders + split estratificado

**Files:**
- Create: `swarm_optimizer/synth_data.py`
- Test: `swarm_optimizer/tests/test_synth_eval.py`

- [ ] **Step 1: Write the failing test**

Crear `swarm_optimizer/tests/test_synth_eval.py`:

```python
import json
import pandas as pd
from pathlib import Path

from swarm_optimizer.synth_data import (
    load_synth_unions, load_synth_split, load_synth_articles, load_synth_gold,
)


def _make_fixture(base: Path) -> Path:
    d = base / "mini"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"article_id": "A1", "title": "t", "body": "Boric criticó a Matthei.",
         "dominio": "politica", "registro": "formal", "es_distractor": False},
        {"article_id": "A2", "title": "t", "body": "Matthei respondió a Boric.",
         "dominio": "politica", "registro": "coloquial", "es_distractor": False},
        {"article_id": "A3", "title": "t", "body": "El club ganó la copa.",
         "dominio": "futbol", "registro": "formal", "es_distractor": True},
        {"article_id": "A4", "title": "t", "body": "Kast habló en el congreso.",
         "dominio": "politica", "registro": "formal", "es_distractor": False},
    ]).to_parquet(d / "articles.parquet", index=False)
    pd.DataFrame([
        {"article_id": "A1", "u_from": "U1", "u_to": "U2", "act_type": "accuses",
         "polarity": "negative", "issue": None, "evidence_quote": "Boric criticó a Matthei"},
    ]).to_parquet(d / "gold.parquet", index=False)
    unions = {"A1": {
        "U1": {"union_id": "U1", "type": "roster_actor",
               "canonical_names": ["Gabriel Boric"], "surfaces": ["Boric"]},
        "U2": {"union_id": "U2", "type": "roster_actor",
               "canonical_names": ["Evelyn Matthei"], "surfaces": ["Matthei"]}}}
    (d / "unions.json").write_text(json.dumps(unions), encoding="utf-8")
    return base


def test_load_synth_unions(tmp_path):
    _make_fixture(tmp_path)
    u = load_synth_unions("mini", base=tmp_path)
    assert u["A1"]["U1"]["canonical_names"] == ["Gabriel Boric"]


def test_load_synth_articles_and_gold(tmp_path):
    _make_fixture(tmp_path)
    assert len(load_synth_articles("mini", base=tmp_path)) == 4
    assert len(load_synth_gold("mini", base=tmp_path)) == 1


def test_load_synth_split_deterministic_and_stratified(tmp_path):
    _make_fixture(tmp_path)
    s1 = load_synth_split("mini", seed=42, base=tmp_path)
    s2 = load_synth_split("mini", seed=42, base=tmp_path)   # persistido → idéntico
    assert s1 == s2
    assert set(s1["train"]).isdisjoint(s1["test"])
    assert set(s1["train"]) | set(s1["test"]) == {"A1", "A2", "A3", "A4"}
    assert len(s1["test"]) >= 3   # 3 estratos, cada uno aporta ≥1 a test
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_synth_eval.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swarm_optimizer.synth_data'`.

- [ ] **Step 3: Create `swarm_optimizer/synth_data.py`**

```python
"""Carga de datasets sintéticos (verdad plantada) y split estratificado.
Espejo de splits.py pero para results/synthetic/<dataset>/.
Los loaders aceptan `base` para tests herméticos (tmp_path)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).parent.parent / "results/synthetic"


def _dir(dataset: str, base: Path | None = None) -> Path:
    return (base or SYNTH_ROOT) / dataset


def load_synth_articles(dataset: str = "v1", base: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(_dir(dataset, base) / "articles.parquet")


def load_synth_gold(dataset: str = "v1", base: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(_dir(dataset, base) / "gold.parquet")


def load_synth_unions(dataset: str = "v1", base: Path | None = None) -> dict:
    """{article_id: {uid: {union_id, type, canonical_names, surfaces}}}."""
    return json.loads((_dir(dataset, base) / "unions.json").read_text(encoding="utf-8"))


def load_synth_split(dataset: str = "v1", seed: int = 42, base: Path | None = None) -> dict:
    """Split estratificado por dominio×registro, ~75% train / 25% test, persistido en
    <dataset>/split.json. Cada estrato aporta al menos 1 a test."""
    path = _dir(dataset, base) / "split.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    arts = load_synth_articles(dataset, base)
    rng = np.random.default_rng(seed)
    strat = arts["dominio"].astype(str) + "|" + arts["registro"].astype(str)
    train, test = [], []
    for _, group in arts.groupby(strat, sort=True):
        ids = group["article_id"].tolist()
        arr = np.array(ids, dtype=object)
        rng.shuffle(arr)
        ids = arr.tolist()
        n_test = max(1, round(len(ids) * 0.25))
        test.extend(ids[:n_test])
        train.extend(ids[n_test:])
    split = {"train": sorted(train), "test": sorted(test), "seed": seed}
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_synth_eval.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/synth_data.py swarm_optimizer/tests/test_synth_eval.py
git commit -m "feat: synth_data — loaders del dataset sintético + split estratificado"
```

---

### Task 2: `synth_agent_eval.py` — dump + score

**Files:**
- Create: `scripts/synth_agent_eval.py`
- Test: `swarm_optimizer/tests/test_synth_eval.py`

- [ ] **Step 1: Write the failing test**

APPEND a `swarm_optimizer/tests/test_synth_eval.py`:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from synth_agent_eval import do_dump, do_score   # noqa: E402
from swarm_optimizer.genome import Genome, AnalysisConfig   # noqa: E402


def test_do_dump_includes_analysis_block(tmp_path):
    _make_fixture(tmp_path)
    arts = load_synth_articles("mini", base=tmp_path)
    unions = load_synth_unions("mini", base=tmp_path)
    g = Genome(prompt_text="INSTRUCCIONES", architecture="given_entities",
               analysis=AnalysisConfig())
    rows = do_dump(g, ["A1"], arts, unions)
    assert len(rows) == 1 and rows[0]["article_id"] == "A1"
    assert "=== ANÁLISIS DE ACTORES ===" in rows[0]["prompt"]
    assert "INSTRUCCIONES" in rows[0]["prompt"]


def test_do_dump_appends_verify_when_flagged(tmp_path):
    _make_fixture(tmp_path)
    arts = load_synth_articles("mini", base=tmp_path)
    unions = load_synth_unions("mini", base=tmp_path)
    g = Genome(prompt_text="X", architecture="given_entities", verify=True)
    rows = do_dump(g, ["A1"], arts, unions)
    assert "VERIFICACIÓN" in rows[0]["prompt"]


def test_do_score_perfect_extraction(tmp_path):
    _make_fixture(tmp_path)
    arts = load_synth_articles("mini", base=tmp_path)
    gold = load_synth_gold("mini", base=tmp_path)
    unions = load_synth_unions("mini", base=tmp_path)
    g = Genome(prompt_text="X", architecture="given_entities")
    preds = [{"article_id": "A1", "entities": [], "relations": [
        {"from_entity": "Boric", "to_entity": "Matthei", "act_type": "accuses",
         "polarity": "negative", "issue": "x",
         "evidence_quote": "Boric criticó a Matthei"}]}]
    ids = ["A1", "A2", "A3", "A4"]
    m, predictions = do_score(g, ids, arts, gold, unions, preds)
    assert m["Precision_rel"] == 1.0
    assert m["Recall_rel"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_synth_eval.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'synth_agent_eval'`.

- [ ] **Step 3: Create `scripts/synth_agent_eval.py`**

```python
"""Harness dump/score sobre el oráculo sintético (verdad plantada), para un swarm de
subagentes Claude (cero API). El Workflow externo hace la extracción; este script solo
arma prompts (dump) y puntúa (score).

Flujo:
  1. dump   --genome g.json --dataset v1 [--split train] --out run/r0
            → run/r0/prompts.json = [{"article_id","prompt"}], run/r0/ids.json
  2. (Workflow externo extrae; el controller escribe run/r0/preds.json
      = [{"article_id","entities","relations"}])
  3. score  --genome g.json --dataset v1 [--split train] --preds run/r0/preds.json [--label x]
            → métricas JSON + desglose por dominio + disciplina en distractores
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm_optimizer.genome import Genome
from swarm_optimizer.synth_data import (load_synth_articles, load_synth_gold,
                                        load_synth_unions, load_synth_split)

_SWARM_VERIFY = ("\n\nVERIFICACIÓN (hazla antes de responder): por cada relación candidata, "
                 "confirmá que la evidence_quote es literal del artículo y que sostiene la "
                 "dirección y la polaridad; descartá las que no. Devolvé solo las que sobreviven.")


def do_dump(genome, ids, articles_df, union_map) -> list[dict]:
    from swarm_optimizer.extractor import build_prompt
    meta = articles_df.set_index("article_id")
    rows = []
    for art_id in ids:
        if art_id not in meta.index:
            continue
        body = str(meta.loc[art_id, "body"] or "")
        prompt = build_prompt(genome, body, union_map.get(art_id, {}), [])
        if getattr(genome, "verify", False):
            prompt += _SWARM_VERIFY
        rows.append({"article_id": art_id, "prompt": prompt})
    return rows


def do_score(genome, ids, articles_df, gold_df, union_map, preds_raw):
    from swarm_optimizer.validation import apply_validation
    from swarm_optimizer.rubric import compute_metrics
    from swarm_optimizer.fitness import f_beta
    pm = {p["article_id"]: p for p in preds_raw}
    meta = articles_df.set_index("article_id")
    predictions = []
    for art_id in ids:
        p = pm.get(art_id, {"entities": [], "relations": []})
        body = str(meta.loc[art_id, "body"]) if art_id in meta.index else ""
        validated = apply_validation(
            {"entities": p.get("entities", []), "relations": p.get("relations", [])},
            body, union_map.get(art_id, {}), genome.validation)
        predictions.append({"article_id": art_id, **validated})
    m = compute_metrics(predictions, ids, gold_df, union_map)
    m["f05_rel"] = round(f_beta(m["Precision_rel"], m["Recall_rel"], 0.5), 4)
    return m, predictions


def _ids_for(args, articles_df) -> list[str]:
    if getattr(args, "split", None):
        return load_synth_split(args.dataset)[args.split]
    return list(articles_df["article_id"])


def cmd_dump(args) -> None:
    genome = Genome.from_json(Path(args.genome).read_text(encoding="utf-8"))
    arts = load_synth_articles(args.dataset)
    unions = load_synth_unions(args.dataset)
    ids = _ids_for(args, arts)
    rows = do_dump(genome, ids, arts, unions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "ids.json").write_text(json.dumps([r["article_id"] for r in rows], indent=1), encoding="utf-8")
    print(f"dump OK: {len(rows)} prompts en {out / 'prompts.json'}")


def cmd_score(args) -> None:
    from swarm_optimizer.rubric import compute_metrics
    genome = Genome.from_json(Path(args.genome).read_text(encoding="utf-8"))
    arts = load_synth_articles(args.dataset)
    gold = load_synth_gold(args.dataset)
    unions = load_synth_unions(args.dataset)
    ids = _ids_for(args, arts)
    preds_raw = json.loads(Path(args.preds).read_text(encoding="utf-8"))
    m, predictions = do_score(genome, ids, arts, gold, unions, preds_raw)
    m["label"] = args.label or Path(args.preds).stem
    m["n_articles"] = len(ids)

    meta = arts.set_index("article_id")
    groups: dict[str, list[str]] = {}
    if "dominio" in arts.columns:
        for art_id in ids:
            if art_id in meta.index:
                groups.setdefault(str(meta.loc[art_id, "dominio"]), []).append(art_id)
    by_dom = {}
    for val, sub in groups.items():
        sg = gold[gold["article_id"].isin(sub)]
        sp = [p for p in predictions if p["article_id"] in sub]
        bm = compute_metrics(sp, sub, sg, unions)
        by_dom[val] = {"P": round(bm["Precision_rel"], 3), "R": round(bm["Recall_rel"], 3),
                       "F1": round(bm["F1_rel"], 3), "n": len(sub)}
    m["by_dominio"] = by_dom
    if "es_distractor" in arts.columns:
        dis = {i for i in ids if i in meta.index and bool(meta.loc[i, "es_distractor"])}
        m["distractor_fp"] = sum(len(p["relations"]) for p in predictions if p["article_id"] in dis)
        m["n_distractors"] = len(dis)
    print(json.dumps(m, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_d = sub.add_parser("dump")
    p_d.add_argument("--genome", required=True)
    p_d.add_argument("--dataset", default="v1")
    p_d.add_argument("--split", default=None, choices=["train", "test"])
    p_d.add_argument("--out", required=True)
    p_d.set_defaults(func=cmd_dump)

    p_s = sub.add_parser("score")
    p_s.add_argument("--genome", required=True)
    p_s.add_argument("--dataset", default="v1")
    p_s.add_argument("--split", default=None, choices=["train", "test"])
    p_s.add_argument("--preds", required=True)
    p_s.add_argument("--label", default=None)
    p_s.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_synth_eval.py -q`
Expected: PASS (6 passed total).

- [ ] **Step 5: Commit**

```bash
git add scripts/synth_agent_eval.py swarm_optimizer/tests/test_synth_eval.py
git commit -m "feat: synth_agent_eval — dump (prompts) + score (métricas + desglose) sobre sintético"
```

---

### Task 3: `make_synth_genomes.py` — los 3 genomas baseline

**Files:**
- Create: `scripts/make_synth_genomes.py`

- [ ] **Step 1: Create `scripts/make_synth_genomes.py`**

```python
"""Escribe los 3 genomas baseline para el eval sintético:
  seed / analysis / analysis_verify  →  results/synthetic/genomes/<name>.json
El campo model es irrelevante en el swarm (el extractor es un subagente), pero el
Genome lo requiere; queda como gemini-2.5-flash y el swarm lo ignora.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm_optimizer.config import SEED_PROMPT
from swarm_optimizer.genome import (Genome, ValidationConfig, AnalysisConfig,
                                    SEED_ALLOWED_ACT_TYPES)

ANTI_UID = ("\nUsa el NOMBRE del actor tal como aparece en la lista proporcionada, "
            "nunca su código interno (U1, U2…).")


def _seed(analysis=None, verify=False) -> Genome:
    return Genome(
        prompt_text=SEED_PROMPT + ANTI_UID,
        architecture="given_entities",
        model="gemini-2.5-flash",
        verify=verify,
        validation=ValidationConfig(max_relations_per_article=10,
                                    allowed_act_types=list(SEED_ALLOWED_ACT_TYPES)),
        analysis=analysis,
    )


def main() -> None:
    out = Path(__file__).parent.parent / "results/synthetic/genomes"
    out.mkdir(parents=True, exist_ok=True)
    genomes = {
        "seed": _seed(),
        "analysis": _seed(analysis=AnalysisConfig()),
        "analysis_verify": _seed(analysis=AnalysisConfig(), verify=True),
    }
    for name, g in genomes.items():
        (out / f"{name}.json").write_text(g.to_json(), encoding="utf-8")
        print(f"wrote {out / f'{name}.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/make_synth_genomes.py`
Expected: imprime 3 líneas `wrote .../seed.json`, `analysis.json`, `analysis_verify.json`.

- [ ] **Step 3: Verify the genomes load and carry the analysis config**

Run: `python -c "from swarm_optimizer.genome import Genome; g=Genome.from_json(open('results/synthetic/genomes/analysis.json',encoding='utf-8').read()); print('analysis ok:', g.analysis is not None, '| verify:', g.verify)"`
Expected: `analysis ok: True | verify: False`.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_synth_genomes.py
git commit -m "feat: make_synth_genomes — 3 genomas baseline (seed/analysis/analysis_verify)"
```

---

### Task 4: `synth_extract_workflow.js` — el Workflow de extracción

**Files:**
- Create: `scripts/synth_extract_workflow.js`

**Nota:** los scripts de Workflow no se unit-testean (se validan ejecutándolos en el smoke de la Tarea 5 / runbook). Crear el archivo con exactamente este contenido.

- [ ] **Step 1: Create `scripts/synth_extract_workflow.js`**

```javascript
export const meta = {
  name: 'synth-extract',
  description: 'Extrae relaciones de artículos sintéticos con un swarm de subagentes (cero API)',
  phases: [{ title: 'Extract', detail: 'fan-out por lotes de artículos' }],
}

// args = [{article_id, prompt}, ...]  (de run/<r>/prompts.json)
const items = Array.isArray(args) ? args : []
const BATCH = 15
const batches = []
for (let i = 0; i < items.length; i += BATCH) batches.push(items.slice(i, i + BATCH))

const SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          article_id: { type: 'string' },
          entities: {
            type: 'array',
            items: {
              type: 'object',
              properties: { name: { type: 'string' }, type: { type: 'string' } },
              required: ['name'],
            },
          },
          relations: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                from_entity: { type: 'string' }, to_entity: { type: 'string' },
                act_type: { type: 'string' }, polarity: { type: 'string' },
                issue: { type: 'string' }, evidence_quote: { type: 'string' },
              },
              required: ['from_entity', 'to_entity', 'act_type', 'evidence_quote'],
            },
          },
        },
        required: ['article_id', 'entities', 'relations'],
      },
    },
  },
  required: ['results'],
}

phase('Extract')
log(`Extrayendo ${items.length} artículos en ${batches.length} lotes`)

const out = await parallel(batches.map((batch, bi) => () => {
  const block = batch
    .map(it => `### TAREA ${it.article_id}\n${it.prompt}`)
    .join('\n\n---\n\n')
  const prompt =
    `Sos un extractor de relaciones políticas. Abajo hay ${batch.length} tareas, cada una ` +
    `con su prompt completo (instrucciones + análisis de actores + artículo). Resolvé CADA ` +
    `tarea siguiendo SUS instrucciones al pie de la letra. Devolvé un objeto con "results": ` +
    `una entrada por tarea, con su article_id exacto y las entities/relations que pide ese ` +
    `prompt. No agregues texto fuera del schema.\n\n${block}`
  return agent(prompt, { label: `extract:batch${bi}`, phase: 'Extract', schema: SCHEMA })
    .then(r => (r && Array.isArray(r.results)) ? r.results : [])
    .catch(() => [])
}))

const flat = out.filter(Boolean).flat()
log(`Extracción completa: ${flat.length}/${items.length} artículos`)
return flat
```

- [ ] **Step 2: Commit**

```bash
git add scripts/synth_extract_workflow.js
git commit -m "feat: synth_extract_workflow — Workflow de extracción por swarm (fan-out por lotes)"
```

---

### Task 5: Verificación + smoke en pilot

**Files:** (ninguno nuevo — verificación)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest swarm_optimizer/tests/ -q`
Expected: PASS — la suite existente (146) más los 6 nuevos de `test_synth_eval.py`.

- [ ] **Step 2: Generate baseline genomes**

Run: `python scripts/make_synth_genomes.py`
Expected: 3 genomas escritos.

- [ ] **Step 3: Smoke del dump sobre pilot (20 art, sin gastar quota)**

Run: `python scripts/synth_agent_eval.py dump --genome results/synthetic/genomes/analysis.json --dataset pilot --out results/synthetic/runs/smoke`
Expected: `dump OK: 20 prompts en .../prompts.json`.

- [ ] **Step 4: Verify the dumped prompt carries the analysis block**

Run: `python -c "import json; r=json.load(open('results/synthetic/runs/smoke/prompts.json',encoding='utf-8')); print('analysis block:', '=== ANÁLISIS DE ACTORES ===' in r[0]['prompt'])"`
Expected: `analysis block: True`.

- [ ] **Step 5: Commit (si quedó algo de los archivos del plan sin commitear)**

NUNCA `git add -A`. Verificar solo los archivos del plan:
```bash
git status --short swarm_optimizer/synth_data.py scripts/synth_agent_eval.py scripts/make_synth_genomes.py scripts/synth_extract_workflow.js swarm_optimizer/tests/test_synth_eval.py
```
Expected: sin salida (todo commiteado en tareas 1-4).

---

## Runbook del loop de mejora (ejecutado por el controller con subagentes)

El código de las tareas 1-5 es el harness. El loop de mejora lo orquesta el controller en runtime
(no es un script). Pasos concretos:

### Baseline (las 3 configs, sobre synth-train)

Para cada `cfg` en {seed, analysis, analysis_verify}:
1. `python scripts/synth_agent_eval.py dump --genome results/synthetic/genomes/<cfg>.json --dataset v1 --split train --out results/synthetic/runs/<cfg>_r0`
2. Lanzar el Workflow de extracción: `Workflow({scriptPath: "scripts/synth_extract_workflow.js", args: <contenido de prompts.json>})`. (El controller lee `prompts.json` y lo pasa como `args`.)
3. El controller escribe el resultado del Workflow a `results/synthetic/runs/<cfg>_r0/preds.json` (es exactamente la lista `[{article_id,entities,relations}]` que retorna).
4. `python scripts/synth_agent_eval.py score --genome results/synthetic/genomes/<cfg>.json --dataset v1 --split train --preds results/synthetic/runs/<cfg>_r0/preds.json --label <cfg>_r0`
5. Registrar P_rel/R_rel/F1_rel/f05_rel + by_dominio + distractor_fp.

Elegir el mejor baseline por `f05_rel`.

### Rondas de mejora (RoboPhD)

Por ronda k (2-3 rondas, desde el mejor genoma):
1. **Diagnose** — despachar un subagente con los peores artículos (FP/FN del `preds.json` vs el gold).
   Schema de salida: `{patterns: [{pattern, evidence, fix_hint}]}` (3-5 patrones).
2. **Propose** — despachar un subagente con el genoma actual (JSON) + el diagnóstico.
   Schema: un Genome JSON modificado (`prompt_text` editado y/o `analysis` flags/`role_keywords`
   y/o `validation`). El controller lo guarda en `results/synthetic/genomes/r<k>.json` tras validar
   `Genome.from_json`.
3. **Re-eval** — dump→workflow→score del genoma nuevo sobre synth-train, mismos ids (pareado).
4. **Aceptar/revertir** — aceptar si `f05_rel` sube > 0 (verdad perfecta → el delta es real, sin
   piso de ruido). Si acepta, medir en synth-test (`--split test`).
5. **Barridos de B gratis** — probar N `ValidationConfig` re-corriendo solo `score` (mismo `preds.json`,
   distinto genoma con otra `validation`). Costo $0, sin re-extraer.

### Chequeo final anti-circularidad (opcional, sin API)

Correr el mejor genoma sobre el **test real held-out** con el mismo swarm: adaptar `--dataset` a un
dump del split real (`load_splits()["test"]`) — o documentar como follow-up si excede el alcance.

---

## Notas

- **Quota:** smoke en pilot primero (paso 3); v1 (200) son ~14 subagentes/ronda. Acotar a 2-3 rondas.
- **Recall saturado en v1:** el eje real que mide es precisión; al leer resultados, no celebrar R alto.
- **El campo `model` del genoma es ignorado por el swarm** (el extractor es el subagente).
</content>
