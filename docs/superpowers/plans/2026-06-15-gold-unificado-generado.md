# Gold único v3 (generado-anclado) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un gold standard único `gold_v3` = los 93 artículos reales + 100 chunks cortos generados por Opus 4.8 (anclados en noticias reales de los top-10, difficulty 1–10), consumible por el loop vía un switch de versión.

**Architecture:** Pipeline determinista en Python (filtrar anclas, plantar verdad, ensamblar al esquema del gold real) + fan-out LLM en un Workflow JS (Opus lee el ancla → andamiaje; Opus redacta el chunk + quotes; verificador no-Opus descarta relaciones sin quote defendible). Un módulo `gold_version` centraliza las rutas para apuntar al gold actual (93, repo hermano) o a `data/gold_v3/` (193) sin tocar el repo hermano.

**Tech Stack:** Python 3 (pandas, pyarrow, pyyaml, rapidfuzz, pytest), Workflow JS (herramienta Workflow), Opus 4.8 (redactor) + Sonnet (verificador, no-Opus).

**Spec:** `docs/superpowers/specs/2026-06-15-gold-unificado-generado-design.md`

---

## File Structure

**Nuevos:**
- `swarm_optimizer/gold_version.py` — resuelve rutas del gold por `GOLD_VERSION` (env). Una responsabilidad: dónde vive el gold.
- `scripts/synth_mine_anchors.py` — filtra el parquet del corpus → `data/gold_v3/anchors.parquet`.
- `scripts/build_gold_v3.py` — compone guiones (planta verdad), ensambla redacciones al esquema del gold real, unifica (copia 93 + añade 100), regenera splits. CLI con subcomandos.
- `scripts/gen_gold_v3_workflow.js` — Workflow LLM, dos modos (`mine`, `redact`).
- `swarm_optimizer/tests/test_gold_version.py`, `test_mine_anchors.py`, `test_build_gold_v3.py` — tests.

**Modificados:**
- `swarm_optimizer/splits.py`, `swarm_optimizer/loop.py`, `swarm_optimizer/rubric.py` — usar `gold_version` en vez de constantes hardcodeadas.
- `.gitignore` — ignorar `data/`.

**Esquemas de los JSON intermedios (en `data/gold_v3/`):**
- `scaffolding.json`: `[{anchor_id, source, publish_date, domain, registro, actores:[{name,type}], distractor_patterns:[str], difficulty_hint:int(1-10), es_distractor:bool}]`
- `guiones.json`: `[{gen_id, anchor_id, source, publish_date, domain, registro, difficulty:int, stratum, es_distractor, actores:[{union_id,name,type}], relaciones:[{u_from,u_to,act_type,polarity}]}]`
- `redacciones.json`: `[{gen_id, title, body, entities:[{union_id,canonical_name,type,surfaces:[str]}], relations:[{u_from,u_to,act_type,polarity,evidence_quote,defensible:bool}]}]`

**Layout final de `data/gold_v3/`:** `articles.parquet`, `gold_final.parquet`, `entity_unions/<id16>.yaml`, `splits.json`.

---

## Task 1: Módulo `gold_version` (switch de rutas)

**Files:**
- Create: `swarm_optimizer/gold_version.py`
- Test: `swarm_optimizer/tests/test_gold_version.py`

- [ ] **Step 1: Write the failing test**

```python
# swarm_optimizer/tests/test_gold_version.py
import importlib
from pathlib import Path
from swarm_optimizer import gold_version as gv


def test_current_points_to_sibling_repo(monkeypatch):
    monkeypatch.delenv("GOLD_VERSION", raising=False)
    p = gv.gold_paths()
    assert p["articles"].as_posix().endswith("gold_standard_v5/data/pilot_gold_articles.parquet")
    assert p["relations"].as_posix().endswith("gold_standard_v5/data/pilot_gold_final.parquet")
    assert p["unions_dir"].as_posix().endswith("gold_standard_v5/data/pilot_entity_unions")
    assert p["splits"].as_posix().endswith("results/swarm/splits.json")


def test_v3_points_local(monkeypatch):
    monkeypatch.setenv("GOLD_VERSION", "v3")
    p = gv.gold_paths()
    assert p["articles"].as_posix().endswith("data/gold_v3/articles.parquet")
    assert p["relations"].as_posix().endswith("data/gold_v3/gold_final.parquet")
    assert p["unions_dir"].as_posix().endswith("data/gold_v3/entity_unions")
    assert p["splits"].as_posix().endswith("data/gold_v3/splits.json")


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("GOLD_VERSION", "v3")
    p = gv.gold_paths("current")
    assert p["articles"].as_posix().endswith("gold_standard_v5/data/pilot_gold_articles.parquet")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_gold_version.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_optimizer.gold_version'`

- [ ] **Step 3: Write minimal implementation**

```python
# swarm_optimizer/gold_version.py
"""Resuelve las rutas del gold standard según GOLD_VERSION.

- "current" (default): el gold real en el repo hermano gold_standard_v5/ (93 art).
- "v3": el gold unificado local en data/gold_v3/ (193 art).

Centraliza lo que antes estaba hardcodeado en loop.py / splits.py / rubric.py.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).parent.parent          # text2graph-evolve/
_SIBLING = _REPO.parent / "gold_standard_v5" / "data"


def gold_paths(version: str | None = None) -> dict[str, Path]:
    """Devuelve {articles, relations, unions_dir, splits} para la versión pedida.

    version=None -> usa la env var GOLD_VERSION, o "current" si no está seteada.
    """
    version = version or os.environ.get("GOLD_VERSION", "current")
    if version == "v3":
        base = _REPO / "data" / "gold_v3"
        return {
            "articles": base / "articles.parquet",
            "relations": base / "gold_final.parquet",
            "unions_dir": base / "entity_unions",
            "splits": base / "splits.json",
        }
    if version == "current":
        return {
            "articles": _SIBLING / "pilot_gold_articles.parquet",
            "relations": _SIBLING / "pilot_gold_final.parquet",
            "unions_dir": _SIBLING / "pilot_entity_unions",
            "splits": _REPO / "results" / "swarm" / "splits.json",
        }
    raise ValueError(f"GOLD_VERSION desconocido: {version!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_gold_version.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/gold_version.py swarm_optimizer/tests/test_gold_version.py
git commit -m "feat(gold): switch de versión del gold (current|v3) centralizado"
```

---

## Task 2: Conectar `splits.py`, `rubric.py`, `loop.py` al switch

**Files:**
- Modify: `swarm_optimizer/splits.py:10-13` (GOLD_ARTICLES, SPLITS_PATH)
- Modify: `swarm_optimizer/rubric.py:17-24` (UNIONS_DIR, GOLD_PARQUET)
- Modify: `swarm_optimizer/loop.py:22-23` (GOLD_ARTICLES, GOLD_PARQUET)
- Test: `swarm_optimizer/tests/test_gold_version.py` (añadir)

- [ ] **Step 1: Write the failing test**

Añadir a `swarm_optimizer/tests/test_gold_version.py`:

```python
def test_current_matches_legacy_hardcoded_paths(monkeypatch):
    """Con GOLD_VERSION ausente, las rutas deben ser EXACTAMENTE las viejas (no romper)."""
    monkeypatch.delenv("GOLD_VERSION", raising=False)
    from swarm_optimizer import splits, rubric
    p = gv.gold_paths()
    assert splits.gold_articles_path() == p["articles"]
    assert splits.splits_path() == p["splits"]
    assert rubric.unions_dir() == p["unions_dir"]
    assert rubric.gold_relations_path() == p["relations"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_gold_version.py::test_current_matches_legacy_hardcoded_paths -q`
Expected: FAIL with `AttributeError: module 'swarm_optimizer.splits' has no attribute 'gold_articles_path'`

- [ ] **Step 3: Write minimal implementation**

En `swarm_optimizer/splits.py`, reemplazar las constantes (líneas 10-13) por funciones:

```python
from swarm_optimizer.gold_version import gold_paths


def gold_articles_path():
    return gold_paths()["articles"]


def splits_path():
    return gold_paths()["splits"]
```

Y en `generate_splits`/`load_splits`, usar `gold_articles_path()` y `splits_path()` donde antes se usaban `GOLD_ARTICLES` y `SPLITS_PATH` (líneas 29, 59, 62, 63).

En `swarm_optimizer/rubric.py`, reemplazar las constantes (líneas 17-24) por:

```python
from swarm_optimizer.gold_version import gold_paths


def unions_dir():
    return gold_paths()["unions_dir"]


def gold_relations_path():
    return gold_paths()["relations"]
```

Y en `load_union` (línea 56) usar `unions_dir() / f"{article_id[:16]}.yaml"`; donde se use `GOLD_PARQUET` usar `gold_relations_path()`.

En `swarm_optimizer/loop.py` (líneas 22-23, y 138/140), reemplazar:

```python
from swarm_optimizer.splits import gold_articles_path, load_splits, subsample
from swarm_optimizer.rubric import compute_metrics, load_union_map, gold_relations_path
# ...
if articles_df is None:
    articles_df = pd.read_parquet(gold_articles_path())
if gold_df is None:
    gold_df = pd.read_parquet(gold_relations_path())
```

- [ ] **Step 4: Run full test suite to verify nothing broke**

Run: `python -m pytest swarm_optimizer/tests/ -q`
Expected: PASS (todos los tests existentes + los nuevos; el comportamiento por defecto es idéntico al viejo)

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/splits.py swarm_optimizer/rubric.py swarm_optimizer/loop.py swarm_optimizer/tests/test_gold_version.py
git commit -m "refactor(gold): loop/splits/rubric leen rutas vía gold_version"
```

---

## Task 3: `.gitignore` + copiar el parquet del corpus

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Añadir a `.gitignore`**

```
# corpus de noticias (7.2GB) y gold generado local
data/raw/
data/gold_v3/anchors.parquet
data/gold_v3/scaffolding.json
data/gold_v3/guiones.json
data/gold_v3/redacciones.json
```

(El gold final `data/gold_v3/{articles,gold_final}.parquet`, `entity_unions/`, `splits.json` SÍ se versionan — son el dataset; los intermedios y el corpus crudo no.)

- [ ] **Step 2: Copiar el parquet del corpus al repo**

```bash
mkdir -p "data/raw"
cp "/c/Users/Benjamin Palacios/Downloads/data/raw/articles_all.parquet" "data/raw/articles_all.parquet"
```

Expected: archivo de ~7.2GB en `data/raw/` (ignorado por git).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore(gold): ignorar corpus crudo e intermedios de gold_v3"
```

---

## Task 4: Minado de anclas (`synth_mine_anchors.py`)

**Files:**
- Create: `scripts/synth_mine_anchors.py`
- Test: `swarm_optimizer/tests/test_mine_anchors.py`

- [ ] **Step 1: Write the failing test**

```python
# swarm_optimizer/tests/test_mine_anchors.py
import pandas as pd
from scripts.synth_mine_anchors import fix_encoding, is_candidate, infer_domain, mine_anchors


def test_fix_encoding_repairs_mojibake():
    assert fix_encoding("FarÃ­as") == "Farías"          # mojibake recuperable (ftfy)
    assert fix_encoding("Viña del Mar") == "Viña del Mar"  # ya limpio


def test_fix_encoding_flags_unrecoverable():
    assert fix_encoding("Far�as") is None          # U+FFFD = irrecuperable -> None (descartar)


def test_is_candidate_filters_by_len_and_keyword():
    pol = "El ministro de Hacienda anunció el proyecto de ley ante el Congreso."
    assert is_candidate(pol, "latercera.com") is True
    assert is_candidate("x" * 50, "latercera.com") is False         # muy corto
    assert is_candidate("Receta de cazuela casera para el invierno chileno y sus secretos de cocina tradicional aquí.", "latercera.com") is False  # no político


def test_infer_domain():
    assert infer_domain("El Senado votó el proyecto de pensiones del gobierno") == "politica"
    assert infer_domain("El delantero marcó un gol en el partido de la ANFP") == "futbol"


def test_mine_anchors_deterministic_and_stratified():
    df = pd.DataFrame({
        "id": [f"a{i}" for i in range(40)],
        "source": (["latercera.com", "emol.com"] * 20),
        "publish_date": pd.to_datetime(["2024-06-01"] * 40),
        "year": [2024] * 40,
        "title": ["t"] * 40,
        "body": ["El ministro y el Senado debatieron el proyecto de ley del gobierno. " * 5] * 40,
    })
    out1 = mine_anchors(df, n=10, seed=42)
    out2 = mine_anchors(df, n=10, seed=42)
    assert len(out1) == 10
    assert out1["id"].tolist() == out2["id"].tolist()        # determinista
    assert set(out1.columns) >= {"id", "source", "publish_date", "title", "body", "domain"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_mine_anchors.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.synth_mine_anchors'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/synth_mine_anchors.py
"""Filtra el corpus (data/raw/articles_all.parquet) a anclas candidatas:
top-10 medios políticos, recientes, cortas, con señal política. Salida:
data/gold_v3/anchors.parquet (~250 candidatos para elegir 100 al generar)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ftfy import fix_text as _ftfy
except Exception:  # ftfy opcional
    _ftfy = None

TOP10 = ["latercera.com", "emol.com", "cooperativa.cl", "biobiochile.cl", "elmostrador.cl",
         "t13.cl", "adnradio.cl", "24horas.cl", "lanacion.cl", "cnnchile.com"]

_KW = re.compile(r"(?i)\b(ministr|congreso|senado|diputad|gobierno|oposici|partido|"
                 r"moci[oó]n|proyecto de ley|contralor|fiscal|presidente|subsecretari|"
                 r"alcalde|parlament|moneda|boric|kast|matthei|jara)\w*")
_FUTBOL = re.compile(r"(?i)\b(gol|delantero|anfp|estadio|hincha|dt|partido|copa|seleccci|árbitro)")

REPO = Path(__file__).parent.parent
OUT = REPO / "data" / "gold_v3"


def fix_encoding(text: str) -> str | None:
    """Repara mojibake recuperable (ftfy); devuelve None si hay U+FFFD (irrecuperable)."""
    if "�" in text:
        return None
    return _ftfy(text) if _ftfy else text


def is_candidate(body: str, source: str) -> bool:
    return source in TOP10 and 450 <= len(body) <= 1300 and bool(_KW.search(body))


def infer_domain(body: str) -> str:
    if _FUTBOL.search(body):
        return "futbol"
    return "politica"


def mine_anchors(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["body"] = df["body"].astype(str)
    df = df[df.apply(lambda r: is_candidate(r["body"], r["source"]), axis=1)]
    df["domain"] = df["body"].map(infer_domain)
    # muestreo estratificado por (source, domain), determinista
    rng = np.random.default_rng(seed)
    df = df.sort_values("id").reset_index(drop=True)
    per = max(1, n // max(1, df["source"].nunique()))
    picks = []
    for _, g in df.groupby("source", sort=True):
        ids = g["id"].to_numpy(dtype=object)
        rng.shuffle(ids)
        picks.extend(ids[:per].tolist())
    picks = picks[:n] if len(picks) >= n else picks
    return df[df["id"].isin(picks)].sort_values("id").head(n).reset_index(drop=True)


def main():
    src = REPO / "data" / "raw" / "articles_all.parquet"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    df = pd.read_parquet(src, columns=["id", "source", "publish_date", "year", "title", "body"])
    df = df[(df["source"].isin(TOP10)) & (df["year"] >= 2024)]
    df["body"] = df["body"].map(lambda t: fix_encoding(str(t)))
    df["title"] = df["title"].map(lambda t: fix_encoding(str(t)))
    df = df.dropna(subset=["body", "title"])          # descarta texto irrecuperable
    out = mine_anchors(df, n=n, seed=42)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "anchors.parquet")
    print(f"anclas: {len(out)} | dominios: {out['domain'].value_counts().to_dict()} | "
          f"medios: {out['source'].nunique()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_mine_anchors.py -q`
Expected: PASS (5 passed). Si falla `test_fix_encoding_repairs_mojibake` por falta de ftfy: `pip install ftfy` y re-correr.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth_mine_anchors.py swarm_optimizer/tests/test_mine_anchors.py
git commit -m "feat(gold): minado de anclas del corpus (top-10, político, corto)"
```

---

## Task 5: Correr el minado sobre el corpus real

**Files:** ninguno (ejecución)

- [ ] **Step 1: Generar anclas**

Run: `python scripts/synth_mine_anchors.py 250`
Expected: imprime `anclas: 250 | dominios: {...} | medios: 10` y crea `data/gold_v3/anchors.parquet`.

- [ ] **Step 2: Verificar a ojo 3 anclas**

```bash
python -c "import pandas as pd; d=pd.read_parquet('data/gold_v3/anchors.parquet'); print(d[['source','domain','title']].head(3).to_string()); print('len body p50:', int(d['body'].str.len().median()))"
```
Expected: títulos políticos legibles (sin `�`), body mediana 600–1200 chars. Si hay mucho ruido no-político, endurecer `_KW` y re-correr.

---

## Task 6: Composición de guiones (plantar verdad) en `build_gold_v3.py`

**Files:**
- Create: `scripts/build_gold_v3.py`
- Test: `swarm_optimizer/tests/test_build_gold_v3.py`

- [ ] **Step 1: Write the failing test**

```python
# swarm_optimizer/tests/test_build_gold_v3.py
import random
from scripts.build_gold_v3 import (
    ACT_TYPES, difficulty_to_stratum, compose_guion,
)


def test_difficulty_to_stratum():
    assert difficulty_to_stratum(2) == "G_d1-3"
    assert difficulty_to_stratum(5) == "G_d4-6"
    assert difficulty_to_stratum(8) == "G_d7-8"
    assert difficulty_to_stratum(10) == "G_d9-10"


def test_compose_guion_relational():
    scaffold = {
        "anchor_id": "a1", "source": "emol.com", "publish_date": "2024-06-01",
        "domain": "politica", "registro": "formal",
        "actores": [{"name": "Gabriel Boric", "type": "roster_actor"},
                    {"name": "José Antonio Kast", "type": "roster_actor"},
                    {"name": "Ministerio de Hacienda", "type": "institutional_actor"}],
        "distractor_patterns": ["co_mencion"], "difficulty_hint": 7, "es_distractor": False,
    }
    g = compose_guion(scaffold, gen_id="gen_001", rng=random.Random(1))
    assert g["gen_id"] == "gen_001"
    assert g["stratum"] == "G_d7-8"
    assert 1 <= len(g["relaciones"]) <= 5
    # union_ids asignados y referidos por las relaciones
    uids = {a["union_id"] for a in g["actores"]}
    for r in g["relaciones"]:
        assert r["u_from"] in uids and r["u_to"] in uids
        assert r["act_type"] in ACT_TYPES          # solo los 9 canónicos
        assert r["act_type"] != "co_occurs"         # nunca la clase FP


def test_compose_guion_distractor_has_zero_relations():
    scaffold = {"anchor_id": "a2", "source": "t13.cl", "publish_date": "2024-06-01",
                "domain": "politica", "registro": "formal",
                "actores": [{"name": "X", "type": "non_roster_actor"},
                            {"name": "Y", "type": "non_roster_actor"}],
                "distractor_patterns": ["co_mencion"], "difficulty_hint": 8, "es_distractor": True}
    g = compose_guion(scaffold, gen_id="gen_050", rng=random.Random(2))
    assert g["relaciones"] == []
    assert g["es_distractor"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_gold_v3'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/build_gold_v3.py
"""Construye el gold unificado v3: compone guiones (planta verdad sobre el
andamiaje), ensambla las redacciones al esquema del gold real, y unifica
(copia los 93 reales + añade los 100 generados). CLI con subcomandos."""
from __future__ import annotations

import random

# 9 act_types canónicos (del sintético); NUNCA co_occurs (es la clase FP).
ACT_TYPES = {
    "accuses": "negative", "endorses": "positive", "calls_on": "neutral",
    "allies_with": "positive", "distances_from": "negative", "attacks": "negative",
    "questions": "negative", "negotiates_with": "positive", "competes_with": "negative",
}
_ACTS = list(ACT_TYPES)


def difficulty_to_stratum(d: int) -> str:
    if d <= 3:
        return "G_d1-3"
    if d <= 6:
        return "G_d4-6"
    if d <= 8:
        return "G_d7-8"
    return "G_d9-10"


def compose_guion(scaffold: dict, gen_id: str, rng: random.Random) -> dict:
    """Asigna union_ids a los actores y planta 1-5 relaciones (0 si distractor)."""
    actores = [{"union_id": f"U{i+1}", "name": a["name"], "type": a["type"]}
               for i, a in enumerate(scaffold["actores"])]
    relaciones = []
    if not scaffold.get("es_distractor") and len(actores) >= 2:
        n_rel = rng.randint(1, min(5, len(actores)))
        usados = set()
        intentos = 0
        while len(relaciones) < n_rel and intentos < 40:
            intentos += 1
            a, b = rng.sample(actores, 2)
            key = (a["union_id"], b["union_id"])
            if key in usados:
                continue
            usados.add(key)
            act = rng.choice(_ACTS)
            relaciones.append({"u_from": a["union_id"], "u_to": b["union_id"],
                               "act_type": act, "polarity": ACT_TYPES[act]})
    d = int(scaffold["difficulty_hint"])
    return {
        "gen_id": gen_id, "anchor_id": scaffold["anchor_id"], "source": scaffold["source"],
        "publish_date": scaffold["publish_date"], "domain": scaffold["domain"],
        "registro": scaffold.get("registro", "formal"), "difficulty": d,
        "stratum": difficulty_to_stratum(d), "es_distractor": bool(scaffold.get("es_distractor")),
        "actores": actores, "relaciones": relaciones,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_gold_v3.py swarm_optimizer/tests/test_build_gold_v3.py
git commit -m "feat(gold): composición de guiones (planta verdad, 9 act_types, sin co_occurs)"
```

---

## Task 7: Parsear redacciones → filas del gold (descartar lo no defendible)

**Files:**
- Modify: `scripts/build_gold_v3.py`
- Test: `swarm_optimizer/tests/test_build_gold_v3.py` (añadir)

- [ ] **Step 1: Write the failing test**

Añadir a `swarm_optimizer/tests/test_build_gold_v3.py`:

```python
from scripts.build_gold_v3 import redaccion_to_rows


def _guion(gen_id="gen_001"):
    return {"gen_id": gen_id, "anchor_id": "a1", "source": "emol.com",
            "publish_date": "2024-06-01", "domain": "politica", "registro": "formal",
            "difficulty": 7, "stratum": "G_d7-8", "es_distractor": False,
            "actores": [{"union_id": "U1", "name": "Boric", "type": "roster_actor"},
                        {"union_id": "U2", "name": "Kast", "type": "roster_actor"}],
            "relaciones": [{"u_from": "U1", "u_to": "U2", "act_type": "attacks", "polarity": "negative"}]}


def test_redaccion_to_rows_drops_undefendable():
    red = {"gen_id": "gen_001", "title": "T", "body": "Boric atacó a Kast por la reforma.",
           "entities": [{"union_id": "U1", "canonical_name": "Gabriel Boric", "type": "roster_actor",
                         "surfaces": ["Boric", "el Presidente"]},
                        {"union_id": "U2", "canonical_name": "José Antonio Kast", "type": "roster_actor",
                         "surfaces": ["Kast"]}],
           "relations": [
               {"u_from": "U1", "u_to": "U2", "act_type": "attacks", "polarity": "negative",
                "evidence_quote": "Boric atacó a Kast", "defensible": True},
               {"u_from": "U2", "u_to": "U1", "act_type": "accuses", "polarity": "negative",
                "evidence_quote": "", "defensible": False},   # sin quote defendible -> se descarta
           ]}
    art, gold, union = redaccion_to_rows(_guion(), red)
    assert art["article_id"] == "gen_001" and art["stratum"] == "G_d7-8" and art["difficulty"] == 7
    assert len(gold) == 1                                   # la no-defendible se eliminó
    assert gold[0]["evidence_quote"] == "Boric atacó a Kast"
    assert gold[0]["source"] == "opus_planted"
    assert {e["union_id"] for e in union["entities_union"]} == {"U1", "U2"}
    assert union["entities_union"][0]["surfaces"]           # surfaces presentes para el matching
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py::test_redaccion_to_rows_drops_undefendable -q`
Expected: FAIL with `ImportError: cannot import name 'redaccion_to_rows'`

- [ ] **Step 3: Write minimal implementation**

Añadir a `scripts/build_gold_v3.py`:

```python
def redaccion_to_rows(guion: dict, red: dict) -> tuple[dict, list[dict], dict]:
    """Convierte una redacción verificada en (fila de articles, filas de gold, union YAML).

    Regla "100% seguro": solo entran relaciones con defensible=True y evidence_quote no vacío.
    """
    gid = guion["gen_id"]
    body = red["body"]
    article = {
        "article_id": gid, "stratum": guion["stratum"], "period": "generado",
        "title": red.get("title", ""), "body": body,
        "publish_date": guion["publish_date"], "difficulty": guion["difficulty"],
        "domain": guion["domain"], "registro": guion["registro"],
        "es_distractor": guion["es_distractor"], "body_chars": len(body),
    }
    gold = []
    for r in red.get("relations", []):
        if not r.get("defensible") or not (r.get("evidence_quote") or "").strip():
            continue
        gold.append({
            "article_id": gid, "u_from": r["u_from"], "u_to": r["u_to"],
            "act_type": r["act_type"], "polarity": r.get("polarity", ""),
            "is_reactive": False, "issue": guion["domain"],
            "evidence_quote": r["evidence_quote"], "source": "opus_planted",
            "n_inclusion_votes": None, "dispute_type": None,
        })
    union = {
        "article_id": gid, "n_annotators_total": 1, "annotators": ["opus"],
        "entities_union": [
            {"union_id": e["union_id"], "type": e.get("type", "non_roster_actor"),
             "actor_id": None, "canonical_names": [e["canonical_name"]],
             "surfaces": e.get("surfaces", [e["canonical_name"]]), "annotators": ["opus"]}
            for e in red.get("entities", [])
        ],
    }
    return article, gold, union
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_gold_v3.py swarm_optimizer/tests/test_build_gold_v3.py
git commit -m "feat(gold): redacción verificada -> filas de gold (descarta no-defendibles)"
```

---

## Task 8: Ensamblar + unificar (copiar 93 reales + añadir generados + splits)

**Files:**
- Modify: `scripts/build_gold_v3.py`
- Test: `swarm_optimizer/tests/test_build_gold_v3.py` (añadir)

- [ ] **Step 1: Write the failing test**

Añadir a `swarm_optimizer/tests/test_build_gold_v3.py`:

```python
import json
import pandas as pd
import yaml
from scripts.build_gold_v3 import assemble_unified


def test_assemble_unified_merges_and_splits(tmp_path):
    # gold real falso (2 art) + unions
    real = tmp_path / "real"
    (real / "entity_unions").mkdir(parents=True)
    pd.DataFrame([{"article_id": "r1aaaaaaaaaaaaaa", "stratum": "S1", "period": "2024H1",
                   "title": "t", "body": "b", "publish_date": "2024-01-01",
                   "n_elite_actors_matched": 1, "n_inst_actors_matched": 0, "body_chars": 1}
                  ]).to_parquet(real / "pilot_gold_articles.parquet")
    pd.DataFrame([{"article_id": "r1aaaaaaaaaaaaaa", "u_from": "U1", "u_to": "U2",
                   "act_type": "endorses", "polarity": "positive", "is_reactive": False,
                   "issue": "x", "evidence_quote": "q", "source": "unanimous_pass2",
                   "n_inclusion_votes": None, "dispute_type": None}]
                 ).to_parquet(real / "pilot_gold_final.parquet")
    (real / "entity_unions" / "r1aaaaaaaaaaaaaa.yaml").write_text(
        yaml.safe_dump({"article_id": "r1aaaaaaaaaaaaaa", "entities_union": [
            {"union_id": "U1", "type": "roster_actor", "canonical_names": ["A"], "surfaces": ["A"]}]}),
        encoding="utf-8")

    arts = [{"article_id": "gen_001", "stratum": "G_d7-8", "period": "generado", "title": "T",
             "body": "Boric atacó a Kast", "publish_date": "2024-06-01", "difficulty": 7,
             "domain": "politica", "registro": "formal", "es_distractor": False, "body_chars": 18}]
    gold = [{"article_id": "gen_001", "u_from": "U1", "u_to": "U2", "act_type": "attacks",
             "polarity": "negative", "is_reactive": False, "issue": "politica",
             "evidence_quote": "Boric atacó a Kast", "source": "opus_planted",
             "n_inclusion_votes": None, "dispute_type": None}]
    unions = [{"article_id": "gen_001", "entities_union": [
        {"union_id": "U1", "type": "roster_actor", "canonical_names": ["Boric"], "surfaces": ["Boric"]},
        {"union_id": "U2", "type": "roster_actor", "canonical_names": ["Kast"], "surfaces": ["Kast"]}]}]

    out = tmp_path / "gold_v3"
    assemble_unified(real_dir=real, articles=arts, gold=gold, unions=unions, out_dir=out, seed=42)

    a = pd.read_parquet(out / "articles.parquet")
    g = pd.read_parquet(out / "gold_final.parquet")
    assert set(a["article_id"]) == {"r1aaaaaaaaaaaaaa", "gen_001"}      # 93 reales (aquí 1) + generados
    assert len(g) == 2
    assert (out / "entity_unions" / "gen_001.yaml").exists()           # nombre = id[:16]
    assert (out / "entity_unions" / "r1aaaaaaaaaaaaaa.yaml").exists()  # copiado
    split = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    assert set(split["eval"]).isdisjoint(split["test"])                # held-out limpio
    assert set(split["eval"]) | set(split["test"]) == {"r1aaaaaaaaaaaaaa", "gen_001"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py::test_assemble_unified_merges_and_splits -q`
Expected: FAIL with `ImportError: cannot import name 'assemble_unified'`

- [ ] **Step 3: Write minimal implementation**

Añadir a `scripts/build_gold_v3.py`:

```python
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _write_union_yaml(union: dict, unions_dir: Path) -> None:
    aid = union["article_id"]
    (unions_dir / f"{aid[:16]}.yaml").write_text(
        yaml.safe_dump(union, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _stratified_split(articles_df: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    eval_ids, test_ids = [], []
    for _, g in articles_df.groupby("stratum", sort=True):
        ids = g["article_id"].to_numpy(dtype=object)
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * 30 / 93))
        test_ids.extend(ids[:n_test].tolist())
        eval_ids.extend(ids[n_test:].tolist())
    return {"eval": sorted(eval_ids), "test": sorted(test_ids), "seed": seed}


def assemble_unified(real_dir: Path, articles: list[dict], gold: list[dict],
                     unions: list[dict], out_dir: Path, seed: int) -> None:
    """Copia el gold real (real_dir) y añade los generados → out_dir (gold_v3)."""
    real_dir, out_dir = Path(real_dir), Path(out_dir)
    (out_dir / "entity_unions").mkdir(parents=True, exist_ok=True)

    real_arts = pd.read_parquet(real_dir / "pilot_gold_articles.parquet")
    real_gold = pd.read_parquet(real_dir / "pilot_gold_final.parquet")
    gen_arts = pd.DataFrame(articles)
    gen_gold = pd.DataFrame(gold)

    # union de columnas (los generados aportan difficulty/domain/registro/es_distractor)
    pd.concat([real_arts, gen_arts], ignore_index=True).to_parquet(out_dir / "articles.parquet")
    pd.concat([real_gold, gen_gold], ignore_index=True).to_parquet(out_dir / "gold_final.parquet")

    # copiar unions reales + escribir los generados
    for y in (real_dir / "entity_unions").glob("*.yaml"):
        shutil.copy(y, out_dir / "entity_unions" / y.name)
    for u in unions:
        _write_union_yaml(u, out_dir / "entity_unions")

    all_arts = pd.read_parquet(out_dir / "articles.parquet")
    split = _stratified_split(all_arts, seed=seed)
    (out_dir / "splits.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_gold_v3.py swarm_optimizer/tests/test_build_gold_v3.py
git commit -m "feat(gold): ensamblaje unificado (copia 93 reales + añade generados + split)"
```

---

## Task 9: Workflow LLM `gen_gold_v3_workflow.js` (Opus redacta, Sonnet verifica)

**Files:**
- Create: `scripts/gen_gold_v3_workflow.js`

> Nota: este Workflow no se testea con pytest (es no-determinista). Se valida con un probe de 2 ítems en el Step 2. Modela el patrón de `scripts/synth_extract_workflow.js`.

- [ ] **Step 1: Write the workflow**

```javascript
// scripts/gen_gold_v3_workflow.js
export const meta = {
  name: 'gen-gold-v3',
  description: 'Genera chunks anclados con verdad plantada (Opus redacta, Sonnet verifica)',
  phases: [{ title: 'Mine', detail: 'Opus lee anclas → andamiaje' },
           { title: 'Redact', detail: 'Opus redacta + Sonnet verifica' }],
}

// args = { mode: 'mine'|'redact', items: [...] }
const mode = args && args.mode
const items = (args && Array.isArray(args.items)) ? args.items : []

const MINE_SCHEMA = {
  type: 'object',
  properties: {
    actores: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, type: { type: 'string' } }, required: ['name', 'type'] } },
    domain: { type: 'string' }, registro: { type: 'string' },
    distractor_patterns: { type: 'array', items: { type: 'string' } },
    difficulty_hint: { type: 'integer' },
    es_distractor: { type: 'boolean' },
  },
  required: ['actores', 'domain', 'difficulty_hint', 'es_distractor'],
}

const REDACT_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' }, body: { type: 'string' },
    entities: { type: 'array', items: { type: 'object', properties: {
      union_id: { type: 'string' }, canonical_name: { type: 'string' },
      type: { type: 'string' }, surfaces: { type: 'array', items: { type: 'string' } },
    }, required: ['union_id', 'canonical_name', 'surfaces'] } },
    relations: { type: 'array', items: { type: 'object', properties: {
      u_from: { type: 'string' }, u_to: { type: 'string' }, act_type: { type: 'string' },
      polarity: { type: 'string' }, evidence_quote: { type: 'string' }, defensible: { type: 'boolean' },
    }, required: ['u_from', 'u_to', 'act_type', 'evidence_quote', 'defensible'] } },
  },
  required: ['title', 'body', 'entities', 'relations'],
}

if (mode === 'mine') {
  phase('Mine')
  log(`Minando andamiaje de ${items.length} anclas (Opus)`)
  const out = await parallel(items.map(it => () =>
    agent(
      `Sos analista político chileno. Leé esta NOTICIA REAL y extraé el andamiaje para crear ` +
      `después un chunk sintético verosímil (NO copies el texto). Devolvé: actores (con type ∈ ` +
      `{roster_actor, institutional_actor, non_roster_actor}), domain, registro ` +
      `(formal|coloquial|internacional), distractor_patterns (co-menciones tentadoras sin relación), ` +
      `difficulty_hint 1-10 (qué tan oblicua sería la relación), y es_distractor (true si no hay ` +
      `relación política entre actores).\n\nNOTICIA (${it.source}):\n${it.body}`,
      { label: `mine:${it.anchor_id}`, phase: 'Mine', model: 'opus', schema: MINE_SCHEMA })
      .then(r => ({ anchor_id: it.anchor_id, source: it.source, publish_date: it.publish_date, ...r }))
      .catch(() => null)))
  return out.filter(Boolean)
}

if (mode === 'redact') {
  phase('Redact')
  log(`Redactando + verificando ${items.length} guiones (Opus→Sonnet)`)
  const out = await parallel(items.map(g => () => {
    const actBlock = g.actores.map(a => `${a.union_id}=${a.name} (${a.type})`).join('; ')
    const relBlock = g.relaciones.map(r => `${r.u_from} ${r.act_type} ${r.u_to} (${r.polarity})`).join('; ')
    const redactPrompt =
      `Escribí un chunk de noticia chilena NUEVO (120-250 palabras), registro ${g.registro}, ` +
      `dominio ${g.domain}, dificultad ${g.difficulty}/10. Usá EXACTAMENTE estos actores: ${actBlock}. ` +
      `Expresá SOLO estas relaciones plantadas: ${relBlock || '(NINGUNA — es distractor: co-menciona ' +
      'actores sin relación política entre ellos)'}. Para cada relación, marcá la frase exacta del body ` +
      `que la soporta (evidence_quote) y defensible=true SOLO si la frase la expresa sin ambigüedad. ` +
      `Devolvé entities (con surfaces realmente usadas por actor) y relations.`
    return agent(redactPrompt, { label: `redact:${g.gen_id}`, phase: 'Redact', model: 'opus', schema: REDACT_SCHEMA })
      .then(red => {
        const verifyPrompt =
          `Sos un verificador escéptico (NO el autor). Leé el body y juzgá cada relación: ¿la ` +
          `evidence_quote está literalmente en el body y la expresa sin ambigüedad? Marcá defensible=false ` +
          `si la quote no aparece o es ambigua. NO agregues relaciones nuevas salvo que sean inequívocas ` +
          `(y marcalas defensible=true). Devolvé el MISMO objeto con defensible corregido.\n\n` +
          `BODY:\n${red.body}\n\nRELATIONS:\n${JSON.stringify(red.relations)}`
        return agent(verifyPrompt, { label: `verify:${g.gen_id}`, phase: 'Redact', model: 'sonnet', schema: REDACT_SCHEMA })
          .then(v => ({ gen_id: g.gen_id, title: v.title || red.title, body: red.body,
                        entities: red.entities, relations: v.relations || red.relations }))
          .catch(() => ({ gen_id: g.gen_id, title: red.title, body: red.body,
                          entities: red.entities, relations: red.relations }))
      })
      .catch(() => null)
  }))
  return out.filter(Boolean)
}

throw new Error(`mode inválido: ${mode}`)
```

- [ ] **Step 2: Probe con 2 anclas (verificación manual)**

Primero generar los prompts de mining de 2 anclas:

```bash
python -c "import pandas as pd, json; d=pd.read_parquet('data/gold_v3/anchors.parquet').head(2); print(json.dumps({'mode':'mine','items':[{'anchor_id':r['id'],'source':r['source'],'publish_date':str(r['publish_date'])[:10],'body':r['body']} for _,r in d.iterrows()]}, ensure_ascii=False))"
```

Luego, **invocar el Workflow** (herramienta Workflow) con `scriptPath: scripts/gen_gold_v3_workflow.js` y `args` = el JSON de arriba.
Expected: devuelve 2 andamiajes con `actores`, `domain`, `difficulty_hint`, `es_distractor`. Revisar a ojo que los actores sean del tema del ancla.

- [ ] **Step 3: Commit**

```bash
git add scripts/gen_gold_v3_workflow.js
git commit -m "feat(gold): workflow LLM gen-gold-v3 (mine/redact, Opus+Sonnet verify)"
```

---

## Task 10: CLI puente Python↔Workflow en `build_gold_v3.py`

**Files:**
- Modify: `scripts/build_gold_v3.py`
- Test: `swarm_optimizer/tests/test_build_gold_v3.py` (añadir)

- [ ] **Step 1: Write the failing test**

Añadir a `swarm_optimizer/tests/test_build_gold_v3.py`:

```python
from scripts.build_gold_v3 import compose_all


def test_compose_all_assigns_ids_and_difficulty_distribution():
    scaffolds = [{"anchor_id": f"a{i}", "source": "emol.com", "publish_date": "2024-06-01",
                  "domain": "politica", "registro": "formal",
                  "actores": [{"name": "Boric", "type": "roster_actor"},
                              {"name": "Kast", "type": "roster_actor"}],
                  "distractor_patterns": [], "difficulty_hint": (i % 10) + 1,
                  "es_distractor": (i % 7 == 0)} for i in range(20)]
    guiones = compose_all(scaffolds, seed=42)
    assert [g["gen_id"] for g in guiones] == [f"gen_{i+1:03d}" for i in range(20)]
    assert all(g["stratum"].startswith("G_d") for g in guiones)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py::test_compose_all_assigns_ids_and_difficulty_distribution -q`
Expected: FAIL with `ImportError: cannot import name 'compose_all'`

- [ ] **Step 3: Write minimal implementation**

Añadir a `scripts/build_gold_v3.py`:

```python
import sys


def compose_all(scaffolds: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [compose_guion(s, gen_id=f"gen_{i+1:03d}", rng=rng) for i, s in enumerate(scaffolds)]


def _cmd_compose(seed: int = 42):
    base = Path(__file__).parent.parent / "data" / "gold_v3"
    scaffolds = json.loads((base / "scaffolding.json").read_text(encoding="utf-8"))
    guiones = compose_all(scaffolds, seed=seed)
    (base / "guiones.json").write_text(json.dumps(guiones, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"guiones: {len(guiones)} | con relaciones: {sum(1 for g in guiones if g['relaciones'])}")


def _cmd_assemble(seed: int = 42):
    base = Path(__file__).parent.parent / "data" / "gold_v3"
    guiones = {g["gen_id"]: g for g in json.loads((base / "guiones.json").read_text(encoding="utf-8"))}
    reds = json.loads((base / "redacciones.json").read_text(encoding="utf-8"))
    arts, gold, unions = [], [], []
    for red in reds:
        g = guiones.get(red["gen_id"])
        if not g:
            continue
        a, gr, u = redaccion_to_rows(g, red)
        arts.append(a)
        gold.extend(gr)
        unions.append(u)
    from swarm_optimizer.gold_version import gold_paths
    real_dir = gold_paths("current")["articles"].parent
    assemble_unified(real_dir=real_dir, articles=arts, gold=gold, unions=unions,
                     out_dir=base, seed=seed)
    print(f"gold_v3: {len(arts)} generados + reales. articles/gold_final/splits escritos.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "compose":
        _cmd_compose()
    elif cmd == "assemble":
        _cmd_assemble()
    else:
        print("uso: build_gold_v3.py [compose|assemble]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_build_gold_v3.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_gold_v3.py swarm_optimizer/tests/test_build_gold_v3.py
git commit -m "feat(gold): CLI compose/assemble (puente con el workflow)"
```

---

## Task 11: Correr la generación completa de los 100 (LLM)

**Files:** escribe `data/gold_v3/{scaffolding,guiones,redacciones}.json` y el gold final.

> Esta tarea ejecuta el Workflow real (Opus + Sonnet). Es la fase que consume tokens. Hacela con checkpoints.

- [ ] **Step 1: Mining de ~120 anclas**

Generar prompts de mining:
```bash
python -c "import pandas as pd, json; d=pd.read_parquet('data/gold_v3/anchors.parquet').head(120); print(json.dumps({'mode':'mine','items':[{'anchor_id':r['id'],'source':r['source'],'publish_date':str(r['publish_date'])[:10],'body':r['body']} for _,r in d.iterrows()]}, ensure_ascii=False))" > data/gold_v3/_mine_args.json
```
Invocar Workflow (`scriptPath: scripts/gen_gold_v3_workflow.js`, `args` = contenido de `_mine_args.json`). Guardar el resultado (lista de andamiajes) a `data/gold_v3/scaffolding.json`.

- [ ] **Step 2: Ajustar la distribución de dificultad a 100**

```bash
python -c "import json; s=json.load(open('data/gold_v3/scaffolding.json',encoding='utf-8')); import collections; print('n=',len(s)); print('dist hint:', collections.Counter(x['difficulty_hint'] for x in s)); print('distractores:', sum(x['es_distractor'] for x in s))"
```
Expected: ~120 andamiajes. Recortar/seleccionar a 100 con sesgo a 5–8 (editar `scaffolding.json` para quedar con ≈15 en 1–3, 35 en 4–6, 35 en 7–8, 15 en 9–10; ~15 distractores). Documentar cuántos se descartaron.

- [ ] **Step 3: Componer guiones (Python)**

Run: `python scripts/build_gold_v3.py compose`
Expected: `guiones: 100 | con relaciones: ~85`

- [ ] **Step 4: Redacción + verificación (Workflow)**

Generar args de redact:
```bash
python -c "import json; g=json.load(open('data/gold_v3/guiones.json',encoding='utf-8')); print(json.dumps({'mode':'redact','items':g}, ensure_ascii=False))" > data/gold_v3/_redact_args.json
```
Invocar Workflow (mismo `scriptPath`, `args` = `_redact_args.json`). Guardar el resultado a `data/gold_v3/redacciones.json`.

- [ ] **Step 5: Ensamblar el gold unificado**

Run: `python scripts/build_gold_v3.py assemble`
Expected: `gold_v3: 100 generados + reales...` y se crean `data/gold_v3/{articles.parquet,gold_final.parquet,splits.json}` + `entity_unions/*.yaml`.

- [ ] **Step 6: Verificar criterios de éxito**

```bash
python -c "
import pandas as pd, json
a=pd.read_parquet('data/gold_v3/articles.parquet'); g=pd.read_parquet('data/gold_v3/gold_final.parquet')
gen=a[a['article_id'].str.startswith('gen_')]
print('total art:', len(a), '| generados:', len(gen))
print('difficulty dist:', gen['difficulty'].value_counts().sort_index().to_dict())
print('distractores con 0 rels:', sum(1 for i in gen[gen['es_distractor']]['article_id'] if (g['article_id']==i).sum()==0), '/', int(gen['es_distractor'].sum()))
print('rels generadas con quote no vacío:', (g[g['source']=='opus_planted']['evidence_quote'].str.len()>0).mean())
"
```
Expected: ~100 generados; distribución 1–3/4–6/7–8/9–10 ≈ 15/35/35/15; **todos** los distractores con 0 relaciones; **100%** de las relaciones generadas con quote (defendibles).

- [ ] **Step 7: Spot-check humano**

Leer 8 chunks (`python -c "import pandas as pd; d=pd.read_parquet('data/gold_v3/articles.parquet'); d=d[d.article_id.str.startswith('gen_')].sample(8, random_state=1); [print('\n---',r.article_id, 'dif', r.difficulty, '\n', r.title, '\n', r.body) for r in d.itertuples()]"`) y confirmar: texto legible, relaciones plantadas presentes y defendibles. Si algo falla, regenerar ese subconjunto.

- [ ] **Step 8: Commit**

```bash
git add data/gold_v3/articles.parquet data/gold_v3/gold_final.parquet data/gold_v3/splits.json data/gold_v3/entity_unions
git commit -m "feat(gold): gold_v3 unificado (93 reales + 100 generados Opus, difficulty 1-10)"
```

---

## Task 12: Re-baseline + validación por stratum/difficulty

**Files:**
- Create: `scripts/report_gold_v3.py`

- [ ] **Step 1: Re-medir el subset-93 (no-regresión)**

Correr el campeón sobre `GOLD_VERSION=current` y luego sobre `v3` filtrando al subset-93, y comparar:
```bash
GOLD_VERSION=current python scripts/report_run.py -o results/swarm/baseline_current.md
```
Expected: las métricas del campeón sobre los 93 reales (referencia). Anotar P_rel/R_rel/f05.

- [ ] **Step 2: Correr el extractor objetivo sobre gold_v3**

Usar el harness de extracción existente (`scripts/synth_agent_eval.py` / loop) con `GOLD_VERSION=v3` para obtener `preds` del campeón sobre los 193. (Reusa el flujo dump→workflow→collect→score ya existente; apunta a las rutas de `gold_version`.)
Expected: `preds.jsonl` sobre 193 artículos.

- [ ] **Step 3: Reporte por stratum y difficulty**

```python
# scripts/report_gold_v3.py
"""Desglosa métricas del gold_v3 por stratum (S* reales vs G_* generados) y por difficulty."""
import sys
import pandas as pd
from swarm_optimizer.gold_version import gold_paths
from swarm_optimizer.rubric import compute_metrics, load_union_map


def main(preds_path: str):
    import os
    os.environ["GOLD_VERSION"] = "v3"
    arts = pd.read_parquet(gold_paths("v3")["articles"])
    gold = pd.read_parquet(gold_paths("v3")["relations"])
    preds = [__import__("json").loads(l) for l in open(preds_path, encoding="utf-8") if l.strip()]
    pred_map = {p["article_id"]: p for p in preds}
    # por difficulty (solo generados, que tienen la columna)
    gen = arts[arts["article_id"].str.startswith("gen_")]
    for d in sorted(gen["difficulty"].dropna().unique()):
        ids = gen[gen["difficulty"] == d]["article_id"].tolist()
        um = load_union_map(ids)
        m = compute_metrics({i: pred_map.get(i, {"entities": [], "relations": []}) for i in ids},
                            gold[gold["article_id"].isin(ids)], um)
        print(f"difficulty {int(d)}: P_rel={m['precision_rel']:.3f} R_rel={m['recall_rel']:.3f} "
              f"f05={m.get('f05_rel', float('nan')):.3f} (n={len(ids)})")


if __name__ == "__main__":
    main(sys.argv[1])
```

Run: `GOLD_VERSION=v3 python scripts/report_gold_v3.py results/swarm/preds_v3.jsonl`
Expected: una línea por difficulty con P_rel/R_rel/f05; la precisión debería bajar al subir la dificultad (señal de que la escala discrimina).

> Nota: ajustar las claves del dict que devuelve `compute_metrics` (`precision_rel`, `recall_rel`, `f05_rel`) a las reales de `swarm_optimizer/rubric.py` si difieren; verificar leyendo la función `compute_metrics` antes de correr.

- [ ] **Step 4: Documentar el re-baseline**

Escribir `results/swarm/gold_v3_baseline.md` con: métricas globales sobre v3, desglose por stratum y difficulty, y confirmación de que el subset-93 coincide con `baseline_current.md` (±ruido). Esto deja registrado el cambio de target.

- [ ] **Step 5: Commit**

```bash
git add scripts/report_gold_v3.py results/swarm/gold_v3_baseline.md
git commit -m "feat(gold): reporte y re-baseline de gold_v3 por stratum/difficulty"
```

---

## Self-Review (hecho)

- **Cobertura del spec:** §3 esquema de fusión → T6/T7/T8; §4 pipeline → T4/T5/T9/T10/T11; §5 difficulty 1–10 → T6 (`difficulty_to_stratum`) + T11 step 2/6; §6 top-10 → T4 (`TOP10`); §7 unificación/switch/re-baseline → T1/T2/T8/T12; §8 circularidad (Opus redacta, Sonnet verifica, gemini extrae) → T9; §9 criterios de éxito → T11 step 6/7; §11 archivos → todos.
- **Placeholders:** sin TBD/TODO; el único punto a verificar en ejecución son las claves de `compute_metrics` (T12 step 3, anotado explícitamente).
- **Consistencia de tipos:** `gen_id`/`article_id` (`gen_NNN`), `union_id` (`U#`), esquemas `scaffolding/guiones/redacciones` definidos en File Structure y usados igual en T6/T7/T9/T10; `assemble_unified(real_dir, articles, gold, unions, out_dir, seed)` consistente entre T8 y T10.
