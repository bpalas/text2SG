# Analysis Tool (artefacto C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introducir un artefacto determinista (Analysis Tool) que transforma el roster por-artículo (`union`) + el body en un bloque estructurado de análisis de actores, reemplazando la inyección plana de 6 líneas, y medirlo con un harness limpio + conteo de LoC.

**Architecture:** Tercer artefacto `AnalysisConfig` (en genome.py, junto a ValidationConfig) + módulo `analysis.py` con `build_analysis(union, body, cfg)` que emite 8 secciones gateadas (dossier, alias, roles, dirección, hablante, co-menciones, canon de act_type, gate de dominio). Se cablea en `build_prompt` vía `getattr(config, "analysis", None)` (backward-compatible). Se evalúa pareado en el test split comparando semilla vs +analysis vs +analysis+verify.

**Tech Stack:** Python 3, dataclasses, `re`, `rapidfuzz` (ya en uso), pytest, pandas/parquet, Gemini API (solo para el eval pago).

---

## File Structure

- **Create** `swarm_optimizer/analysis.py` — Analysis Tool: constantes + helpers por sección + `build_analysis`. Una responsabilidad: convertir (union, body) → bloque de texto.
- **Create** `swarm_optimizer/tests/test_analysis.py` — tests unitarios del tool.
- **Modify** `swarm_optimizer/genome.py` — nueva dataclass `AnalysisConfig` + campo `Genome.analysis` + serialización en `from_dict`.
- **Modify** `swarm_optimizer/extractor.py` — `build_prompt` usa el bloque rico cuando hay `analysis`.
- **Create** `scripts/measure_loc.py` — tabla de LoC de los 3 artefactos.
- **Create** `scripts/eval_analysis_tool.py` — comparación pareada de las 3 configs en el test split.

`AnalysisConfig` vive en genome.py (no en analysis.py) para evitar import circular: `analysis.py` importa `AnalysisConfig` desde `genome.py`, igual que `validation.py` importa `ValidationConfig`. El default de `role_keywords` es `None`; `build_analysis` cae a `DEFAULT_ROLE_KEYWORDS` (definido en analysis.py) cuando es `None`. Así genome.py no referencia constantes de analysis.py.

**Nota sobre `Config` legacy (config.py):** NO se modifica. `build_prompt` lee el artefacto con `getattr(config, "analysis", None)`, que devuelve `None` para `Config`. Esto evita un import circular (config.py ↔ genome.py) y es más limpio que el texto literal del spec.

---

### Task 1: `AnalysisConfig` y campo `Genome.analysis`

**Files:**
- Modify: `swarm_optimizer/genome.py`
- Test: `swarm_optimizer/tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Crear `swarm_optimizer/tests/test_analysis.py` con:

```python
from swarm_optimizer.genome import Genome, AnalysisConfig


def test_genome_roundtrip_with_analysis():
    g = Genome(prompt_text="x", analysis=AnalysisConfig(emit_dossier=False, role_window=25))
    g2 = Genome.from_json(g.to_json())
    assert g2.analysis is not None
    assert g2.analysis.emit_dossier is False
    assert g2.analysis.role_window == 25


def test_genome_roundtrip_without_analysis():
    g = Genome(prompt_text="x")
    g2 = Genome.from_json(g.to_json())
    assert g2.analysis is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: FAIL con `ImportError: cannot import name 'AnalysisConfig'`.

- [ ] **Step 3: Add `AnalysisConfig` and the field**

En `swarm_optimizer/genome.py`, agregar la dataclass justo después de `ValidationConfig` (antes de `class Genome`):

```python
@dataclass
class AnalysisConfig:
    """Artefacto C: análisis determinista pre-extracción (costo $0).
    Cada flag gatea una sección del bloque que produce analysis.build_analysis()."""
    emit_dossier: bool = True
    emit_alias_map: bool = True
    emit_role_hints: bool = True
    emit_direction_scaffold: bool = True
    emit_main_speaker: bool = True
    emit_comention_pairs: bool = True
    emit_act_type_canon: bool = True
    emit_domain_gate: bool = True
    role_window: int = 80          # ventana ±chars alrededor de cada mención para detectar rol
    role_keywords: dict | None = None   # None → usa DEFAULT_ROLE_KEYWORDS de analysis.py
```

En `class Genome`, agregar el campo después de `validation`:

```python
    analysis: "AnalysisConfig | None" = None
```

Reemplazar `from_dict` por:

```python
    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        data = dict(data)
        vc = data.pop("validation", {}) or {}
        adata = data.pop("analysis", None)
        analysis = AnalysisConfig(**adata) if adata else None
        return cls(validation=ValidationConfig(**vc), analysis=analysis, **data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/genome.py swarm_optimizer/tests/test_analysis.py
git commit -m "feat: AnalysisConfig (artefacto C) + serialización en Genome"
```

---

### Task 2: `analysis.py` — constantes, canon de act_type, dossier y alias

**Files:**
- Create: `swarm_optimizer/analysis.py`
- Test: `swarm_optimizer/tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Agregar a `swarm_optimizer/tests/test_analysis.py`:

```python
from swarm_optimizer.analysis import (
    canon_act_type, _actor_dossier, _alias_map, _act_type_canon_block,
)


def _union():
    return {
        "U1": {"type": "roster_actor", "canonical_names": ["Luis Hermosilla"],
               "surfaces": ["Hermosilla", "Luis Hermosilla"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Juan Pablo Hermosilla"],
               "surfaces": ["Juan Pablo Hermosilla"]},
        "U9": {"type": "NIL", "canonical_names": ["Ruido"], "surfaces": ["ruido"]},
    }


def test_canon_act_type_maps_noncanonical():
    assert canon_act_type("kill") == "attacks"
    assert canon_act_type("CRITICIZES") == "accuses"
    assert canon_act_type("attacks") == "attacks"     # ya canónico, sin cambio


def test_actor_dossier_excludes_nil_and_lists_aliases():
    out = _actor_dossier(_union())
    assert "Luis Hermosilla" in out
    assert "Juan Pablo Hermosilla" in out
    assert "Ruido" not in out                          # NIL excluido


def test_alias_map_maps_surface_to_canonical():
    out = _alias_map(_union())
    assert "Hermosilla" in out and "Luis Hermosilla" in out


def test_act_type_canon_block_lists_canonical_types():
    out = _act_type_canon_block()
    assert "attacks" in out and "endorses" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swarm_optimizer.analysis'`.

- [ ] **Step 3: Create `swarm_optimizer/analysis.py`**

```python
"""
Artefacto C: Analysis Tool determinista (pre-extracción, costo $0).
Transforma el roster por-artículo (union) + el body en un bloque estructurado
que se inyecta en el prompt. Mueve hechos por-artículo fuera del prompt frágil.
"""
from __future__ import annotations

import re

from swarm_optimizer.genome import SEED_ALLOWED_ACT_TYPES
from swarm_optimizer.rubric import normalize

# Vocabulario canónico de act_type (los permitidos en la semilla, sin co_occurs).
CANONICAL_ACT_TYPES = list(SEED_ALLOWED_ACT_TYPES)

# Equivalencias act_type no-canónico → canónico (auditoría gold 2026-06-10: ~20%
# del gold usa tipos no canónicos que castigan Act_acc sin afectar P_rel/R_rel).
ACT_TYPE_CANON = {
    "kill": "attacks", "kills": "attacks", "attack": "attacks", "ataca": "attacks",
    "defends": "endorses", "defend": "endorses", "supports": "endorses",
    "respalda": "endorses", "apoya": "endorses",
    "criticizes": "accuses", "criticize": "accuses", "critica": "accuses",
    "meet_with": "negotiates_with", "meets_with": "negotiates_with",
    "se_reune_con": "negotiates_with",
    "appoints": "endorses", "names": "endorses", "nombra": "endorses",
    "calls_for": "calls_on", "urges": "calls_on", "llama_a": "calls_on",
    "opposes": "competes_with", "se_opone": "competes_with",
}

# Palabras de rol → etiqueta normalizada (para los hints de rol, Task 3).
DEFAULT_ROLE_KEYWORDS = {
    "abogado": "abogado/defensa", "abogada": "abogado/defensa",
    "defensor": "abogado/defensa", "defensa": "abogado/defensa",
    "imputado": "imputado", "imputada": "imputado", "acusado": "imputado",
    "fiscal": "fiscal",
    "ministro": "ministro", "ministra": "ministro",
    "diputado": "diputado", "diputada": "diputado",
    "senador": "senador", "senadora": "senador",
    "presidente": "presidente", "presidenta": "presidente",
    "alcalde": "alcalde", "alcaldesa": "alcalde",
}

# Términos de dominio deportivo (gate de no-político, Task 4).
FOOTBALL_TERMS = ["gol", "partido", "futbol", "club", "seleccion", "delantero",
                  "arquero", "copa", "estadio", "entrenador"]


def canon_act_type(act: str | None) -> str:
    """Mapea un act_type a su forma canónica (idempotente si ya es canónico)."""
    a = (act or "").strip().lower()
    return ACT_TYPE_CANON.get(a, a)


def _real_actors(union: dict) -> list[tuple[str, dict]]:
    """Actores no-NIL del union, como [(uid, ent), ...]."""
    return [(uid, ent) for uid, ent in union.items() if ent.get("type") != "NIL"]


def _canon_name(ent: dict) -> str:
    names = ent.get("canonical_names") or ["?"]
    return names[0]


def _actor_dossier(union: dict) -> str:
    lines = []
    for _uid, ent in _real_actors(union):
        aliases = ", ".join(ent.get("surfaces") or []) or "—"
        lines.append(f"  {_canon_name(ent)} (tipo: {ent.get('type', '?')}; alias: {aliases})")
    if not lines:
        return ""
    return "ACTORES (usa el nombre canónico, nunca el código U1/U2):\n" + "\n".join(lines)


def _alias_map(union: dict) -> str:
    lines = []
    for _uid, ent in _real_actors(union):
        canon = _canon_name(ent)
        for surf in ent.get("surfaces") or []:
            if normalize(surf) != normalize(canon):
                lines.append(f"  '{surf}' → {canon}")
    if not lines:
        return ""
    return "MAPA DE ALIAS (normaliza menciones al canónico):\n" + "\n".join(lines)


def _act_type_canon_block() -> str:
    canon = ", ".join(CANONICAL_ACT_TYPES)
    mappings = "; ".join(f"{k}→{v}" for k, v in ACT_TYPE_CANON.items())
    return (f"ACT_TYPES CANÓNICOS (usa SOLO estos): {canon}.\n"
            f"Si dudás, mapea: {mappings}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: PASS (todos verdes).

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/analysis.py swarm_optimizer/tests/test_analysis.py
git commit -m "feat: analysis.py — canon act_type, dossier de actores, mapa de alias"
```

---

### Task 3: Hints de rol — desambiguación de apellido compartido (bug Hermosilla)

**Files:**
- Modify: `swarm_optimizer/analysis.py`
- Test: `swarm_optimizer/tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Agregar a `swarm_optimizer/tests/test_analysis.py`:

```python
from swarm_optimizer.analysis import _role_hints


def test_role_hints_disambiguates_shared_surname():
    union = _union()
    # Las dos menciones separadas por relleno > role_window para no contaminarse.
    body = ("Juan Pablo Hermosilla, abogado defensor, intervino en la audiencia. "
            "Relleno neutral de la nota para separar bien las dos menciones del texto. "
            "El imputado Luis Hermosilla guardó silencio ante el tribunal.")
    out = _role_hints(union, body, None, 25)
    assert "AMBIGÜEDAD" in out                 # comparten apellido 'Hermosilla'
    assert "abogado/defensa" in out            # Juan Pablo
    assert "imputado" in out                   # Luis


def test_role_hints_no_role_when_absent():
    union = {"U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"],
                    "surfaces": ["Boric"]}}
    out = _role_hints(union, "Gabriel Boric habló en La Moneda.", None, 80)
    assert "rol no detectado" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py::test_role_hints_disambiguates_shared_surname -q`
Expected: FAIL con `ImportError: cannot import name '_role_hints'`.

- [ ] **Step 3: Add `_role_hints` to `analysis.py`**

Agregar al final de `swarm_optimizer/analysis.py`:

```python
def _role_hints(union: dict, body: str, role_keywords: dict | None, window: int) -> str:
    """Detecta el rol probable de cada actor por palabras-clave en una ventana de
    texto alrededor de sus menciones. Para evitar el sobre-match del apellido pelado
    (que aparece en ambos 'Hermosilla'), busca solo con nombres multi-token (o el
    canónico). Marca AMBIGÜEDAD cuando dos actores comparten apellido."""
    role_keywords = role_keywords or DEFAULT_ROLE_KEYWORDS
    actors = _real_actors(union)
    if not actors:
        return ""

    # Apellidos compartidos → ambigüedad.
    surnames: dict[str, list[str]] = {}
    for _uid, ent in actors:
        sn = normalize(_canon_name(ent)).split()
        if sn:
            surnames.setdefault(sn[-1], []).append(_canon_name(ent))
    ambiguous = {s for s, names in surnames.items() if len(names) > 1}

    body_norm = normalize(body)
    lines = []
    for _uid, ent in actors:
        canon = _canon_name(ent)
        # Nombres de búsqueda: multi-token (o canónico si no hay), nunca apellido pelado.
        search_names = [
            n for n in (ent.get("surfaces") or []) + (ent.get("canonical_names") or [])
            if len(normalize(n).split()) >= 2
        ] or [canon]

        roles: set[str] = set()
        for name in search_names:
            sn = normalize(name)
            if not sn:
                continue
            start = 0
            while True:
                i = body_norm.find(sn, start)
                if i == -1:
                    break
                lo = max(0, i - window)
                hi = min(len(body_norm), i + len(sn) + window)
                ctx = body_norm[lo:hi]
                for kw, role in role_keywords.items():
                    if kw in ctx:
                        roles.add(role)
                start = i + len(sn)

        surname = normalize(canon).split()[-1] if normalize(canon).split() else ""
        flag = "  [AMBIGÜEDAD: comparte apellido con otro actor]" if surname in ambiguous else ""
        role_str = ", ".join(sorted(roles)) if roles else "rol no detectado"
        lines.append(f"  {canon}: {role_str}{flag}")

    return "ROLES DETECTADOS (heurística — desambigua atribución):\n" + "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/analysis.py swarm_optimizer/tests/test_analysis.py
git commit -m "feat: hints de rol con desambiguación de apellido compartido (bug Hermosilla)"
```

---

### Task 4: Dirección, hablante principal, co-menciones y gate de dominio

**Files:**
- Modify: `swarm_optimizer/analysis.py`
- Test: `swarm_optimizer/tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Agregar a `swarm_optimizer/tests/test_analysis.py`:

```python
from swarm_optimizer.analysis import (
    _direction_scaffold, _main_speaker, _comention_pairs, _domain_gate,
)


def test_direction_scaffold_passive_voice():
    out = _direction_scaffold("Boric fue criticado por Matthei en la sesión.")
    assert "from=Matthei" in out and "to=Boric" in out


def test_direction_scaffold_no_passive_returns_empty():
    assert _direction_scaffold("Boric habló en La Moneda.") == ""


def test_main_speaker_picks_most_mentioned():
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"], "surfaces": ["Boric"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Evelyn Matthei"], "surfaces": ["Matthei"]},
    }
    body = "Boric anunció. Boric defendió. Boric insistió. Matthei respondió."
    out = _main_speaker(union, body)
    assert "Gabriel Boric" in out


def test_comention_pairs_same_sentence():
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"], "surfaces": ["Boric"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Evelyn Matthei"], "surfaces": ["Matthei"]},
    }
    out = _comention_pairs(union, "Boric y Matthei coincidieron en el acto.")
    assert "Gabriel Boric" in out and "Evelyn Matthei" in out


def test_domain_gate_flags_football():
    out = _domain_gate({}, "El club ganó el partido con un gol en el estadio.")
    assert out != "" and "deportivo" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: FAIL con `ImportError` de los 4 helpers nuevos.

- [ ] **Step 3: Add the four helpers to `analysis.py`**

Agregar al final de `swarm_optimizer/analysis.py`:

```python
def _direction_scaffold(body: str) -> str:
    """Detecta pasiva '<paciente> fue <participio> por <agente>' y emite la dirección
    correcta (from=agente, to=paciente). Usa el body crudo para preservar mayúsculas."""
    pat = re.compile(
        r"([A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*)*)"
        r"\s+fue\s+\w+\s+por\s+"
        r"([A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ.]*)*)"
    )
    hits = []
    for m in pat.finditer(body):
        patient, agent = m.group(1).strip(), m.group(2).strip()
        hits.append(f"  voz pasiva: from={agent}, to={patient}")
    if not hits:
        return ""
    return "DIRECCIÓN (corrige el sentido en pasiva):\n" + "\n".join(hits)


def _main_speaker(union: dict, body: str) -> str:
    body_norm = normalize(body)
    ranked = []
    for _uid, ent in _real_actors(union):
        count = 0
        for surf in (ent.get("surfaces") or []) + (ent.get("canonical_names") or []):
            sn = normalize(surf)
            if sn:
                count += body_norm.count(sn)
        ranked.append((count, _canon_name(ent)))
    if not ranked:
        return ""
    ranked.sort(reverse=True)
    top_count, top_name = ranked[0]
    if top_count == 0:
        return ""
    return (f"HABLANTE PRINCIPAL (más mencionado): {top_name} ({top_count} menciones) — "
            f"sujeto probable de varias relaciones; no lo omitas.")


def _comention_pairs(union: dict, body: str) -> str:
    actors = _real_actors(union)
    pairs: set[tuple[str, str]] = set()
    for sentence in re.split(r"[.!?]\s+", body):
        sent_norm = normalize(sentence)
        present = []
        for _uid, ent in actors:
            for surf in (ent.get("surfaces") or []) + (ent.get("canonical_names") or []):
                sn = normalize(surf)
                if sn and sn in sent_norm:
                    present.append(_canon_name(ent))
                    break
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                pairs.add(tuple(sorted((present[a], present[b]))))
    if not pairs:
        return ""
    lines = [f"  {a} ↔ {b}" for a, b in sorted(pairs)]
    return "PARES CO-MENCIONADOS (candidatos a relación, verifica el verbo):\n" + "\n".join(lines)


def _domain_gate(union: dict, body: str) -> str:
    body_norm = normalize(body)
    hits = [t for t in FOOTBALL_TERMS if t in body_norm]
    if len(hits) >= 2:
        return ("AVISO DE DOMINIO: el texto parece deportivo/no-político (términos: "
                + ", ".join(hits) + "). Sé MÁS estricto: solo relaciones políticas explícitas.")
    return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/analysis.py swarm_optimizer/tests/test_analysis.py
git commit -m "feat: dirección/voz, hablante principal, co-menciones y gate de dominio"
```

---

### Task 5: `build_analysis` — orquestador con gating

**Files:**
- Modify: `swarm_optimizer/analysis.py`
- Test: `swarm_optimizer/tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Agregar a `swarm_optimizer/tests/test_analysis.py`:

```python
from swarm_optimizer.analysis import build_analysis


def test_build_analysis_empty_union_returns_empty():
    assert build_analysis({}, "texto", AnalysisConfig()) == ""


def test_build_analysis_respects_gates():
    cfg = AnalysisConfig(
        emit_dossier=True, emit_alias_map=False, emit_role_hints=False,
        emit_direction_scaffold=False, emit_main_speaker=False,
        emit_comention_pairs=False, emit_act_type_canon=False, emit_domain_gate=False,
    )
    out = build_analysis(_union(), "Luis Hermosilla habló.", cfg)
    assert "ACTORES" in out
    assert "MAPA DE ALIAS" not in out
    assert "=== ANÁLISIS DE ACTORES ===" in out


def test_build_analysis_full_has_all_sections():
    out = build_analysis(_union(), "Juan Pablo Hermosilla, abogado, habló.", AnalysisConfig())
    assert "ACTORES" in out
    assert "MAPA DE ALIAS" in out
    assert "ROLES DETECTADOS" in out
    assert "ACT_TYPES CANÓNICOS" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: FAIL con `ImportError: cannot import name 'build_analysis'`.

- [ ] **Step 3: Add `build_analysis` to `analysis.py`**

Agregar al final de `swarm_optimizer/analysis.py`:

```python
def build_analysis(union: dict, body: str, cfg) -> str:
    """Orquesta las secciones gateadas en un bloque único. Devuelve '' si el union
    está vacío o ninguna sección produce contenido."""
    if not union:
        return ""
    sections = []
    if cfg.emit_dossier:
        sections.append(_actor_dossier(union))
    if cfg.emit_alias_map:
        sections.append(_alias_map(union))
    if cfg.emit_role_hints:
        sections.append(_role_hints(union, body, cfg.role_keywords, cfg.role_window))
    if cfg.emit_direction_scaffold:
        sections.append(_direction_scaffold(body))
    if cfg.emit_main_speaker:
        sections.append(_main_speaker(union, body))
    if cfg.emit_comention_pairs:
        sections.append(_comention_pairs(union, body))
    if cfg.emit_act_type_canon:
        sections.append(_act_type_canon_block())
    if cfg.emit_domain_gate:
        sections.append(_domain_gate(union, body))

    sections = [s for s in sections if s]
    if not sections:
        return ""
    return "=== ANÁLISIS DE ACTORES ===\n" + "\n\n".join(sections) + "\n=== FIN ANÁLISIS ===\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swarm_optimizer/analysis.py swarm_optimizer/tests/test_analysis.py
git commit -m "feat: build_analysis — orquestador de secciones con gating"
```

---

### Task 6: Cablear el Analysis Tool en `build_prompt`

**Files:**
- Modify: `swarm_optimizer/extractor.py:54-82` (función `build_prompt`)
- Test: `swarm_optimizer/tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Agregar a `swarm_optimizer/tests/test_analysis.py`:

```python
from swarm_optimizer.extractor import build_prompt


def test_build_prompt_uses_analysis_block_when_present():
    g = Genome(prompt_text="INSTRUCCIONES", architecture="given_entities",
               analysis=AnalysisConfig())
    prompt = build_prompt(g, "Luis Hermosilla habló en la audiencia.", _union(), [])
    assert "=== ANÁLISIS DE ACTORES ===" in prompt
    assert "INSTRUCCIONES" in prompt


def test_build_prompt_falls_back_to_flat_list_without_analysis():
    g = Genome(prompt_text="INSTRUCCIONES", architecture="given_entities", analysis=None)
    prompt = build_prompt(g, "Luis Hermosilla habló.", _union(), [])
    assert "=== ANÁLISIS DE ACTORES ===" not in prompt
    assert "ACTORES PRESENTES EN EL ARTÍCULO:" in prompt   # comportamiento legacy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py::test_build_prompt_uses_analysis_block_when_present -q`
Expected: FAIL — el bloque de análisis no aparece (build_prompt aún no lo usa).

- [ ] **Step 3: Modify `build_prompt`**

En `swarm_optimizer/extractor.py`, reemplazar el bloque `given_entities` (líneas 73-79) por:

```python
    analysis_cfg = getattr(config, "analysis", None)
    if analysis_cfg is not None and union:
        from swarm_optimizer.analysis import build_analysis
        block = build_analysis(union, body, analysis_cfg)
        if block:
            parts.append(block)
    elif config.architecture == "given_entities" and union:
        actor_list = "\n".join(
            f"  {uid}: {ent.get('canonical_names', ['?'])[0]} (tipo: {ent.get('type', '?')})"
            for uid, ent in union.items()
            if ent.get("type") != "NIL"
        )
        parts.append(f"ACTORES PRESENTES EN EL ARTÍCULO:\n{actor_list}\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest swarm_optimizer/tests/test_analysis.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite (no regression)**

Run: `python -m pytest swarm_optimizer/tests/ -q`
Expected: PASS (la suite existente sigue verde — el cambio es backward-compatible).

- [ ] **Step 6: Commit**

```bash
git add swarm_optimizer/extractor.py swarm_optimizer/tests/test_analysis.py
git commit -m "feat: build_prompt usa el bloque de análisis cuando hay AnalysisConfig"
```

---

### Task 7: Scripts de medición de LoC y evaluación pareada

**Files:**
- Create: `scripts/measure_loc.py`
- Create: `scripts/eval_analysis_tool.py`

- [ ] **Step 1: Create `scripts/measure_loc.py`**

```python
"""Tabla de líneas de código de los 3 artefactos (métrica narrativa estilo RoboPhD)."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _nonblank(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def main() -> None:
    cfg = (ROOT / "swarm_optimizer/config.py").read_text(encoding="utf-8")
    m = re.search(r'SEED_PROMPT = """\\?\n(.*?)"""', cfg, re.S)
    a_lines = len([l for l in m.group(1).splitlines() if l.strip()]) if m else 0
    b_lines = _nonblank(ROOT / "swarm_optimizer/genome.py")
    c_lines = _nonblank(ROOT / "swarm_optimizer/analysis.py")
    total = a_lines + b_lines + c_lines

    print("=== Líneas de código por artefacto (estilo RoboPhD) ===")
    print(f"  A — Prompt (SEED_PROMPT):         {a_lines:4d}")
    print(f"  B — Configs (genome.py):          {b_lines:4d}")
    print(f"  C — Analysis Tool (analysis.py):  {c_lines:4d}")
    print(f"  TOTAL:                            {total:4d}")
    print("  (RoboPhD: naive 70 → evolucionado ~1500)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run measure_loc**

Run: `python scripts/measure_loc.py`
Expected: imprime la tabla con A/B/C/TOTAL (números reales, C ya no es 0).

- [ ] **Step 3: Create `scripts/eval_analysis_tool.py`**

```python
"""Compara, pareado en el test split, 3 configs:
  semilla  |  +analysis  |  +analysis+verify
Mide P_rel/R_rel/F1_rel/Act_acc + tokens. ESTE EVAL GASTA API (Gemini).
Uso: python scripts/eval_analysis_tool.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd

from swarm_optimizer.config import SEED_PROMPT
from swarm_optimizer.genome import (Genome, ValidationConfig, AnalysisConfig,
                                    SEED_ALLOWED_ACT_TYPES)
from swarm_optimizer.extractor import run_extraction
from swarm_optimizer.rubric import compute_metrics, load_union_map
from swarm_optimizer.splits import load_splits

GOLD_ARTICLES = Path(__file__).parent.parent / "gold_standard_v5/data/pilot_gold_articles.parquet"
GOLD_PARQUET = Path(__file__).parent.parent / "gold_standard_v5/data/pilot_gold_final.parquet"

ANTI_UID = ("\nUsa el NOMBRE del actor tal como aparece en la lista proporcionada, "
            "nunca su código interno (U1, U2…).")


def _seed(analysis: AnalysisConfig | None = None, verify: bool = False) -> Genome:
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
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY no definida.")
        sys.exit(1)

    articles_df = pd.read_parquet(GOLD_ARTICLES)
    gold_df = pd.read_parquet(GOLD_PARQUET)
    test_ids = load_splits()["test"]
    union_map = load_union_map(test_ids)

    configs = {
        "semilla":          _seed(),
        "+analysis":        _seed(analysis=AnalysisConfig()),
        "+analysis+verify": _seed(analysis=AnalysisConfig(), verify=True),
    }

    print(f"Test split: {len(test_ids)} artículos (pareado, mismos ids)\n")
    for name, g in configs.items():
        preds, tokens = run_extraction(test_ids, articles_df, gold_df, union_map, g)
        m = compute_metrics(preds, test_ids, gold_df, union_map)
        print(f"{name:18s} P_rel={m['Precision_rel']:.3f} R_rel={m['Recall_rel']:.3f} "
              f"F1_rel={m['F1_rel']:.3f} Act_acc={m['Act_acc']:.3f} tok={tokens}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-check the eval script imports (no API spend)**

Run: `python -c "import ast; ast.parse(open('scripts/eval_analysis_tool.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK` (sintaxis válida; no ejecuta la API).

- [ ] **Step 5: Commit**

```bash
git add scripts/measure_loc.py scripts/eval_analysis_tool.py
git commit -m "feat: scripts de medición de LoC y evaluación pareada de las 3 configs"
```

---

### Task 8: Verificación final

**Files:** (ninguno nuevo — verificación)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest swarm_optimizer/tests/ -q`
Expected: PASS — la suite existente (111+ tests) más los nuevos de `test_analysis.py`, todos verdes.

- [ ] **Step 2: Print the LoC table**

Run: `python scripts/measure_loc.py`
Expected: tabla con C (analysis.py) bien por encima de 0 — el artefacto determinista ya carga peso.

- [ ] **Step 3: (Opcional, gasta API) correr el eval pareado**

Solo si hay `GEMINI_API_KEY` y se quiere medir el efecto real:
Run: `python scripts/eval_analysis_tool.py`
Expected: 3 filas; comparar `+analysis` vs `semilla` (delta de P_rel/F1_rel). Recordar: en el test (n≈30) deltas <0.05 son ruido — repetir ≥2 veces o confirmar en sintético.

- [ ] **Step 4: Confirmar que no quedó nada sin commitear de los archivos del plan**

NUNCA usar `git add -A` / `git add .` — el working tree tiene cambios no relacionados del usuario.
Verificar solo los archivos del plan:

```bash
git status --short swarm_optimizer/analysis.py swarm_optimizer/genome.py swarm_optimizer/extractor.py swarm_optimizer/tests/test_analysis.py scripts/measure_loc.py scripts/eval_analysis_tool.py
```
Expected: sin salida (todo commiteado en las tareas 1-7). Si aparece algo, `git add <ruta exacta>` solo de esos archivos y commitear.

---

## Notas de riesgo (de la spec)

- **Ventana de rol:** la heurística de `_role_hints` es sensible a `role_window`. En textos densos puede sobre-atribuir; el default 80 es punto de partida, no óptimo. Medir y ajustar contra sintético antes de tocar nada más.
- **Costo en tokens:** el bloque infla el prompt → sube `tokens` y baja el `score` por la penalización de costo en `fitness`. El eval imprime tokens para vigilarlo; si pesa, gatear secciones de bajo valor.
- **Sobreajuste:** el test split es el juez final; el sintético solo prioriza. No adoptar por una sola corrida con delta <0.05.
- **Oráculo sintético (follow-up, fuera de este plan):** el spec §5 lo pone como primario, pero este plan evalúa en el test split (APIs verificadas). Para la comparación en sintético hay que pasar el genoma `_seed(analysis=AnalysisConfig())` por el pipeline existente (`synth_sample_guiones.py → workflow → synth_assemble_and_eval.py`); requiere confirmar primero el hook de genoma de `synth_assemble_and_eval.py` (no inspeccionado acá). Se hace como paso separado una vez que el test split confirme que el andamiaje no empeora.
</content>
