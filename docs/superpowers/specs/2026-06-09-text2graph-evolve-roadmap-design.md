# Diseño — Rediseño evolutivo del pipeline de extracción text2graph

**Fecha:** 2026-06-09
**Estado:** Aprobado (brainstorming) — pendiente plan de implementación
**Autor:** Benjamín Palacios (con Claude)

---

## 1. Contexto y motivación

`text2graph-evolve` optimiza la configuración de un extractor de relaciones políticas
chilenas (texto → grafo signado) mediante un loop de auto-mejora basado en LLM
(Gemini congelado) contra un gold standard humano.

### Estado actual (diagnóstico)

El `swarm_optimizer` actual implementa un loop diagnóstico → propuesta → evaluación,
pero está estancado. Evidencia en `results/swarm/history.jsonl` y en el código:

- **Score clavado en ~0.52** tras 3 iteraciones; `is_plateau` corta el loop.
- **F1_rel ≈ 0.25, Precision_rel ≈ 0.20** — el cuello de botella real: demasiados
  falsos positivos en relaciones. (Polarity_acc ≈ 0.82 y Precision_ent ≈ 0.81 están bien.)
- El loop es **hill-climbing greedy de una sola línea** (`loop.py:249`): mantiene un
  único `current_config`, acepta solo si mejora +0.01. Sin población ni diversidad →
  óptimo local inmediato.
- `_propose` regenera **el JSON completo** del config (incluido el prompt entero);
  cualquier fallo de parseo cae silenciosamente al config actual (`loop.py:80`) →
  muchas propuestas son no-ops.
- No hay validación de que `evidence_quote` sea substring real del artículo, pese a que
  el prompt lo exige.
- `few_shots` nunca se puebla en la práctica.
- Se evalúa el set `eval` completo cada iteración (caro, sin submuestreo ni cascada).
- `archive.py` no es un archivo evolutivo: solo guarda historial y devuelve `best()`.

### Objetivo

Extraer **entidades y relaciones con alta precisión, al menor costo posible**, mediante
un proceso autoevolutivo más capaz, inspirado en papers de frontera.

### Decisión estratégica: camino evolutivo (no RL de pesos)

"Aprendizaje por refuerzo" aquí significa **optimización evolutiva del scaffolding** de un
LLM congelado (AlphaEvolve / DGM / RoboPhD), **no** actualización de pesos (GRPO/
REINFORCE++ requeriría un modelo open-weights + GPU). La recompensa guía la *selección*,
no el gradiente. RoboPhD muestra además que la evolución da las mayores ganancias en
modelos baratos (Haiku +8.9 pts vs Opus +2.3) — `gemini-2.5-flash` está justo en ese
régimen.

---

## 2. Decisiones de diseño (cerradas en brainstorming)

| # | Decisión | Elección |
|---|---|---|
| 1 | Mecanismo de selección | **Híbrido: ELO (skirmish barato) + championship en validación fija** |
| 2 | Función de fitness | **F0.5 (precisión 2×) ent+rel + piso de recall + penalización de costo** |
| 3 | Estructura de artefactos | **Prompt evolucionado (A) + capa de validación determinista parametrizada (B)** |
| 4 | Estrategia de búsqueda | **Enfoque 1: archivo evolutivo único (DGM) + ELO + cross-pollination** |
| 5 | Camino de "RL" | **Evolutivo sobre Gemini congelado (sin actualización de pesos)** |
| 6 | Alcance | **Rediseño estilo RoboPhD (Tier 1 completo)** |

### Técnicas de papers incorporadas (Tier 1)

- **DGM** — Archivo open-ended con selección de padres ∝ score y ∝ 1/#hijos; conserva
  stepping stones subóptimos.
- **RoboPhD** — Selección ELO (no-transitividad, entrada asíncrona); submuestreo de
  artículos por iteración (anti-overfit + costo); separación en dos artefactos;
  cross-pollination de top-2; verificación agéntica en inferencia; hallazgo de
  "inverse scaling" (evolución rinde más en modelos baratos).
- **AlphaEvolve** — Mutación por diff SEARCH/REPLACE (vs regenerar todo); ensemble/
  cascada de evaluación.

### Puertas de upgrade (Tier 2/3, fuera de alcance ahora — YAGNI)

- Grid **MAP-Elites** precision×costo (AlphaEvolve).
- **Retrospective Memory** con hybrid retrieval BM25+FAISS+RRF (MLEvolve).
- **Progressive MCGS** búsqueda en grafo (MLEvolve).
- **Proposer/Solver/Judge** co-evolución sin labels (Multi-Agent Evolve) — para romper el
  techo de 93 artículos.

### Descartado

- MARL DNN-mapping (nicho hardware, no aplica).
- Tournament de 6 agentes de AI Co-Scientist (ELO logra lo mismo más simple).
- RL de pesos (requiere open-weights + infra de entrenamiento).

---

## 3. Arquitectura

### 3.1 Genoma (lo que evoluciona)

Cada candidato extractor = dos artefactos:

- **Artefacto A — Prompt de extracción:** `prompt_text` + `few_shots` + flag
  `architecture` (`one_pass` | `given_entities`) + `model` + flag `verify`
  (verificación agéntica on/off). Muta por **diffs mínimos SEARCH/REPLACE** y
  cross-pollination.
- **Artefacto B — Config de validación determinista** (post-proceso puro-Python, costo $0):
  - `require_evidence_substring: bool` — descarta relaciones cuyo `evidence_quote` no sea
    substring verbatim (normalizado) del artículo.
  - `min_quote_len: int` — descarta citas demasiado cortas.
  - `direction_rules` — normalización de voz pasiva ("criticado por" → swap from/to).
  - `dedup` — colapsa triples (from, to, act) duplicados.
  - `allowed_act_types` — filtro a subconjunto de tipos.
  - `polarity_consistency` — coherencia act_type ↔ polaridad (ej. attacks → negative).
  - `max_relations_per_article` — guarda de precisión.

  Es la **palanca de precisión a costo cero**.

### 3.2 Archivo open-ended (DGM)

Guarda **todos** los genomas evaluados, cada uno con: rating ELO, score de championship
(cuando se mide), #hijos, linaje (id de padre, tipo de mutación, **artefacto tocado**),
e historial de métricas.

- **Selección de padres:** muestreo ∝ `sigmoid(ELO)` y ∝ `1/(1+#hijos)` → favorece
  genomas fuertes pero poco explorados; todos con probabilidad no-nula (stepping stones).

### 3.3 Fitness (reemplaza `Config.score`)

```
fitness = w_rel · F0.5_rel + w_ent · F0.5_ent + w_pol · Polarity_acc + w_act · Act_acc
          − λ · (tokens_por_artículo normalizado)
```

- `F0.5` pondera precisión 2× sobre recall (F-beta, β=0.5) para entidades y relaciones.
- **Piso de recall:** si `recall_rel < piso`, el genoma queda **descalificado como campeón**
  (fitness penalizado) pero **se conserva en el archivo** como stepping stone. Esto blinda
  contra la solución degenerada de "emitir casi nada para inflar precisión".
- `λ` (penalización de costo) favorece configs baratos en tokens — clave para "al menor costo".

### 3.4 Dos modos de evaluación

1. **Skirmish (barato, cada iteración):** el/los hijos compiten head-to-head contra el
   campeón actual + un contendiente top-ELO sobre un **submuestreo aleatorio fresco de K
   artículos** del pool `eval` (undersampling RoboPhD). La competencia 3-way se descompone
   en 3 pares → update ELO (K=32).
2. **Championship (anclado, cada M iteraciones):** re-evalúa el top-T por ELO sobre el set
   `eval` **fijo completo** → métricas F0.5 absolutas → corona al campeón desplegable. Cada
   cierto tiempo toca el split `test` para el chequeo de gap Goodhart.

### 3.5 Operadores de mutación (el "Evolution AI")

- `diagnose(fps, fns)` — desde los FP/FN reales del padre (reusa `_format_errors`).
- `mutate_diff(genome, diagnosis)` — propone un diff SEARCH/REPLACE mínimo al Artefacto A
  **o** un cambio único al Artefacto B (un artefacto por mutación, **etiquetado**).
  Reemplaza la regeneración de JSON completo.
- `cross_pollinate(top2)` — cada N iteraciones, combina las fortalezas complementarias de
  los dos genomas top-ELO en un hijo.
- Aplicación robusta de diffs: si el SEARCH no se encuentra → **no-op logueado** (no
  silencioso), el genoma no se agrega.

### 3.6 Verificación agéntica en inferencia (opcional, gateada por costo)

Pasada de auto-chequeo tras la extracción (RoboPhD "universal verification"):
"aquí están tus relaciones + el artículo; elimina las no soportadas literalmente, corrige
dirección/polaridad", hasta k=2 reintentos a temperatura progresiva (0.0 → 0.2 → 0.3).
Es un costo LLM → gateado por el flag `verify` del genoma, de modo que la evolución decida
si la ganancia de precisión supera su costo en tokens (interactúa con `λ`).

### 3.7 Flujo de una iteración

```
seleccionar padre del archivo (∝ ELO, ∝ 1/#hijos)
  → diagnosticar desde sus FP/FN reales
  → mutar (diff a A o cambio en B, etiquetado; cada N iters cross-pollination top-2)
  → extraer + aplicar validación determinista B (+ verificación agéntica si verify=on)
  → skirmish: head-to-head vs campeón + contendiente sobre submuestreo K
  → update ELO
  → cada M iteraciones: championship sobre eval fijo (+ test para Goodhart)
  → registrar genoma en el archivo
```

---

## 4. Componentes (módulos)

Se mantiene la cultura de módulos pequeños, puros y testeables. Frontera nítida entre
lógica de búsqueda (pura, determinista) y lo no-determinista/costoso (llamadas a Gemini).

| Módulo | Estado | Responsabilidad / interfaz |
|---|---|---|
| `genome.py` | nuevo (evoluciona `config.py`) | `Genome{ artifact_a, artifact_b, model, flags }` + serialización. `Config` se migra/envuelve para no romper tests. |
| `validation.py` | nuevo | Artefacto B determinista. `apply_validation(raw_output, body, union, vcfg) -> cleaned_output`. Puro, sin LLM, sin I/O. |
| `elo.py` | nuevo | `update_pairwise(a, b, result)`, `sample_parent(archive)`. Solo matemática de ratings + muestreo. |
| `archive.py` | upgrade | Archivo open-ended: ELO, linaje, #hijos, `select_parent()`, `top_by_elo(T)`, `champion()`. Hook stub para grid MAP-Elites. |
| `fitness.py` | nuevo (extrae de `config.py`) | `fitness(metrics, tokens) -> float` con F0.5 + piso de recall + penalización de costo. |
| `mutate.py` | nuevo (extrae de `loop.py`) | "Evolution AI": `diagnose`, `mutate_diff`, `cross_pollinate`. Aplicación robusta de diffs con log de no-op. |
| `rubric.py` | se mantiene | `match_entity`, `compute_metrics`, `load_union`. Se le añade F0.5 (o lo consume `fitness.py`). |
| `extractor.py` | upgrade | Llama a `validation.apply_validation`; pasada opcional de verificación agéntica gateada por `flags.verify`. |
| `loop.py` | rewrite | Orquesta skirmish/championship, archivo, budget, parada. Reemplaza el accept/reject greedy. |
| `splits.py` | se mantiene | Eval/test estratificado + submuestreo aleatorio por iteración (seed por iteración → reproducible). |
| `tests/` | upgrade | Un test por módulo nuevo + smoke test de una iteración con LLM mockeado. Los tests de rubric actuales quedan verdes. |

**Reutilización:** `_format_errors`, `_diagnose`, `_propose` y la lógica de Goodhart del
`loop.py` actual no se botan — se mueven/refinan dentro de `mutate.py` y `loop.py`.

---

## 5. Manejo de errores y guardas

| Situación | Comportamiento |
|---|---|
| Diff no aplica (SEARCH no encontrado) | No-op **logueado**; el genoma no se agrega al archivo. |
| LLM/JSON falla al parsear | Extracción vacía + retry con backoff (`extractor.py:127`); el fallo se cuenta. |
| Genoma degenerado (`recall_rel < piso`) | Fitness penalizado, descalificado como campeón, pero conservado en el archivo (stepping stone). |
| Verificación agéntica produce vacío/error | Se queda con la extracción pre-verificación (fail-safe). |
| Gap Goodhart `eval − test > 0.10` | Warning; el championship usa `test` para no coronar un campeón sobre-ajustado. |
| Budget (USD/tokens) o plateau | Para el loop (guardas actuales se mantienen). |
| Reproducibilidad | `validation`/`elo`/`fitness` puros; seed por iteración para el submuestreo. |

---

## 6. Estrategia de testing

- **Unit (sin API):** reglas de validación (substring filter, swap de dirección, dedup,
  polarity_consistency); matemática de ELO + distribución de `sample_parent`; `fitness`
  (piso de recall descalifica, costo penaliza); aplicación de diff (aplica / no-op).
- **Integración (LLM mockeado):** una iteración completa sobre un fixture mínimo → skirmish
  actualiza ELO, el archivo crece, el championship corona. Reusa `conftest.py`.
- **Regresión:** los tests actuales de `rubric` quedan verdes.
- **Smoke real (manual, opcional):** 2-3 iteraciones contra el gold real con budget mínimo
  antes de una corrida larga.

---

## 7. Métricas de éxito

- **Primaria:** `Precision_rel` y `Precision_ent` suben respecto al baseline (Prec_rel 0.20),
  manteniendo `recall_rel` por encima del piso.
- **Secundaria:** `F0.5_rel` sube y rompe el plateau de score ~0.52.
- **Costo:** tokens/artículo del campeón ≤ baseline (idealmente menor, vía Artefacto B
  determinista y prompts más magros).
- **Anti-Goodhart:** gap `eval − test` se mantiene < 0.10 en el championship.

---

## 8. Fuera de alcance (YAGNI)

Grid MAP-Elites, Retrospective Memory (BM25+FAISS), MCGS, y co-evolución sin labels (MAE)
quedan como stubs/hooks para cuando el archivo único se estanque. No se construyen ahora.
