# Eval + mejora sobre el oráculo sintético vía swarm de subagentes (Workflow), cero API

**Fecha:** 2026-06-14
**Rama:** `feat/evolutionary-optimizer-redesign`
**Estado:** Diseño aprobado, pendiente plan de implementación
**Specs relacionadas:** `2026-06-13-text2graph-rearchitecture-analysis-tool-design.md` (Analysis Tool / artefacto C)

## 1. Contexto y motivación

El diagnóstico identificó que el cuello #1 es el **reward ruidoso**: el gold real (n≈30 test) tiene
varianza corrida-a-corrida ±0.02–0.11, así que una mejora real de +0.02 del andamiaje determinista
queda invisible bajo el ruido. El **benchmark sintético** (`results/synthetic/v1/`, 200 art, 723 rels
plantadas, 30 distractores) tiene **verdad por construcción (100% control)** → los deltas chicos SÍ
se resuelven. Hoy ese sintético NO está conectado al optimizador: es un pipeline separado que evalúa
un genoma viejo (p017_attrib) una vez.

Además, el usuario quiere **no gastar API de Gemini**: el extractor pasa a ser un **swarm de
subagentes Claude** (Agent tool / Workflow, sobre la quota del plan — como RoboPhD usa Claude Code
como agente, sin costo directo de API).

Esta spec conecta ambas cosas: evaluar el extractor (incluido el Analysis Tool) sobre el oráculo
sintético con subagentes Claude, y un **loop de mejora estilo RoboPhD** donde los subagentes son
tanto el extractor como el agente de evolución.

## 2. Objetivo y no-objetivos

**Objetivo:** (1) Un harness que evalúa cualquier genoma sobre el sintético usando un swarm de
subagentes Claude (cero API), y (2) un loop de mejora que diagnostica errores, propone un genoma
mejor, y re-evalúa, corriendo varias rondas para subir el puntaje sobre verdad de control total.

**No-objetivos:**
- Integrar esto al loop evolutivo con ELO/bandit de `loop.py` (ese es otro rediseño).
- Endurecer/regenerar el dataset sintético (v2 con recall no saturado) — se usa v1 tal cual.
- Gastar API de Gemini o Anthropic metered. Todo extractor/evolución = subagentes Claude.

## 3. Arquitectura general

Patrón **dump → (swarm extrae) → collect → score** reusando el de [agent_eval.py](../../../scripts/agent_eval.py),
re-apuntado al sintético, más el loop de mejora:

```
genoma_k ──dump (Python)──▶ prompts.json
                                │  args
                                ▼
                  [Workflow: fan-out de subagentes extractores] ──▶ extracciones (structured)
                                │  el controller escribe
                                ▼
                          preds.jsonl ──score (Python)──▶ métricas vs gold plantado (100% control)
                                                              │  FP/FN
                                                              ▼
                          [subagente diagnose] ─▶ [subagente propose: genoma_k+1] ─▶ (repetir)
```

**División de responsabilidades (restricción dura del Workflow):** los scripts de Workflow NO tienen
filesystem ni Python. Por eso:
- **Determinista y en Python (scripts):** `dump` (construye prompts con `build_prompt`, que ya incluye
  el bloque del Analysis Tool), `score` (validación B + `compute_metrics` + desgloses). Costo $0.
- **LLM y fan-out (Workflow):** la extracción masiva. Los prompts entran por `args`; los agentes
  devuelven `{article_id, entities, relations}` (structured output); el Workflow retorna la lista
  agregada; el **controller** la escribe a `preds.jsonl`.
- **Orquestación de rondas:** el controller (no el Workflow) encadena dump → workflow → score →
  diagnose/propose → guardar genoma → repetir. Cada ronda lanza un Workflow de extracción.

## 4. El harness sintético — `scripts/synth_agent_eval.py`

Adaptado de `agent_eval.py`, apuntado a `results/synthetic/<dataset>/` (default `v1`):

- `dump --genome g.json --dataset v1 --out run/r0 [--split train]`
  - Carga `articles.parquet`, `unions.json` (dict, NO YAMLs — diferencia clave vs agent_eval).
  - Filtra al split (ver §6) si se pasa `--split`.
  - Para cada artículo: `build_prompt(genome, body, union, [])` → acumula en `out/prompts.json` =
    `[{"article_id": id, "prompt": "..."}, ...]` y `out/ids.json`.
- `score --genome g.json --dataset v1 --out run/r0 [--preds preds.jsonl] [--label name]`
  - Carga `gold.parquet`, `unions.json`, `articles.parquet`.
  - Aplica `apply_validation(..., genome.validation)` sobre las preds crudas, luego `compute_metrics`.
  - Imprime JSON con métricas globales + **desglose por dominio/registro/dureza** + **disciplina en
    distractores** (FP en artículos con 0 relaciones), como ya hace `synth_assemble_and_eval`.
- `collect` (opcional, para compat con respuestas en archivos): `responses/<id>.txt` → `preds.jsonl`.
  En el flujo Workflow no se usa (el controller escribe `preds.jsonl` directo desde el resultado).

Reutiliza sin cambios: `build_prompt`, `parse_llm_output`, `apply_validation`, `compute_metrics`.

## 5. El swarm de extracción — Workflow

Un Workflow recibe en `args` la lista `[{article_id, prompt}, ...]` (de `prompts.json`) y hace
**fan-out por lotes** (~15 artículos/agente → ~14 agentes para 200, dentro del cap de concurrencia).
Cada agente:
- Recibe su lote de prompts.
- Para cada uno actúa de extractor de relaciones y devuelve `{article_id, entities, relations}`
  validado con un `schema` (StructuredOutput → sin parseo frágil).
- El Workflow agrega y retorna la lista completa.

El **controller** escribe `preds.jsonl` desde el resultado del Workflow y corre `score`. **Cero API**
(los agentes son Claude sobre la quota del plan). El Workflow corre en background; reintenta los
artículos faltantes en una segunda pasada si algún agente muere.

## 6. Anti-sobreajuste — split sintético

El sintético no tiene split → riesgo de sobreajustar el prompt al oráculo. Mitigación:
- **Split estratificado** por `dominio`×`registro` guardado en `results/synthetic/v1/split.json`
  (~75% train ≈ 150, ~25% test ≈ 50, semilla fija). Helper `load_synth_split(dataset)`.
- Se **mejora sobre synth-train**, se **confirma sobre synth-test** (solo se mira al final de cada
  ronda aceptada).
- **Árbitro final opcional, también sin API:** un swarm de subagentes sobre el **test real held-out
  (n≈30)** del gold real — el chequeo anti-circularidad. Mismo harness, `--dataset real`.

## 7. El loop de mejora (RoboPhD, subagente-driven)

Orquestado por el controller, repetible:

1. **Ronda 0 (baseline):** 3 configs — `semilla` / `+analysis` / `+analysis+verify` — dump→workflow→score
   sobre synth-train. Se registran métricas y se vuelcan los FP/FN por artículo.
2. **Diagnose:** un subagente lee los FP/FN + el gold plantado de los peores artículos y devuelve
   3-5 patrones de error (structured).
3. **Propose:** un subagente recibe el genoma actual + el diagnóstico y devuelve `genoma_k+1`
   concreto (prompt A editado y/o flags/`role_keywords` del Analysis Tool C y/o ValidationConfig B).
4. **Re-eval:** dump→workflow→score del genoma nuevo sobre synth-train, **pareado** (mismos artículos;
   verdad perfecta → el delta es real, sin piso de ruido).
5. **Aceptar/revertir:** se acepta si sube el F0.5 (o el eje objetivo) más allá de 0; si acepta,
   se mide en synth-test. Se repite 2-3 rondas o hasta no mejorar.
6. **Barridos de B gratis:** como B es determinista, se prueban N ValidationConfig sobre las MISMAS
   preds crudas a costo $0 (re-`score` sin re-extraer).

Genomas y corridas se guardan en `results/synthetic/runs/<timestamp>/` (genoma_k.json, métricas,
preds, diagnósticos).

## 8. Testing

Unit tests (`swarm_optimizer/tests/test_synth_eval.py`): 
- carga de unions desde JSON (estructura `{id: {uid: {...}}}`),
- `dump` produce `prompts.json` con el bloque `=== ANÁLISIS DE ACTORES ===` cuando el genoma tiene
  `analysis`,
- `score` calcula métricas correctas sobre un mini-gold sintético fijo (2-3 artículos inline),
- `load_synth_split` es determinista y estratificado (train/test disjuntos, cubren todos los dominios).
Mantener verde la suite existente (146 tests).

## 9. Alcance / secuencia

1. **Smoke en pilot (20 art)** — validar dump→workflow(2 agentes)→score end-to-end antes de gastar
   quota en 200.
2. **Baseline en v1 (200)** — las 3 configs sobre synth-train, con desglose por dominio (el valor del
   sintético: categorías controladas, ej. fútbol P 0.435).
3. **2-3 rondas de mejora** sobre synth-train, confirmando en synth-test.

**Deliverables:** `synth_agent_eval.py` + `load_synth_split` + `results/synthetic/v1/split.json` +
el script de Workflow de extracción + genomas baseline + el runbook del loop ejecutado por subagentes.

## 10. Riesgos

- **Quota del plan:** 200 art × varias rondas = muchos subagentes. Mitigación: smoke en pilot;
  lotes de ~15/agente; barridos de B gratis; rondas acotadas (2-3).
- **Variabilidad de subagentes como extractor:** distintos subagentes pueden extraer distinto.
  Mitigación: `schema` estricto en la salida; prompt de extracción idéntico al dumped; el split
  test confirma robustez.
- **Sobreajuste al sintético:** synth-test + árbitro real held-out (§6) lo acotan.
- **Recall saturado del sintético v1** (los redactores expresan muy explícito): el eje real que mide
  es precisión; documentarlo al leer resultados (no celebrar recall alto).
- **Workflow sin FS/Python:** ya resuelto por diseño (prompts por `args`, extracciones por return,
  I/O en el controller).

## 11. Archivos afectados

- **Nuevo:** `scripts/synth_agent_eval.py`, `scripts/synth_extract_workflow.js` (o inline),
  `results/synthetic/v1/split.json` (generado), `swarm_optimizer/tests/test_synth_eval.py`.
- **Reutilizado sin cambios:** `swarm_optimizer/extractor.build_prompt`, `parse_llm_output`,
  `validation.apply_validation`, `rubric.compute_metrics`, `analysis.build_analysis`.
</content>
