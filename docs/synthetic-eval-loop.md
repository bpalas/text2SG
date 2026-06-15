# Runbook — Loop de evaluación + mejora sobre el oráculo sintético

Loop estilo RoboPhD: un agente **Opus** da feedback y propone mutaciones (el "Evolution
Agent"); un modelo barato **ejecuta** la extracción. Oráculo = sintético con verdad plantada
(`results/synthetic/v1/`, 200 art). Métrica = `f05_rel` (precisión 2:1). Árbitro final = test
real held-out.

## Artefactos (genoma)
- **A** prompt — `genome.prompt_text` (semilla en `config.py`).
- **B** ValidationConfig determinista — `validation.py` (post-proceso $0).
- **C** Analysis Tool determinista — `analysis.py` (pre-extracción $0).
- Genomas baseline: `python scripts/make_synth_genomes.py` → `results/synthetic/genomes/*.json`.

## Ejecutar la extracción — dos caminos

### A) Modelo de API real (Gemini / OpenAI / Claude) — una línea, scripteable
```
# smoke barato primero (5 art), luego full
python scripts/synth_run_model.py --genome results/synthetic/genomes/seed.json \
    --model gemini-2.5-flash-lite --dataset v1 --split train --limit 5
python scripts/synth_run_model.py --genome ... --model gpt-4o-mini --dataset v1 --split train
```
Imprime P/R/F1/f05/Act_acc + tokens + costo aprox. Backend automático por prefijo del modelo
(`gemini-*` / `gpt-*`,`o*` / `claude-*`). Keys en `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`).

### B) Swarm de subagentes Claude — cero API (sobre la quota del plan)
```
python scripts/synth_agent_eval.py dump --genome G --dataset v1 --split train --out run/X --chunk-size 8
# despachar subagentes (Agent tool) que leen run/X/chunks/chunk_NN.json y escriben preds_NN.json
python scripts/synth_agent_eval.py collect --out run/X --dataset v1   # merge + redo.json auto
# despachar redo si redo>0; repetir collect
python scripts/synth_agent_eval.py score --genome G --dataset v1 --split train --preds run/X/preds.json
```

## Loop de mejora (RoboPhD)
1. **Baseline**: correr config(s)/modelo(s) → registrar f05 + desglose.
2. **Diagnose+Propose** (agente OPUS): extraer FN/FP reales (relaciones perdidas / inventadas en
   distractores) y dárselos a un subagente Opus que propone un genoma nuevo (prompt A y/o flags de
   C). Es el Evolution Agent del paper.
3. **Re-evaluar pareado** (mismos artículos) → aceptar si sube `f05_rel` (verdad perfecta → delta
   real). Barridos de B son gratis (re-`score` sin re-extraer).
4. **Confirmar** en synth-test (49 held-out) y en el **test real** (anti-circularidad).

## Monitoreo
- `python scripts/synth_loc.py` → LoC prompt vs tools por config (métrica RoboPhD).
- `score` emite `by_dominio` / `by_medio` / `by_dureza` + `distractor_fp`.
- Reporte acumulado: `results/synthetic/runs/v1_baseline_report.md`.

## Grilla próxima ronda (modelos livianos)
`gemini-2.5-flash-lite`, `gemini-3.x-flash-lite`, `gpt-4o-mini`, `gpt-5.x-mini`, `gpt-5.x-nano`.
**Verificar los IDs exactos contra la API antes de gastar** (algunos nombres pueden diferir).
Costo estimado: ~200 art × ~2K tokens ≈ 300-400K tokens/run → bastante < USD 1 por modelo a
precios de flash-lite / mini. Hacer `--limit 5` por modelo primero para validar ID + costo.

## Loop Pareto-reflexivo (synth_evolve)

Selección por **frente de Pareto (P,R) por tier**, sin ELO. El frente se persiste en
`results/synthetic/pareto/<model>.json`. Sembrarlo con los baselines ya medidos:
```
python scripts/synth_evolve.py add --model haiku --genome results/synthetic/genomes/seed.json \
  --preds results/synthetic/runs/v1_seed/preds_haiku.json --dataset v1 --split train
python scripts/synth_evolve.py add --model haiku --genome results/synthetic/genomes/analysis.json \
  --preds results/synthetic/runs/v1_analysis/preds_haiku.json --dataset v1 --split train
```
Una ronda reflexiva:
1. `python scripts/synth_evolve.py pick --model M --dataset v1 --split train`
   → escribe `<M>_expand_genome.json` (miembro menos expandido del frente) y `<M>_expand_diag.json`
   (FN por dureza + FP en distractores).
2. **Reflexión (Opus):** despachar un subagente Opus con esos dos archivos → propone un genoma nuevo
   (edita prompt A y/o flags de `AnalysisConfig` C, condicionado al modelo) → guardar `cand.json`.
   (Es el Evolution Agent; orquestado por el controller con checkpoint, no automático en v1.)
3. **Extraer + puntuar** `cand.json` con el executor:
   - API: `python scripts/synth_run_model.py --genome cand.json --model M --dataset v1 --split train`
   - cero-API: `dump → swarm → collect → score` (sección anterior) → `preds.json`.
4. `python scripts/synth_evolve.py add --model M --genome cand.json --preds preds.json \
      --dataset v1 --split train --parent <picked_id>`
   → entra al frente si no es dominado.
5. Repetir 2-3 rondas; `frontier --model M` para ver el estado; regenerar el gráfico de Pareto.
Árbitro: confirmar los mejores del frente en `--split test` (49 held-out) y en el test real.

## Seguridad
`.env` está gitignoreado. NUNCA commitear keys. Rotarlas si se expusieron.
