# Instrucciones para Claude Code — text2graph-evolve

## Contexto del proyecto

Sistema evolutivo que mejora un extractor de relaciones políticas chilenas.
Evoluciona dos artefactos: prompt (A) y ValidationConfig determinista (B).
Cuello de botella actual: Precision_rel ~0.06–0.22.

## Comandos frecuentes

```bash
python -m swarm_optimizer.run --probe          # verificar pipeline (~$0.001)
python -m swarm_optimizer.run --iterations 20 --budget 8.0
python scripts/report_run.py                   # reporte de la última corrida
python scripts/propose_mutations.py            # contexto para proponer mutaciones
python -m pytest swarm_optimizer/tests/ -q    # 111 tests
```

## Cuando el usuario pide "propone mutaciones"

1. Leer `results/swarm/history.jsonl` con `from swarm_optimizer.archive import Archive`
2. Obtener el campeón: `arc.champion()`
3. Revisar métricas del campeón (`championship_score`, `metrics`)
4. Revisar diagnósticos del linaje (`arc.lineage(champ.id)` → campos `diagnosis`, `fitness_delta`)
5. Revisar el meta_review más reciente si existe
6. Proponer mutaciones concretas (cambio mínimo, una cosa por vez)
7. Agregarlas a `docs/mutation-proposals.md` con el checklist pre-llenado

O simplemente correr: `python scripts/propose_mutations.py` para obtener el contexto completo.

## Cuando el usuario pide "analiza la corrida"

Correr: `python scripts/report_run.py -o results/swarm/reporte_FECHA.md`
Luego leer el reporte y resumir: qué operador rindió más, si hay acumulación en el piso
de recall, el gap Goodhart, y qué propuesta del backlog parece más relevante probar.

## Estructura de archivos importantes

- `swarm_optimizer/genome.py`     — Artefacto A (prompt) + Artefacto B (ValidationConfig)
- `swarm_optimizer/loop.py`       — loop evolutivo principal
- `swarm_optimizer/mutate.py`     — diagnose / propose / cross / fresh / meta_review
- `swarm_optimizer/archive.py`    — archivo open-ended con linaje y memoria
- `swarm_optimizer/meta_policy.py`— bandit Thompson sobre operadores
- `docs/mutation-proposals.md`    — backlog de mutaciones a probar
- `results/swarm/history.jsonl`   — archivo evolutivo (ignorado por git)

## Lo que NO hacer

- No versionar `results/swarm/history*.jsonl` ni `best_config.json` (están en .gitignore)
- No cambiar el gold standard split (eval/test) entre corridas — rompe la comparabilidad
- No subir `min_quote_len` por encima de 25 sin verificar que Recall_rel no colapsa
- No tocar el archivo evolutivo manualmente — usar solo la Archive API
