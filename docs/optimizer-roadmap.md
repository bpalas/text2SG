# Roadmap del optimizer — text2graph-evolve

> Documento "empezar aquí" para retomar la optimización del extractor de relaciones
> políticas chilenas. Última actualización: 2026-06-15.

## Estado actual

**Champion de deployment: `id15`** (en `results/synthetic/genomes/id15_champion.json`).
Modelo `gemini-3.1-flash-lite`, prompt-only sobre arquitectura `given_entities`.

| métrica | valor | vs champion previo (id3) |
|---|---|---|
| f05 (precisión 2:1) | **0.928** | 0.894 |
| Precision_rel | **0.940** | 0.904 |
| Recall_rel | 0.884 | 0.857 |
| alucinaciones en distractores | **3** | 11 |
| pair-recall (undirected) | 0.919 (41 pares perdidos) | 0.888 (57) |

> ⚠️ **Todo medido en `train`.** Falta validar en `test` (ver Próximos pasos #1).

El prompt id15 combina tres estrategias que rompieron el techo: **estructura en inglés**
(mejor recall) + **encabezado de abstención precision-first** (menos alucinaciones) +
**2 ejemplos few-shot** de casos límite (1 oblicuo a no perder, 1 co-ocurrencia a rechazar).

## Lección principal (la más transferible)

**El cuello casi nunca está donde crees; mídelo antes de optimizarlo, y define la métrica
como tu verdadera pérdida — eso rinde más que cualquier algoritmo evolutivo.**

Evidencia de esta sesión:
- La **cascada de 3 checks** (entidades / par / dirección) reveló que la dirección era el
  ~1.8% del problema, no el cuello — tras gastar un workflow entero persiguiéndola. *Medir
  bien cambió qué optimizar.*
- La **métrica definió al ganador**, no la búsqueda: los mismos candidatos, re-rankeados con
  "precision-first + piso + castigo a alucinaciones", produjeron otro champion.
- La maquinaria sofisticada (Pareto por instancia, win-count, merge por componente) **sub-rindió**
  frente a lo simple: medir la cascada, alinear la métrica, y pedirle a Opus que unifique 3
  buenos prompts. El champion id15 salió de **síntesis LLM**, no del merge de Pareto.
- Corolario incómodo: la intuición (aun informada) falló repetido — inglés ranqueado último
  (ganó), dirección apostada (era ruido). **Tests baratos + descomposición correcta > teoría.**

Implicación de diseño (pendiente): el loop debería tener una **meta-reflexión disparada por
estancamiento** (cada K iteraciones sin mejora del frente) que cuestione la medición y el
objetivo — no en cada iteración (eso mueve la portería y nada es comparable), sino cuando se
aplana. Hoy ese paso lo hace el humano.

## Qué funcionó y qué no (no repetir los callejones)

**Funcionó** ✅
- **Demostraciones (few-shot)** y **reescritura estructural** (inglés, abstención) — rompieron el techo donde las reglas abstractas fallaron.
- **Métrica alineada a la pérdida**: precisión-inclinada + piso de recall + castigo a alucinaciones.
- **Merge por componente** (cuando los padres mutaron artefactos distintos A/B/C).
- **Selección win-count por instancia** (GEPA) — rompió el anclaje en la semilla.

**No funcionó** ❌ (evitar)
- **Filtros de precisión** (`min_confidence`, `require_both_in_quote`, subir `min_quote_len`) → colapsan recall (id4/id5: P 0.95, R 0.45).
- **Reglas abstractas de dirección** → la dirección ya está ~98% resuelta (solo ~8 pares invertidos); no es el cuello. Era un espejismo del marco directed/undirected.
- **Perseguir un escalar único** sin diversidad — el loop converge en 1 iteración.

**El cuello real**: pair-recall — los **41 pares oblicuos/mixtos que el modelo nunca encuentra** (fraseo indirecto). Las entidades (F1 1.0, `given_entities` las regala) y la dirección (0.98) están resueltas.

## Próximos pasos (prioridad)

1. **[GATE] Validar id15 en `test` (held-out).** Todo es train; el few-shot sale de artículos de train → riesgo de overfitting. Si aguanta, shippear.
   ```bash
   python scripts/synth_run_model.py --genome results/synthetic/genomes/id15_champion.json \
       --model gemini-3.1-flash-lite --dataset v1 --split test
   ```
2. **Empujar pair-recall**: más demostraciones few-shot apuntadas a los fraseos oblicuos/mixtos perdidos (la palanca con headroom real).
3. **Diversificar a artefactos B/C + merge**: mutar validación (B) o analysis (C), no solo prompt (A), para que `merge` combine "mejor prompt + mejor config".
4. **Tests del fix de dirección en `validation.py`** (el `_maybe_swap_direction` de recepción escrito por un agente — cobertura parcial vía `_shares_token`).
5. **Opcional — tesis gemini-2.5-flash-lite**: recall de pares 0.951 (techo más alto que 3.1) + filtro de precisión. Archivado por preferencia precision-first, pero el techo de recall está medido.

## Cómo correr el loop (Pareto-reflexivo + GEPA)

```bash
# elegir padre (win-count GEPA) -> vuelca genoma + diagnósticos + ejemplos
python scripts/synth_evolve.py pick --model gemini-3_1-flash-lite --select gepa --seed N

#  [reflexión Opus: lee *_expand_{genome,diag,examples}.json -> escribe candidato]

# evaluar con el modelo real
python scripts/synth_run_model.py --genome CAND.json --model gemini-3.1-flash-lite --split train --out DIR

# registrar (auto-tag de gradientes directed/undirected vs el padre)
python scripts/synth_evolve.py add --model gemini-3_1-flash-lite --genome CAND.json --preds DIR/preds.json --parent ID

# merge por componente del mejor par del frente
python scripts/synth_evolve.py merge --model gemini-3_1-flash-lite

# ranking precision-first (el que decide ganador)
python scripts/synth_evolve.py rank --model gemini-3_1-flash-lite --beta 0.5 --recall-floor 0.80

# estado: frente, campeones de gradiente, win-counts
python scripts/synth_evolve.py frontier|gradients --model gemini-3_1-flash-lite
```

Workflows multi-agente (Opus) listos en `scripts/`: `prompt_strategies_round.workflow.js`,
`combine_winners.workflow.js`, `gepa_direction_loop.workflow.js`.

## Parámetros abiertos (tunables)

- **`recall_floor = 0.80`** (fitness.py): piso duro; bajo esto, descalificado. Subir si se quiere más cobertura mínima.
- **`beta = 0.5`** (rank): precisión 2:1. `0.33` = 3:1 (más precision-first).
- **`DISTRACTOR_FP_PENALTY = 0.02`** (fitness.py): castigo por alucinación en distractor. Es **decisivo** en el ranking (define id14 vs id13); recalibrar tras ver `test`.

## Mapa de archivos

| qué | dónde |
|---|---|
| Genoma (A prompt + B validation + C analysis) | `swarm_optimizer/genome.py` |
| Archivo Pareto + win-count + merge + linaje | `swarm_optimizer/pareto.py` |
| Métrica precision-first (`selection_score`) | `swarm_optimizer/fitness.py` |
| Merge por componente + meta-prompt GEPA | `swarm_optimizer/mutate.py` |
| Métricas undirected + cascada | `swarm_optimizer/rubric.py` |
| Clasificador formal/informal | `swarm_optimizer/subsets.py` |
| CLI del loop (pick/add/merge/rank/gradients) | `scripts/synth_evolve.py` |
| Eval con API real | `scripts/synth_run_model.py` |
| Archivo evolutivo (datos, no versionado) | `results/synthetic/pareto/<model>.json` |
| Champion de deployment | `results/synthetic/genomes/id15_champion.json` |

Memoria persistente del proyecto: `[[gradient-loop-convergence]]` (cargada en cada sesión vía MEMORY.md).
