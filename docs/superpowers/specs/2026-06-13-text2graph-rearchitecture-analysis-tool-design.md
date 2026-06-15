# Re-arquitectura text2graph estilo RoboPhD — el Analysis Tool determinista

**Fecha:** 2026-06-13
**Rama:** `feat/evolutionary-optimizer-redesign`
**Estado:** Diseño aprobado, pendiente plan de implementación
**Spec previa relacionada:** `2026-06-09-text2graph-evolve-roadmap-design.md`

## 1. Contexto y motivación

El sistema lleva congelado en el campeón `id=4` (una **semilla**, `championship_score=0.5427`)
desde el inicio: 0 de 6 mutaciones produjeron mejora. El techo de precisión medido en serio
está en ~0.40–0.46 (`P_rel`) y ~0.41 (`F1`). El diagnóstico contra el paper RoboPhD
(text-to-SQL autoevolutivo) identificó que importamos su maquinaria (ELO, 2 artefactos, loop)
pero **invertimos su apuesta ganadora**:

- En RoboPhD la inteligencia evolucionada vive en un **script Python determinista** (~1000
  líneas: análisis de schema, cardinalidad, valores enum, profundidad adaptativa). El paper
  dice textual: *"a deterministic target is easier to optimize in an evolutionary cycle."*
- En nuestro sistema toda la inteligencia vive en el **prompt A (~25 líneas)**, que es frágil
  (al mutarse rompe el JSON → `Recall=0` → penalización −1.0), y el artefacto determinista B
  (`ValidationConfig`) son ~15 líneas de 9 flags — casi sin dónde crecer.

Hoy el `given_entities` inyecta el roster como **6 líneas planas** (una por actor),
[extractor.py:73-79](../../../swarm_optimizer/extractor.py). Ese es exactamente el lugar donde
RoboPhD metió 1000 líneas de análisis. Esta spec corrige la inversión: introduce un artefacto
determinista que carga la inteligencia, fiel al modelo que en RoboPhD dio el mejor puntaje.

## 2. Objetivo y no-objetivos

**Objetivo:** Construir la arquitectura de 3 artefactos con un **Analysis Tool determinista**
(artefacto C) que produce un bloque estructurado de análisis de actores+artículo, reemplazando
la inyección plana. Medir su efecto con un reward limpio (oráculo sintético + test pareado) y
trackear el crecimiento de líneas de código (la métrica narrativa de RoboPhD).

**No-objetivos (siguiente spec, "desestancar"):**
- Que el loop evolutivo *evolucione* el artefacto C vía el bandit Thompson.
- Fixes de infraestructura del loop (ELO que rankea cadáveres, persistencia de meta-policy/
  meta-review, robustez de mutaciones de A).
- Auditoría/curación adicional del gold.

El artefacto C se entrega **hand-authored v1**. Su evolución automática es trabajo posterior.

## 3. Arquitectura: 3 artefactos

Mapeo fiel a RoboPhD (`DatabaseAnalysis ⊕ EvalInstructions ⊕ Question ⊕ Evidence`):

| RoboPhD | text2graph | Rol |
|---|---|---|
| DatabaseAnalysis (tool Python) | **ActorAnalysis (artefacto C)** | determinista, crece, carga hechos por-artículo |
| EvalInstructions | ExtractionInstructions (prompt A) | reglas generales, evoluciona |
| Question (variable) | Artículo (`body`) | input variable |
| Evidence (hints) | Roster (`union`) | dado |

**División de responsabilidades:** las reglas *generales* (dirección, defensa, polaridad-por-
act_type) se quedan en el prompt A. Los *hechos por-artículo* (alias de este actor, rol probable,
pares candidatos) los computa el Analysis Tool. Igual que RoboPhD: ambos crecen, los hechos van
al tool.

## 4. El Analysis Tool (artefacto C) — `swarm_optimizer/analysis.py`

### Interfaz

```python
@dataclass
class AnalysisConfig:
    """Artefacto C: análisis determinista pre-extracción (costo $0)."""
    emit_dossier: bool = True
    emit_alias_map: bool = True
    emit_role_hints: bool = True
    emit_direction_scaffold: bool = True
    emit_main_speaker: bool = True
    emit_comention_pairs: bool = True
    emit_act_type_canon: bool = True
    emit_domain_gate: bool = True
    role_keywords: dict[str, list[str]] = field(default_factory=lambda: DEFAULT_ROLE_KEYWORDS)
    role_window: int = 80   # ventana ±chars alrededor de cada mención para detectar rol

def build_analysis(union: dict, body: str, cfg: AnalysisConfig) -> str:
    """Devuelve el bloque de análisis estructurado (texto) para inyectar en el prompt."""
```

### Secciones de salida (deterministas)

Cada sección es gateada por su flag en `AnalysisConfig` (para aislar su efecto en evaluación):

1. **Dossier de actores** — por actor (excluyendo `type == "NIL"`): nombre canónico
   (`canonical_names[0]`), todos los alias (`surfaces`), tipo. Fuerza uso de canónicos.
2. **Mapa de alias** — `surface → canónico`, normalización determinista de menciones.
3. **Hints de rol (heurística regex)** — para cada actor, busca palabras de rol
   (`abogado`, `defensa`, `imputado`, `ministro`, `fiscal`, `diputado`, `senador`, `presidente`…)
   en ventanas de ±80 caracteres alrededor de cada mención de sus surfaces → etiqueta rol probable
   (ventana configurable en `AnalysisConfig.role_window`, default 80).
   **Caso crítico (bug Hermosilla):** si dos actores comparten apellido, marca la ambigüedad
   explícitamente y adjunta el contexto que los distingue (`Luis Hermosilla [imputado]` vs
   `Juan Pablo Hermosilla [abogado defensor]`).
4. **Andamiaje de dirección/voz** — detecta construcciones pasivas (`"<A> fue <participio> por <B>"`,
   patrón ` por `) y emite hints de dirección explícitos (`from=B, to=A`). Complementa —no
   reemplaza— `normalize_passive_direction` de B (post-proceso).
5. **Hablante principal** — heurística: actor en el titular / primer citado / más mencionado →
   marca como sujeto probable de varias relaciones. Ataca la omisión sistemática del gold y el
   FP de atribución.
6. **Pares co-mencionados** — pares de actores que aparecen en la misma oración (candidatos
   a relación, sin afirmar que exista).
7. **Canonicalización de act_type** — vocabulario canónico (los 10 tipos) + mapa de los ~20%
   tipos no-canónicos del gold (`kill→attacks`, `defends→endorses`, `criticizes→accuses`,
   `meet_with→negotiates_with`, `appoints→endorses`…). Mejora `Act_acc` sin tocar el extractor.
8. **Gate de dominio no-político** — flag heurístico de contexto fútbol/extranjero (debilidad
   medida en sintético: fútbol P 0.435).

### Cableado en `build_prompt`

Nuevo campo en `Genome`: `analysis: AnalysisConfig | None = None`. En
[extractor.py `build_prompt`](../../../swarm_optimizer/extractor.py), cuando `genome.analysis`
está presente (y el `union` no está vacío), se inserta `build_analysis(union, body, cfg)` en
lugar de la lista plana de `given_entities`. Backward-compatible: sin `analysis`, el
comportamiento actual no cambia.

`Config` (legacy, en config.py) y `Genome` (genome.py) ambos necesitan el campo para que el
extractor lo lea vía `getattr(config, "analysis", None)`.

## 5. Evaluación — para que "mejor puntaje" signifique algo

El test real tiene ruido ±0.02–0.11; no se puede afirmar mejora sin reward limpio. Medimos en
dos planos, **pareado** (mismos artículos, config nueva vs campeón):

- **Oráculo sintético** (`results/synthetic/`, verdad plantada, deltas resolubles) — primario
  para iterar. Pipeline existente: `synth_sample_guiones.py → workflow → synth_assemble_and_eval.py`.
- **Test held-out** (split de test, ~n=30) — confirmación, pareado vs campeón actual.

**Tres configs comparadas** para aislar el efecto y encontrar el mejor puntaje:

| Config | Qué mide |
|---|---|
| `semilla` (actual, given_entities plano) | baseline |
| `semilla + Analysis Tool` | efecto puro del andamiaje determinista |
| `semilla + Analysis Tool + verify` | combinación de mejor puntaje (verify ya midió +0.06 P_rel) |

Criterio de adopción (de la auditoría): deltas de una sola corrida <0.05 son ruido en test;
promediar ≥2 corridas o usar comparación pareada con significancia. En sintético los deltas son
limpios por construcción.

## 6. Medición de líneas de código

`scripts/measure_loc.py`: cuenta líneas de A (prompt), B (ValidationConfig), C (analysis.py +
config), y el tamaño promedio del bloque de análisis generado por artículo. Imprime la tabla
estilo RoboPhD (naive ~40 → objetivo) para trackear crecimiento. Es la métrica que el usuario
pidió evaluar explícitamente.

## 7. Testing

Unit tests de `analysis.py` (`swarm_optimizer/tests/test_analysis.py`):
- **Caso Hermosilla:** union con dos actores mismo apellido → la salida marca la ambigüedad y
  adjunta contexto distintivo.
- **Detección de pasiva:** `"Boric fue criticado por Matthei"` → hint `from=Matthei, to=Boric`.
- **Mapa de alias:** surface conocido → canónico correcto.
- **Canonicalización de act_type:** `kill → attacks`, etc.
- **Gating de secciones:** cada flag de `AnalysisConfig` en False omite su sección.
- **Golden test:** 2 artículos conocidos → snapshot del bloque de análisis.

Mantener verde la suite existente (`python -m pytest swarm_optimizer/tests/ -q`).

## 8. Criterios de éxito

1. `build_analysis` produce un bloque estructurado correcto sobre el `union` real (tests verdes).
2. La config `semilla + Analysis Tool` **no empeora** el test held-out pareado, y mejora en
   sintético (o se documenta honestamente que no, con números).
3. `measure_loc.py` reporta la tabla de LoC de los 3 artefactos.
4. El extractor es backward-compatible (sin `analysis` = comportamiento idéntico).
5. La combinación `+ verify` se mide y se reporta como candidata a mejor puntaje.

## 9. Riesgos y mitigaciones

- **El registro no tiene rol fino** (`type` es grueso: roster/institutional/non_roster). Mitigación:
  los hints de rol se *derivan del texto del artículo* por heurística regex, no del registro.
  Si la heurística no alcanza, la arquitectura permite enchufar una desambiguación LLM puntual
  (decisión basada en el puntaje medido, no a priori — RoboPhD §5.2 "middle ground").
- **El bloque de análisis infla tokens** → sube el costo y baja el `score` (penalización de costo
  en `fitness`). Mitigación: secciones gateadas; medir tokens/artículo en la evaluación.
- **Sobreajuste al sintético.** Mitigación: el test held-out es el juez final; el sintético solo
  prioriza qué llevar al test.

## 10. Archivos afectados

- **Nuevo:** `swarm_optimizer/analysis.py`, `swarm_optimizer/tests/test_analysis.py`,
  `scripts/measure_loc.py`.
- **Modificado:** `swarm_optimizer/genome.py` (campo `analysis`), `swarm_optimizer/config.py`
  (campo en `Config` legacy), `swarm_optimizer/extractor.py` (`build_prompt`).
- **Eval:** script de comparación de las 3 configs sobre sintético + test (reusa harness existente).
</content>
</invoke>
