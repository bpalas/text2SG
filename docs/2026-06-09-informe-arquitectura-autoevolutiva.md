# Informe técnico: Arquitectura autoevolutiva para el pipeline Text2Graph

**Fecha:** 2026-06-09
**Rama:** `feat/evolutionary-optimizer-redesign`
**Autoría:** Análisis conjunto (Investigador Principal IA / Arquitectura de Sistemas Multiagente)
**Insumos:** 8 papers (carpeta *Sistemas Multiagentes Evolutivos*) + estado actual del código (`swarm_optimizer/`, spec `docs/superpowers/specs/2026-06-09-text2graph-evolve-roadmap-design.md`)

---

## 0. Resumen ejecutivo

El repositorio ya **no está en fase de diseño**: el Tier 1 del rediseño evolutivo (estilo RoboPhD: genoma de 2 artefactos, ELO pairwise, archivo open-ended DGM, validación determinista, fitness F0.5 con piso de recall graduado) está implementado al 100% con 82 tests verdes y smoke test funcional sobre Ollama local. El cuello de botella diagnosticado sigue siendo **precisión de relaciones** (`Precision_rel ≈ 0.20`, `F1_rel ≈ 0.25`) con entidades y polaridad sanas (`Precision_ent ≈ 0.81`, `Polarity_acc ≈ 0.82`).

Este informe responde tres preguntas:

1. **Qué incorporar de la literatura** — y, con igual importancia, qué *no* incorporar todavía. Veredicto: el sistema actual es una instancia fiel de RoboPhD + DGM. Las incorporaciones de mayor ROI son (a) un **bandit/RL sobre el operador de mutación** (meta-agente ligero), (b) **memoria retrospectiva** explotando los campos `diagnosis`/`fitness_delta` que el archivo ya persiste, (c) **Meta-review agent** (AI Co-Scientist) que sintetiza patrones de error transversales, y (d) **evaluación en cascada** (AlphaEvolve) para abaratar iteraciones.
2. **Cómo se define el RL del meta-agente** — estados, acciones y recompensas concretas sobre la infraestructura existente. La recomendación crítica: **no hacer RL de pesos del LLM** (la spec ya lo descartó con razón); hacer RL *sobre la política de evolución* (qué mutar, con qué operador, desde qué padre), donde el problema es un bandit contextual de ~10 brazos con recompensa densa ya disponible (`fitness_delta`).
3. **Plan por fases** — Fase 1 cierra el MVP corriendo el loop contra Gemini con presupuesto real; Fase 2 instala el bucle de RL meta-nivel y la memoria; Fase 3 escala a co-evolución multiagente y open-endedness controlada.

---

## 1. Análisis tecnológico: qué incorporar y por qué

### 1.1 Mapa papers → componentes (estado actual)

| Paradigma (paper) | Mecanismo | Estado en el repo | Veredicto |
|---|---|---|---|
| **RoboPhD** (Text-to-SQL autoevolutivo) | 2 artefactos (tool determinista + instrucciones), ELO K=32, skirmish/championship, verificación agéntica | ✅ Implementado (`genome.py`, `elo.py`, `loop.py`, `extractor.py::verify_relations`) | Referencia principal; correcto |
| **Darwin Gödel Machine** | Archivo open-ended (nunca se borra), selección de padres ∝ sigmoid(ELO)/(1+hijos), stepping stones | ✅ Implementado (`archive.py`, `elo.py::sample_parent`) | Correcto |
| **AlphaEvolve** | Diffs SEARCH/REPLACE sobre bloques marcados, MAP-Elites + islas, **evaluación en cascada**, ensemble de LLMs | 🟡 Parcial: diff S/R sí (`mutate.py::apply_diff`); cascada, MAP-Elites e islas no | Incorporar **cascada** ya; MAP-Elites condicionado a plateau |
| **AI Co-Scientist** | Torneo ELO con debates, **Meta-review agent** (síntesis de patrones de error → feedback a prompts de todos los agentes), Evolution agent que *genera* en vez de mutar | 🟡 Parcial: ELO sí; Meta-review no; generación-desde-cero no | Incorporar **Meta-review** (Fase 2) |
| **MLEvolve** | Progressive MCGS (grafo, no árbol; cross-branch reference edges), **Retrospective Memory** (BM25+FAISS+RRF), Planner/Coder separados | 🟡 Parcial: cross-pollination ≈ cross-branch pobre; memoria no explotada | Incorporar **memoria retrospectiva v1** (sin FAISS al inicio) |
| **Multi-Agent Evolve (MAE)** | Trío Proposer-Solver-Judge co-evolucionando con Task-Relative REINFORCE++, quality filtering | ❌ No aplica directo (requiere entrenar pesos) | Adaptar solo el *patrón* Proposer (generador de casos difíciles) en Fase 3 |
| **Societies of Thought** | El razonamiento eficaz es debate interno multi-perspectiva; estructura conversacional > monólogo; SAE steering | ❌ No implementado | Adaptar como **prompt-pattern** (debate interno en el extractor), no como SAE (inviable sobre Gemini cerrado) |
| **MARL para DNN mapping** | Descomposición por clustering de correlación de parámetros; reward global compartido | ❌ No implementado | Idea menor: agrupar parámetros de `ValidationConfig` correlacionados al mutar |

### 1.2 Incorporaciones recomendadas (justificadas)

**(a) Evaluación en cascada (AlphaEvolve) — ROI inmediato, costo de implementación bajo.**
Hoy cada skirmish evalúa al hijo sobre K=12 artículos completos. AlphaEvolve demuestra que una cascada de gates (barato → caro) ahorra la mayor parte del compute: la mayoría de los mutantes son malos y se detectan con poco. Propuesta concreta: gate 0 = validación sintáctica del genoma (gratis, ya existe vía `noop`); gate 1 = skirmish sobre K=4 artículos, descartar si `fitness_child < fitness_champ − ε`; gate 2 = skirmish completo K=12 solo para sobrevivientes; championship sin cambios. Con la tasa típica de mutantes inútiles (~60-70% en DGM/RoboPhD), esto reduce el costo por iteración a ~la mitad, lo que con presupuesto fijo ($8) duplica las iteraciones efectivas. **Por qué:** el recurso escaso del sistema no es la calidad del operador de mutación sino el número de iteraciones que el presupuesto permite; la cascada compra iteraciones.

**(b) Meta-agente de evolución con RL ligero (bandit contextual) — la incorporación de RL correcta para este sistema.**
La literatura ofrece dos formas de RL: (i) RL de pesos del LLM (MAE, Societies of Thought con PPO/GRPO) y (ii) RL/búsqueda sobre la *política de evolución* (RoboPhD lo hace hand-crafted; MLEvolve con scheduling UCT; sus autores listan "meta-evolution" como future work explícito). La opción (i) está bien descartada por la spec: Gemini es API cerrada, el gold standard (~93 artículos) es demasiado pequeño para entrenar sin colapso, y el costo es de otro orden de magnitud. La opción (ii) es barata y ataca una debilidad real del loop actual: el calendario de operadores es **rígido** (`cross_every=7`, diff guiado el resto), elegido a mano y ciego al contexto. El archivo ya persiste la señal de recompensa (`fitness_delta` por mutación, con su `mutation_type` y `artifact_touched`): es exactamente el dataset de un bandit contextual. Diseño completo en §2. **Por qué:** convierte hiperparámetros muertos en una política que aprende qué operador rinde en qué régimen del problema, con cero costo de LLM adicional.

**(c) Memoria retrospectiva (MLEvolve / DGM) — explotar lo que ya se persiste.**
`ArchiveEntry` ya guarda `diagnosis` (causas de FP/FN diagnosticadas) y `fitness_delta`. Hoy `propose()` solo ve el diagnóstico del padre actual: cada mutación re-descubre lecciones que el archivo ya contiene ("subir `min_quote_len` no ayudó", "la regla de voz pasiva produjo +0.03"). Propuesta v1 (sin FAISS, YAGNI): al construir el prompt de `propose()`, inyectar (1) las 3 mutaciones con mayor `fitness_delta` positivo del linaje y (2) las 3 con delta más negativo, como "intentos previos: qué funcionó / qué no". Propuesta v2 (si el corpus de mutaciones supera ~100 entradas): retrieval híbrido BM25 + embeddings con RRF como hace MLEvolve. **Por qué:** MLEvolve muestra en ablación que la memoria es de los componentes de mayor impacto; aquí el costo marginal es ~0 porque los datos ya existen.

**(d) Meta-review agent (AI Co-Scientist) — síntesis transversal cada championship.**
El diagnóstico actual es *local* (FP/FN del padre en un submuestreo). El Meta-review de AI Co-Scientist hace lo que ningún diagnóstico local puede: detectar patrones **sistémicos** ("el 70% de los FP de relaciones son `co_occurs` espurios entre entidades del mismo párrafo sin verbo conector"). Propuesta: cada championship (cada M=5 iters), una llamada LLM adicional recibe los FP/FN agregados del top-3 y emite 3-5 patrones recurrentes; ese texto se inyecta en los prompts de `diagnose()`/`propose()` de las siguientes M iteraciones. Costo: 1 llamada extra cada 5 iteraciones. **Por qué:** ataca directamente el cuello de botella (Precision_rel) con la herramienta correcta — los errores de precisión de relaciones suelen ser de *clase* (patrones), no idiosincráticos.

**(e) Patrón "society of thought" como arquitectura de prompt del extractor — vía el propio loop evolutivo.**
Societies of Thought demuestra que estructurar el razonamiento como debate interno (proponer → objetar → reconciliar) mejora accuracy de forma causal, y que el formato conversacional supera al monólogo con los mismos contenidos. No podemos hacer SAE steering sobre Gemini, pero sí podemos sembrar una **variante de arquitectura** `architecture="debate"`: el extractor propone relaciones, luego una voz crítica interna las objeta ("¿hay verbo conector explícito?, ¿la evidencia soporta la dirección?") y solo sobreviven las reconciliadas. Importante: no imponerla — **añadirla como semilla adicional al archivo y dejar que ELO decida**. Es la forma epistémicamente honesta de incorporar el insight: como hipótesis competidora, no como decisión de diseño. **Por qué:** la verificación agéntica ya implementada (`verify_relations`) es un caso degenerado de esto (crítica post-hoc de 1 turno); el paper sugiere que el debate *durante* la extracción rinde más.

### 1.3 Qué NO incorporar (y por qué) — sección crítica

- **RL de pesos (PPO/GRPO/REINFORCE++ sobre el modelo extractor).** Requiere modelo abierto, miles de episodios y un reward model estable. Con 93 artículos gold, el riesgo de Goodhart/colapso es casi seguro. MAE entrena Qwen-3B con 300 steps × batch 128: ese régimen de datos no existe aquí. Reconsiderar solo si (Fase 3+) se migra el extractor a un modelo local fine-tuneable y se construye un generador de datos sintéticos verificados.
- **MAP-Elites + islas (AlphaEvolve) ahora.** Con presupuestos de 10-20 iteraciones por corrida, una población estructurada por nichos no llega a poblarse; es maquinaria para regímenes de miles de evaluaciones. La spec lo dejó en Tier 2 correctamente. Gatillo de activación: plateau de championship_score durante 3 championships consecutivos *con* presupuesto ≥ 50 iteraciones.
- **Progressive MCGS completo (MLEvolve).** El grafo con 4 tipos de expansión y scheduling UCT es el componente de mayor impacto en MLE-Bench, pero su overhead de bookkeeping solo se justifica con cientos de nodos. El archivo DGM + cross-pollination es su aproximación de bajo costo; suficiente por ahora.
- **SAE steering.** Requiere acceso a activaciones; inviable sobre API cerrada. Solo relevante si se adopta un modelo local con SAE públicas (p.ej. familia Llama).
- **Trío Proposer-Solver-Judge entrenado (MAE).** El *patrón* sí es valioso (ver Fase 3: Proposer de casos difíciles sintéticos), pero la versión con co-entrenamiento de los 3 roles es RL de pesos — descartado arriba.

---

## 2. Diseño de la arquitectura autoevolutiva

### 2.1 Topología de agentes (capas)

```
┌─────────────────────────────────────────────────────────────────┐
│ CAPA 3 — META-POLÍTICA (RL bandit)                               │
│   Meta-agente: elige (operador, artefacto, padre-bias)           │
│   Aprende de fitness_delta. Sin LLM. (NUEVO, Fase 2)             │
├─────────────────────────────────────────────────────────────────┤
│ CAPA 2 — EVOLUTION AI (LLM, ya implementada + extensiones)       │
│   Diagnose ──► Propose(diff A | patch B) ──► Cross-pollinate     │
│   + Meta-review (síntesis transversal cada championship) (NUEVO) │
│   + Retrospective Memory (inyección de intentos previos) (NUEVO) │
├─────────────────────────────────────────────────────────────────┤
│ CAPA 1 — EVALUACIÓN (ya implementada + cascada)                  │
│   Gate 1 (K=4) ──► Gate 2 (K=12, skirmish) ──► Championship      │
│   ELO pairwise (K=32) · fitness F0.5 · piso recall graduado      │
├─────────────────────────────────────────────────────────────────┤
│ CAPA 0 — FENOTIPO (el pipeline Text2Graph)                       │
│   Genome = [Artefacto A: prompt+arch+verify] + [Artefacto B:     │
│   ValidationConfig determinista] ──► extract_article()           │
│   ──► apply_validation() ──► triples (from, to, act, polarity)   │
└─────────────────────────────────────────────────────────────────┘
        Archivo open-ended (DGM): linaje, ELO, diagnosis, deltas
        = sustrato de memoria compartido por todas las capas
```

La división clave (heredada de RoboPhD y que hay que **proteger** al extender): la capa 0 contiene un componente determinista barato (Artefacto B, costo $0) y uno estocástico caro (Artefacto A). La evolución puede explotar el barato sin gastar presupuesto — el smoke test ya mostró este comportamiento (las dos primeras mutaciones exitosas fueron `diff_b`).

### 2.2 El bucle de RL del meta-agente: formalización

El meta-agente NO es un LLM: es un bandit contextual (LinUCB o Thompson Sampling discreto; con tan pocas muestras por corrida, Thompson con priors Beta sobre brazos discretizados es lo robusto). Formalización MDP degenerado a bandit (horizonte 1 por iteración, sin dinámica de estado controlable a largo plazo — honestidad: modelarlo como MDP completo con γ>0 sería sobreingeniería con <100 transiciones por corrida):

**Estado / contexto `s_t`** (vector barato, todo ya computable desde el archivo):
- `precision_rel`, `recall_rel`, `f05_ent` del campeón actual (régimen del problema)
- distancia al piso de recall: `recall_rel − 0.15` (¿estamos en zona de penalización?)
- racha: nº de iteraciones desde la última mejora de ELO del linaje
- tasa de éxito reciente por operador (ventana de 10): `%(fitness_delta > 0)` para `diff_a`, `diff_b`, `cross`
- presupuesto restante normalizado

**Acciones `a_t`** (espacio discreto, ~12 brazos):
- **Operador:** `diff_a` (mutar prompt) | `diff_b` (mutar ValidationConfig) | `cross` (cross-pollination top-2) | `fresh` (generación desde cero estilo Evolution-Agent de AI Co-Scientist, nueva semilla con arquitectura distinta — incluye la variante "debate")
- **Sesgo de selección de padre:** `exploit` (top-ELO) | `explore` (DGM estándar: sigmoid(ELO)/(1+hijos))
- (Opcional, segunda etapa) **Foco del diagnóstico:** FP-relaciones | FN-relaciones | entidades

**Recompensa `r_t`** (densa, ya persistida):
- `r_t = fitness_delta` del hijo vs campeón en el skirmish (clip a [−0.2, +0.2] para que outliers de submuestreo no dominen)
- bonus terminal cada championship: `+α · (championship_score_nuevo − championship_score_anterior)` repartido a las acciones de la ventana (crédito retrospectivo simple, α≈0.5) — esto corrige el sesgo de que el skirmish es ruidoso y el championship es la métrica anclada

**Por qué esto funciona donde un calendario fijo falla:** la utilidad de los operadores es *no estacionaria por régimen*. Cuando `precision_rel` es el problema, `diff_b` (endurecer validación) rinde rápido; cuando el sistema choca con el piso de recall, `diff_b` adicional es contraproducente y hay que mover el prompt (`diff_a`) o cruzar. Un bandit contextual con el estado de arriba captura exactamente ese switching; `cross_every=7` no.

**Garantías anti-degeneración:** ε-greedy floor (cada operador conserva probabilidad mínima 0.05) para que el bandit nunca apague un brazo del todo — equivalente funcional de los stepping stones de DGM pero en el espacio de operadores.

### 2.3 Interacción completa por iteración (con las extensiones)

1. **Meta-agente (Capa 3)** lee contexto `s_t` del archivo → elige `(operador, sesgo_padre)`.
2. **Selección de padre (Capa 1/archivo)** según sesgo.
3. **Evolution AI (Capa 2):** construye el prompt de mutación con (a) diagnóstico FP/FN del padre, (b) memoria retrospectiva (top-3 deltas ±), (c) último Meta-review. Emite diff A / patch B / cross / genoma fresco.
4. **Cascada (Capa 1):** gate 1 (K=4) → gate 2 (K=12) → update ELO + `fitness_delta` → recompensa al meta-agente.
5. **Cada M iters:** championship (eval fijo, top-3) + **Meta-review** sobre errores agregados + chequeo Goodhart (gap eval/test > 0.10 ⇒ alerta).
6. **Archivo:** persiste todo (ya implementado); el meta-agente persiste su posterior (nuevo, un JSON pequeño).

### 2.4 Definición de recompensas del sistema (no solo del meta-agente) — análisis crítico

El fitness actual (`0.45·F0.5_rel + 0.25·F0.5_ent + 0.15·Pol + 0.15·Act − 0.10·cost − piso`) está bien orientado al cuello de botella, con dos riesgos que hay que monitorear, no rediseñar a priori:

1. **Goodhart sobre F0.5:** premiar precisión 2× invita a genomas conservadores que emiten solo lo obvio. El piso de recall graduado (commit `94658a0`) es la contramedida correcta; vigilar en las primeras corridas reales si la población se apila justo encima de 0.15 (síntoma: histograma de `recall_rel` con masa en [0.15, 0.20]). Contramedida si ocurre: subir el piso gradualmente (curriculum: 0.15 → 0.25) en lugar de tocar pesos.
2. **Ruido de skirmish:** K=12 artículos dan estimaciones de F0.5_rel con varianza alta (cada artículo aporta pocas relaciones gold). El ELO absorbe parte (resultados binarios win/loss agregan robustez, como argumenta RoboPhD), y el championship ancla. La cascada del §1.2(a) *aumenta* este ruido en gate 1 (K=4) — aceptable porque su único rol es descartar mutantes claramente malos (ε de margen).

---

## 3. Plan de acción por fases

### Fase 1 — MVP real: cerrar el loop contra Gemini (1-2 semanas, ~$10-20 API)

El código está listo; lo que falta es **evidencia con el modelo de producción**. El smoke con qwen2.5:7b validó la mecánica, no la hipótesis.

| # | Tarea | Criterio de salida |
|---|---|---|
| 1.1 | Corrida baseline: `python -m swarm_optimizer.run --iterations 20 --budget 8.0 --subsample-k 12` con `GEMINI_API_KEY` | history.jsonl con ≥15 mutaciones no-noop; championship_score del campeón vs seed |
| 1.2 | Implementar **cascada de evaluación** (gate K=4 con margen ε=0.05 antes del skirmish K=12) | Tests + medición: % de mutantes descartados en gate 1 y ahorro de tokens |
| 1.3 | Instrumentación: histograma `recall_rel` de población, tasa de éxito por `mutation_type`, gap Goodhart por championship | Script `scripts/report_run.py` que emite resumen MD por corrida |
| 1.4 | Sembrar 2-3 semillas (seed actual + variante `verify=True` + variante `architecture="debate"`) y dejar competir | Las 3 en el archivo con ELO tras 20 iters |
| 1.5 | Análisis de la corrida: ¿Precision_rel mejoró? ¿qué artefacto tocaron las mutaciones ganadoras? | Decisión documentada go/no-go a Fase 2 |

**Riesgo principal de fase:** que 20 iteraciones no muevan Precision_rel del 0.20. Mitigación: las tareas 1.3-1.5 están diseñadas para que incluso una corrida "fallida" produzca el diagnóstico que parametriza la Fase 2 (¿el problema es el operador de mutación, el ruido del skirmish, o el techo del modelo?).

### Fase 2 — Bucle de RL meta-nivel + memoria (2-3 semanas)

Precondición: ≥1 corrida de Fase 1 completada (necesitamos `fitness_delta` reales para priors del bandit).

| # | Tarea | Criterio de salida |
|---|---|---|
| 2.1 | `meta_policy.py`: Thompson Sampling sobre brazos {diff_a, diff_b, cross, fresh} × {exploit, explore}; ε-floor 0.05; posterior persistido en JSON; priors desde corridas de Fase 1 | Tests puros (sin API); reemplaza el calendario fijo `cross_every` en `loop.py` (flag `--meta-policy` para A/B) |
| 2.2 | Recompensa con crédito de championship (α=0.5 retrospectivo sobre la ventana M) | Test de asignación de crédito |
| 2.3 | **Memoria retrospectiva v1**: inyección de top-3 deltas ± en `_PROPOSE_PROMPT` | Test: el prompt de propose contiene los intentos previos del linaje |
| 2.4 | **Meta-review agent**: 1 llamada por championship, FP/FN agregados del top-3 → 3-5 patrones → inyección en `_DIAGNOSE_PROMPT` | Campo `meta_review` persistido en archivo |
| 2.5 | Operador `fresh` (generación desde cero inspirada en top-ranked, estilo AI Co-Scientist) | Nuevo `mutation_type="fresh"` con tests |
| 2.6 | A/B: 2 corridas de 20 iters, calendario fijo vs meta-policy, misma seed de submuestreo | Comparación championship_score + tasa de mutaciones útiles; decisión de adopción |

**Por qué este orden:** 2.1-2.2 no requieren LLM (puro Python, testeable hoy); 2.3-2.5 son cambios de prompt de bajo riesgo; 2.6 es el experimento que justifica (o revierte) todo. Mantener la disciplina TDD del Tier 1.

### Fase 3 — Escalado y autoevolución abierta (condicional a resultados, 4+ semanas)

Gatillos explícitos, no calendario — cada subfase se activa por su síntoma:

| Gatillo observado | Acción |
|---|---|
| Plateau de championship con ≥50 iters de presupuesto | **MAP-Elites** ligero: grid 2D (precision_rel × cost) sobre el archivo existente — `archive.py` ya guarda ambas dimensiones; solo cambia `select_parent` |
| Mutaciones útiles dependen de conocimiento que se repite | **Memoria v2**: retrieval BM25+embeddings+RRF sobre el corpus de diagnosis (MLEvolve) |
| Skirmish saturado (los 93 artículos gold ya no discriminan) | **Proposer de casos difíciles** (patrón MAE sin RL de pesos): LLM genera artículos sintéticos cuya dificultad se calibra por la tasa de fallo del campeón; quality filtering con umbral tipo MAE (descartar score <0.7); los sintéticos solo en skirmish, NUNCA en championship (el ancla sigue siendo gold humano) |
| Evolved-Flash se acerca al techo | **Inverse scaling de RoboPhD**: correr la evolución sobre un modelo más barato (Flash-Lite u Ollama local) — RoboPhD muestra +8.9 pts en el tier barato vs +2.3 en el caro; "skip a tier" en costo de producción |
| Todo lo anterior estable | **Co-evolución del Evolution AI**: el meta-nivel evoluciona también `_DIAGNOSE_PROMPT`/`_PROPOSE_PROMPT` (la "meta-evolution" que RoboPhD deja como future work). Requiere sandboxing y revisión humana de los prompts evolucionados — el riesgo de seguridad que señala RoboPhD (código/prompts auto-modificados sin revisión) aplica aquí |

**Líneas rojas de la Fase 3** (decisiones que requieren replanteo, no incremento): fine-tuning de pesos solo si se migra a modelo local + datos sintéticos verificados en volumen; nunca contaminar el championship con datos sintéticos; nunca dar al Evolution AI capacidad de editar código ejecutable del repo sin gate humano (a diferencia de RoboPhD, aquí el genoma es prompt+config — mantener esa frontera es la decisión de seguridad más importante de la arquitectura).

---

## 4. Riesgos transversales y mitigaciones

| Riesgo | Origen (paper/experiencia) | Mitigación instalada o propuesta |
|---|---|---|
| Goodhart sobre eval set | DGM, RoboPhD | Gap eval/test chequeado en championship (implementado); alerta >0.10 |
| Inversión de selección bajo el piso de recall | Smoke test propio | Penalización graduada (commit `94658a0`) — vigilar apilamiento sobre 0.15 |
| Ruido de submuestreo K=12 | RoboPhD (5 BDs × 30 preguntas) | ELO binario agrega robustez; championship ancla; clip de reward del bandit |
| Judge/diagnóstico LLM inconsistente | MAE (−2.63% sin Judge entrenado) | El "judge" final aquí es determinista (gold standard) — ventaja estructural del dominio; el LLM solo diagnostica, no puntúa |
| Costo API descontrolado | DGM (~$5k/semana) | `budget_usd` duro + cascada + PRICE_MULT ya implementados |
| Auto-modificación insegura | RoboPhD, DGM | Genoma = prompt+config (no código ejecutable); mantener esa frontera |
| Sobreingeniería del meta-nivel | MLEvolve (overhead MCGS) | Bandit antes que MDP; gatillos explícitos por síntoma en Fase 3; A/B 2.6 con poder de veto |

---

## 5. Síntesis

La tesis unificadora de la literatura revisada es que **el razonamiento y la optimización eficaces son procesos sociales/poblacionales, no monolíticos**: poblaciones que compiten (RoboPhD, AlphaEvolve), archivan trayectorias subóptimas como peldaños (DGM), debaten internamente (Societies of Thought) y sintetizan críticas transversales (AI Co-Scientist). El repo ya implementa el esqueleto correcto de esa tesis. Las tres incorporaciones que cambian la pendiente de mejora — cascada de evaluación, bandit meta-nivel sobre operadores, y memoria retrospectiva + meta-review — comparten una propiedad: **explotan información que el sistema ya genera y persiste pero todavía no usa**. Ese es el criterio de priorización de este informe, y es también la definición operativa de "autoevolución" que la evidencia soporta hoy: no un sistema que se reescribe a sí mismo, sino uno que deja de tirar a la basura lo que aprende en cada iteración.

**Próximo paso inmediato:** Tarea 1.1 — primera corrida real de 20 iteraciones contra Gemini con el presupuesto de $8 ya parametrizado.
