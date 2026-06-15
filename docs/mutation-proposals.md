# Propuestas de Mutaciones

Documento vivo. Cada propuesta la puede hacer un humano, un modelo potente (Claude/GPT),
o el meta-review del propio loop. El objetivo es tener un backlog de hipótesis a probar
antes de lanzar corridas caras.

**Cómo usar:**
1. Copiar la plantilla de abajo
2. Llenar el razonamiento y el cambio concreto
3. Después de la corrida, marcar el resultado en el checklist de dimensiones
4. Mover a "Historial" con el veredicto final

---

## Plantilla

```markdown
### [ID] Nombre corto de la mutación
**Propuesto por:** (humano / Claude / GPT-4 / meta-review del loop)
**Fecha:** YYYY-MM-DD
**Artefacto:** A (prompt) | B (ValidationConfig)
**Motivación:** ¿Por qué creés que esto ayuda? ¿Qué patrón de error ataca?

**Cambio concreto:**
- Artefacto A: qué parte del prompt cambiaría y cómo
- Artefacto B: qué campo cambiaría y a qué valor

**Hipótesis:** "Si hago X, espero que Precision_rel suba porque Y"

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   |        |         |        | ⬜       |
| Recall_rel      |        |         |        | ⬜       |
| Precision_ent   |        |         |        | ⬜       |
| Polarity_acc    |        |         |        | ⬜       |
| fitness overall |        |         |        | ⬜       |

**Modelo usado en la corrida:** gemini-2.5-flash / gemini-2.5-pro / qwen2.5:7b
**Veredicto:** ✅ Adoptada | ❌ Descartada | 🔄 Parcial (ver notas)
**Notas:**
```

---

## Propuestas pendientes

### [P-001] Restringir allowed_act_types — quitar `co_occurs`
**Propuesto por:** Claude (análisis de FP del sistema viejo)
**Fecha:** 2026-06-09
**Artefacto:** B (ValidationConfig)
**Motivación:** El análisis del history_old muestra que `co_occurs` es el act_type más
emitido y el que más FP genera. La mera co-presencia en un párrafo no es una relación
política. El LLM lo usa como fallback cuando no hay relación clara.

**Cambio concreto:**
```json
"allowed_act_types": ["endorses","accuses","allies_with","calls_on",
                      "distances_from","attacks","questions",
                      "negotiates_with","competes_with"]
```
(quita `co_occurs` de la lista permitida)

**Hipótesis:** "Si elimino `co_occurs`, la Precision_rel sube porque dejo de emitir
relaciones vacías. El recall baja mínimo porque `co_occurs` es semánticamente débil."

**Checklist de resultado:**
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.059  |         |        | ⬜       |
| Recall_rel      | 0.333  |         |        | ⬜       |
| Precision_ent   | 0.818  |         |        | ⬜       |
| Polarity_acc    |        |         |        | ⬜       |
| fitness overall | 0.4532 |         |        | ⬜       |

**Modelo usado:** —
**Veredicto:** ✅ Adoptada en la semilla (2026-06-09)
**Notas:** Señal con agentes Claude (2026-06-09, `results/swarm/agent_run/REPORTE.md`):
+0.01–0.015 P_rel consistente en toda la grilla, costo en recall ≈ 0. Adoptada como
default de `ValidationConfig` (`SEED_ALLOWED_ACT_TYPES` en `genome.py`). Genomas viejos
con `allowed_act_types: null` se preservan al deserializar. Confirmación de magnitud
pendiente del próximo championship Gemini.

---

### [P-002] Subir `min_quote_len` de 8 a 15
**Propuesto por:** Claude (análisis de patrones de FP)
**Fecha:** 2026-06-09
**Artefacto:** B (ValidationConfig)
**Motivación:** Citas muy cortas (8 chars = "lo dijo") no son evidencia suficiente.
El LLM a veces ancla relaciones en fragmentos genéricos que aparecen en muchos artículos.
Exigir citas más largas filtra ruido sin perder relaciones bien fundamentadas.

**Cambio concreto:**
```json
"min_quote_len": 15
```

**Hipótesis:** "Sube Precision_rel porque descarta relaciones con evidencia vaga.
Recall_rel baja leve porque algunas relaciones reales tienen citas cortas."

**Checklist de resultado:**
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.059  |         |        | ⬜       |
| Recall_rel      | 0.333  |         |        | ⬜       |
| Precision_ent   | 0.818  |         |        | ⬜       |
| Polarity_acc    |        |         |        | ⬜       |
| fitness overall | 0.4532 |         |        | ⬜       |

**Modelo usado:** —
**Veredicto:** ⏳ Pendiente
**Notas:** Probar en combinación con P-001. Si se combinan, el efecto puede ser aditivo.
Señal con agentes Claude (2026-06-09): CERO efecto en 42 puntos de grilla — Claude ya emite
citas largas. Antes de gastar en Gemini, mirar la distribución de largos de cita de Gemini.

---

### [P-003] Agregar regla de "verbo conector explícito" al prompt
**Propuesto por:** Claude (análisis del cuello de botella de Precision_rel)
**Fecha:** 2026-06-09
**Artefacto:** A (prompt)
**Motivación:** El principal patrón de FP es relaciones entre entidades que aparecen
en el mismo párrafo sin que una actúe sobre la otra. Agregar una regla explícita
sobre verbos conectores debería atacar esto directamente.

**Cambio concreto:**
Agregar al prompt, después de "La mera co-ocurrencia NO es relación":
```
Una relación requiere que el texto contenga un verbo o acción explícita que conecte
a los dos actores (declaró, criticó, apoyó, votó, firmó, pidió, rechazó, etc.).
Si solo aparecen mencionados en el mismo párrafo sin vínculo verbal, NO emitir relación.
```

**Hipótesis:** "Reduce FP de co-presencia. Precision_rel sube. Recall puede bajar
leve si hay relaciones implícitas en el gold."

**Checklist de resultado:**
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.059  |         |        | ⬜       |
| Recall_rel      | 0.333  |         |        | ⬜       |
| Precision_ent   | 0.818  |         |        | ⬜       |
| Polarity_acc    |        |         |        | ⬜       |
| fitness overall | 0.4532 |         |        | ⬜       |

**Modelo usado:** —
**Veredicto:** ⏳ Pendiente
**Notas:** Señal con agentes Claude (2026-06-09): positiva moderada como mutación AISLADA
(P y R suben sobre el baseline); combinada con given_entities colapsa el recall (R=0.057).
Nunca apilarla con otra instrucción conservadora.

---

### [P-004] Arquitectura debate interno (Societies of Thought)
**Propuesto por:** Informe arquitectura autoevolutiva 2026-06-09
**Fecha:** 2026-06-09
**Artefacto:** A (prompt, architecture="debate")
**Motivación:** El paper Societies of Thought muestra que estructurar el razonamiento
como debate (proponer → objetar → reconciliar) mejora accuracy causalmente. Ya existe
la infraestructura (`architecture="debate"` en el genoma). La semilla debate compite
con la semilla base via ELO — si gana, la adopción es automática.

**Cambio concreto:**
Usar `Genome(prompt_text=SEED_PROMPT, architecture="debate")` como semilla adicional.
El debate ya está implementado en `extractor.py::_DEBATE_INSTRUCTIONS`.

**Hipótesis:** "La voz crítica interna durante la extracción filtra relaciones dudosas
antes de emitirlas, subiendo Precision_rel sin sacrificar Recall."

**Checklist de resultado:**
| Dimensión       | Antes (base) | Después (debate) | Delta  | ¿Mejoró? |
|-----------------|--------------|------------------|--------|----------|
| Precision_rel   | 0.059        |                  |        | ⬜       |
| Recall_rel      | 0.333        |                  |        | ⬜       |
| Precision_ent   | 0.818        |                  |        | ⬜       |
| Polarity_acc    |              |                  |        | ⬜       |
| fitness overall | 0.4532       |                  |        | ⬜       |

**Modelo usado:** —
**Veredicto:** ⏳ Pendiente
**Notas:** Probar con `--multi-seed` para que compita directamente contra la base.
Señal con agentes Claude (2026-06-09): negativa — el debate hunde el recall (0.126→0.080)
sin ganancia de precisión que compense. Despriorizar.

---

### [P-005] Activar `enforce_polarity_consistency`
**Propuesto por:** Claude (análisis del campeón id=0, corrida 2026-06-09)
**Fecha:** 2026-06-09
**Artefacto:** B (ValidationConfig)
**Motivación:** Act_acc está en 0.652 y Polarity_acc en 0.809: una fracción importante
de las relaciones emitidas tiene act_type/polarity contradictorios (ej: `attacks` con
polarity=positive). Esas contradicciones internas son un proxy barato de "relación
adivinada": si el modelo ni siquiera es consistente en el signo del acto, es probable
que la relación sea FP. Filtrarlas cuesta $0.

**Cambio concreto:**
```json
"enforce_polarity_consistency": true
```

**Hipótesis:** "Si descarto relaciones con act_type/polarity inconsistentes, Precision_rel
sube porque las contradicciones internas correlacionan con FP. Recall_rel baja poco porque
las relaciones reales bien extraídas suelen ser consistentes."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** — (descartada sin correr)
**Veredicto:** ❌ Descartada — la motivación asume un filtro que no existe
**Notas:** Invalidada por lectura de código (panel multi-modelo 2026-06-09; confirmado a mano):
`enforce_polarity_consistency` en `validation.py:64-67` SOBREESCRIBE `polarity` con el valor
esperado de `_POLARITY_MAP` — nunca descarta la relación (ver `test_validation.py`,
"corrects_polarity"). No puede mover Precision_rel (el cuello de botella); solo Polarity_acc,
que ya está sana (0.809). Además la rúbrica matchea relaciones SOLO por par
(`rubric.py:178-190`) — act_type/polarity no afectan Precision_rel.

---

### [P-006] Cap de relaciones por artículo (`max_relations_per_article = 8`)
**Propuesto por:** Claude (análisis del campeón id=0, corrida 2026-06-09)
**Fecha:** 2026-06-09
**Artefacto:** B (ValidationConfig)
**Motivación:** Con Precision_rel 0.227 y Recall_rel 0.381, el extractor emite ~1.7x más
relaciones de las que acierta. El patrón típico de sobre-emisión se concentra en artículos
con muchos actores (panoramas electorales, votaciones), donde el modelo tiende a conectar
"todos contra todos". Un cap por artículo recorta la cola de relaciones de baja confianza
justo donde más FP se generan, sin tocar los artículos normales.

**Cambio concreto:**
```json
"max_relations_per_article": 8
```

**Hipótesis:** "Si capeo a 8 relaciones por artículo, Precision_rel sube porque los FP se
concentran en artículos sobre-emitidos. Recall_rel baja levemente solo en artículos densos."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** gemini-2.5-flash
**Veredicto:** ⏳ Pendiente
**Notas:** Señal con agentes Claude (2026-06-09): el mayor efecto B sobre el linaje
given_entities con haiku (P 0.343→0.377 por -0.012 R). Probar sobre ese linaje.
Barrido offline post P-008/P-001 (2026-06-09, /tmp regrid sobre preds crudas): con haiku
cap6 > cap8 > cap10 (P 0.418/0.390/0.377, R plana 0.264); con sonnet cap10 fue el mejor
y los caps casi no mueven. Gold: p90=11 rels/artículo (9 en la muestra) → cap6 trunca
muy bajo el gold denso; **para el loop Gemini: sembrar cap10, NO cap6**.
Ola 3 agentes Claude (2026-06-09): confirmado con preds reales. Dejar que el loop explore
cap hacia abajo desde 10. Muestra de 20: ruido ±0.02, no sobre-leer el cap6.
Cambio a B = $0 tokens. Riesgo: el truncado no sabe cuáles relaciones son
mejores — si descarta TP en artículos legítimamente densos, Recall puede resentirse más
de lo esperado. Vigilar el piso de Recall_rel ≥ 0.15.

---

### [P-007] Regla "ante la duda, omite" + prohibir inferencia vía terceros
**Propuesto por:** Claude (análisis del campeón id=0, corrida 2026-06-09)
**Fecha:** 2026-06-09
**Artefacto:** A (prompt)
**Motivación:** El fitness es F0.5 (precisión pesa 2:1 sobre recall), pero el prompt actual
no le comunica esa asimetría al modelo: pide extraer "TODAS las interacciones", lo que
empuja a sobre-emitir. Además, parte de los FP son relaciones inferidas por transitividad
(A critica al gobierno de B → el modelo emite A→ministro de B). Una regla explícita de
abstención alinea el comportamiento del LLM con la métrica que se optimiza.

**Cambio concreto** (diff SEARCH/REPLACE sobre el prompt del campeón):
```
SEARCH:  Sin cita literal verificable: no emitas la relación.
REPLACE: Sin cita literal verificable: no emitas la relación.
         Si dudas entre emitir o no una relación, NO la emitas: es preferible omitir a inventar.
         No infieras relaciones a través de terceros: si A actúa sobre B, eso NO implica relación de A con los aliados, ministros o coalición de B.
```

**Hipótesis:** "Si instruyo abstención explícita y prohíbo la inferencia transitiva,
Precision_rel sube porque el modelo deja de emitir relaciones especulativas. Recall_rel
baja algo, pero con F0.5 el trade-off es favorable mientras Recall ≥ 0.15."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** gemini-2.5-flash
**Veredicto:** ⏳ Pendiente
**Notas:** Riesgo principal: el modelo puede volverse demasiado conservador y hundir
Recall_rel hacia el piso de 0.15 (con Recall actual 0.381 hay margen, pero vigilar).
Se solapa parcialmente con P-003 (verbo conector) — probar por separado, no juntas,
para poder atribuir el efecto.

---

### [P-008] Dedup por par — quitar `act_type` de la clave de dedup
**Propuesto por:** Panel multi-modelo 2026-06-09 (sonnet/validación-B, haiku/validación-B); mecánica verificada a mano en código
**Fecha:** 2026-06-09
**Artefacto:** B (requiere mini-cambio de código: la clave está hardcodeada en `validation.py`, no es campo del genoma)
**Motivación:** La rúbrica acredita cada par gold UNA sola vez (`rubric.py:178-190`): toda
segunda predicción con el mismo `(from, to)` suma `fp_rel` sin importar el act_type. El dedup
actual (`validation.py:70-74`) usa clave 3-tupla con act_type, así que el LLM emite
`attacks`+`accuses` para el mismo evento y la segunda es FP mecánico garantizado.

**Cambio concreto** (en `apply_validation`, validation.py:70-74):
```python
key = (normalize(rel.get("from_entity", "")), normalize(rel.get("to_entity", "")))
```

**Hipótesis:** "Precision_rel sube eliminando FP mecánicos. Costo en Recall_rel ~0 bajo esta
rúbrica: cada par solo puede puntuar una vez, descartar la segunda relación del par no quita TPs."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** —
**Veredicto:** ✅ Adoptada en código (2026-06-09)
**Notas:** Aplicada en `validation.py` (clave de dedup ahora `(from, to)`, sin act_type);
tests actualizados (`test_dedup_collapses_same_pair_different_act_type`). Aplica a TODOS
los genomas con `dedup=true`, incluido el campeón — su próximo championship ya medirá el
efecto. Riesgo residual: conserva la relación "equivocada" del par (act_type/polarity peor
que la descartada) → puede bajar Act_acc/Polarity_acc, no Precision_rel. La mecánica
del rubric y del dedup está confirmada por lectura directa del código.

---

### [P-009] Quitar "TODAS" — consigna precision-first en el prompt
**Propuesto por:** Panel multi-modelo 2026-06-09 — convergencia 3/3 modelos (haiku, sonnet y opus en lente prompt-A); verificada adversarialmente
**Fecha:** 2026-06-09
**Artefacto:** A (prompt)
**Motivación:** La consigna "extrae TODAS las interacciones" empuja a sobre-emitir mientras el
fitness es F0.5 (precisión pesa 2:1): desalineación instrucción-métrica. Como la rúbrica solo
cuenta pares, reducir pares dudosos ataca Precision_rel directamente.

**Cambio concreto** (diff SEARCH/REPLACE; el verificador recortó la 3ª cláusula del REPLACE
original por redundante con "Sin cita literal verificable" + `require_evidence_substring`):
```
SEARCH:  extrae TODAS las interacciones explícitas entre ellos.
REPLACE: extrae SOLO las interacciones de las que tengas certeza explícita. Es preferible
         omitir una relación dudosa a emitir una incorrecta.
```

**Hipótesis:** "Si elimino la consigna maximalista, Precision_rel sube porque el modelo deja
de emitir pares dudosos. Recall_rel baja con margen amplio (0.381 vs piso 0.15)."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** —
**Veredicto:** ⏳ Pendiente
**Notas:** EXCLUYENTE con P-007 (mismo eje de conservadurismo verbal — riesgo de doble
conservadurismo acumulado en el mismo linaje): probar esta primero; P-007 solo si esta no
mueve Precision_rel. Abortar linaje si Recall_rel < 0.15; alerta temprana si < 0.25. Parte
de los FP no es sobre-emisión sino endpoints no resueltos (ver P-010) — el techo de esta
mutación es menor al que sugiere la hipótesis.
Señal con agentes Claude (2026-06-09): NEGATIVA fuerte — recall colapsa a 0.046 sola y a
0.195 combinada con given_entities, sin ganar precisión. El "doble conservadurismo" es real.
Despriorizar en modelos chicos; si se prueba con Gemini, vigilar recall desde la iteración 1.

---

### [P-010] Semilla `given_entities` — inyectar el roster real de actores
**Propuesto por:** Panel multi-modelo 2026-06-09 — 4 panelistas (sonnet/arquitectura, opus/arquitectura, opus/prompt-A, sonnet/validación-B); verificada adversarialmente con ajustes
**Fecha:** 2026-06-09
**Artefacto:** A (arquitectura)
**Motivación:** `one_pass` (el campeón) NUNCA inyecta el union_map — `extractor.py:73-79` solo
lo hace con `architecture="given_entities"` — pese a que el SEED_PROMPT promete "una lista de
actores presentes". El modelo extrae contra un roster vacío y fabrica endpoints fuera del
roster real; todo endpoint NIL o no resuelto es FP automático (`rubric.py:174-175`).

**Cambio concreto** (con los ajustes del verificador adversarial):
1. Sembrar SOLO la variante nueva (NO usar `--multi-seed`, que re-sembraría las 3 semillas
   existentes como clones y pagaría championships full-eval extra):
   `run_loop(seed_genomes=[Genome(prompt_text=SEED_PROMPT + "\nUsa el NOMBRE del actor tal
   como aparece en la lista, nunca su código (U1, U2...).", architecture="given_entities")])`
   — la frase anti-uid evita perder por artefacto de formato (el roster se inyecta como
   "U1: Nombre" y `match_entity` no resuelve uids).
2. Criterio de adopción: comparar Precision_rel y Recall_rel del championship contra el campeón
   one_pass, NO solo fitness/ELO — given_entities infla el componente de entidades por
   construcción (w_ent=0.25 en fitness.py).

**Hipótesis:** "Si el modelo ve el roster real, deja de fabricar endpoints fuera del union y
Precision_rel sube eliminando la clase de FP por endpoint no resuelto."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** —
**Veredicto:** ⏳ Pendiente
**Notas:** Señal con agentes Claude (2026-06-09): **GANADOR CLARO de la grilla completa** —
P_rel 0.229→0.343 (+50%), R_rel 0.126→0.276 (+119%), P_ent 1.000 con haiku; con sonnet
P_rel 0.406 / R_rel 0.322. El roster real elimina FP de endpoint Y ayuda al recall.
Máxima prioridad para sembrar en el loop Gemini.
Con Precision_ent=0.748, los endpoints no resueltos explican una fracción de los FP,
no demostrablemente la mayoría — la adopción la decide el ELO/championship, no la hipótesis.
Si el union_map no cubre todos los actores del gold, Recall_rel baja (vigilar piso 0.15).
Variantes diferidas del mismo cluster (probar por separado, solo si esta se adopta): filtro B
nil-endpoint-precheck; regla A "ambos extremos deben ser actores de la lista". Actualizar el
help de `--multi-seed` y el docstring de `seed_variants` si la semilla entra a `seed_variants()`.

---

### [P-011] Semilla `verify=True` — verificación agéntica de segunda pasada
**Propuesto por:** Panel multi-modelo 2026-06-09 — 3 panelistas (lente arquitectura en los 3 modelos); verificada adversarialmente con prerequisito
**Fecha:** 2026-06-09
**Artefacto:** A (arquitectura)
**Motivación:** La segunda pasada que elimina relaciones sin soporte literal ya está
implementada (`extractor.py:137-152`, gateada por el flag `verify` en 179-183) y declarada en
`seed_variants()` (genome.py), pero jamás se ha corrido. Es un jurado independiente que ataca
directamente el cuello de Precision_rel.

**PREREQUISITO (bug encontrado por el verificador adversarial):** ✅ RESUELTO 2026-06-09.
`verify_relations` no contabilizaba tokens — `extract_article` sumaba solo el usage del
PRIMER call. Sin arreglarlo: (a) el genoma verify competía con costo medido ~mitad del real
→ ventaja injusta en el fitness price-aware, y (b) `should_stop` subcontaba el presupuesto
real de la corrida. Arreglo aplicado: `verify_relations` retorna `(relations, tokens)`
leyendo el usage del segundo call y `extract_article` los suma (fail-safe ante excepción:
originales + tokens=0; con output inválido los tokens sí cuentan porque el call ocurrió).

**Cambio concreto:** sembrar solo la variante (no `--multi-seed`):
`run_loop(seed_genomes=[Genome(prompt_text=SEED_PROMPT, verify=True)])`

**Hipótesis:** "El verificador rechaza relaciones sin anclaje literal → Precision_rel sube;
Recall cae poco por el fail-safe que conserva las originales ante error."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** —
**Veredicto:** ⏳ Pendiente (señal positiva condicionada al verificador)
**Notas:** ~2x tokens/artículo reales. Fail-safe asimétrico: `return verified if verified else
relations` (extractor.py:150) restaura los originales cuando el verificador rechaza TODAS las
relaciones — justo los artículos 100% FP donde más precisión se ganaría. El verificador ve
`body[:4000]` y puede reescribir `evidence_quote` (citas no literales mueren después en
`require_evidence_substring`). La variante "gatear verify solo si len(relations)>6" NO está
implementada — descartada como está escrita; re-proponerla como cambio de código si verify-plano
muestra señal.
Señal ola 3 agentes Claude (2026-06-09, `results/swarm/agent_run2/REPORTE.md`): el verify
prompt real aplicado sobre preds crudas — con SONNET como verificador sobre el linaje given:
P_rel 0.403→0.462 por −0.01 R, el mejor F0.5 de toda la grilla (0.407). Con HAIKU como
verificador: neutro (P sube, R lo paga). Implicación: verify=True con flash-verificando-flash
puede quedar corto; vale probar verificador más fuerte (pro) aunque suba el costo.

---

### [P-012] Few-shot contrastivo: 1 ejemplo positivo + 1 negativo de co-ocurrencia
**Propuesto por:** Claude (sesión 2026-06-10, análisis del campeón P_rel=0.227)
**Fecha:** 2026-06-10
**Artefacto:** A (prompt)
**Motivación:** El prompt declara reglas abstractas ("la mera co-ocurrencia NO es relación")
pero nunca muestra un caso límite. Los ejemplos contrastivos calibran el umbral de decisión
mejor que las reglas, sin empujar conservadurismo verbal global (eje que ya falló en P-007/P-009).

**Cambio concreto:** Agregar al prompt, antes de "Responde ÚNICAMENTE con JSON válido":
```
Ejemplos:
1) "La senadora Vodanovic emplazó al ministro Elizalde a acelerar la agenda"
   → SÍ: {from_entity: "Vodanovic", to_entity: "Elizalde", act_type: "calls_on"}
2) "En la ceremonia estuvieron presentes Boric, Tohá y Marcel"
   → NO emitir nada: co-presencia sin acto explícito.
```

**Hipótesis:** "Si muestro un par contrastivo, Precision_rel sube porque el modelo aprende
el límite emitir/no-emitir por ejemplo y deja de emitir pares de co-presencia, sin la señal
de abstención explícita que colapsa el recall."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** —
**Veredicto:** ⏳ Pendiente
**Notas:** Riesgo: anclaje al act_type del ejemplo (sobre-emisión de `calls_on`) y ~150
tokens extra por artículo. No apilar con P-003 (mismo eje co-presencia) para poder atribuir.

---

### [P-013] Filtro B: la cita debe mencionar al menos un endpoint (`require_entity_in_quote`)
**Propuesto por:** Claude (sesión 2026-06-10)
**Fecha:** 2026-06-10
**Artefacto:** B (requiere mini-cambio de código: nuevo campo en `validation.py`)
**Motivación:** `require_evidence_substring` solo verifica que la cita exista en el artículo,
no que conecte a los actores declarados — una cita genérica real ("el proyecto fue rechazado")
ancla cualquier par fabricado y hoy pasa el filtro.

**Cambio concreto:** Nuevo campo `"require_entity_in_quote": true` en ValidationConfig:
descartar la relación si `evidence_quote` (normalizada) no contiene ningún token del nombre
de `from_entity` NI de `to_entity` (mismo `normalize` del dedup, match por substring de tokens).

**Hipótesis:** "Si exijo que la cita nombre al menos un endpoint, Precision_rel sube porque
las relaciones cuyos actores no aparecen en su propia evidencia son mayoritariamente pares
fabricados sobre texto real."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** —
**Veredicto:** ⏳ Pendiente
**Notas:** Riesgo: FN por correferencia ("el mandatario afirmó…" sin nombrar a Boric en la
cita) — vigilar Recall ≥ 0.15. Exigir AMBOS endpoints sería demasiado agresivo: empezar con
uno. Antes de la corrida, medir offline sobre preds crudas existentes (agent_run2) qué
fracción de TP/FP caería — el filtro es determinista, el barrido es gratis.

---

### [P-014] Cap de fan-out por cita (`max_relations_per_quote = 2`)
**Propuesto por:** Claude (sesión 2026-06-10)
**Fecha:** 2026-06-10
**Artefacto:** B (mini-cambio de código en `validation.py`)
**Motivación:** El patrón "todos contra todos" en artículos densos (motivación de P-006) tiene
una firma determinista más fina que el cap global: el modelo recicla UNA misma cita para
abanicar N pares, y solo 1–2 de ellos son el evento real. Filtro quirúrgico donde P-006 es romo.

**Cambio concreto:** En `apply_validation`, agrupar por `evidence_quote` normalizada y
conservar solo las 2 primeras relaciones de cada grupo (orden de emisión).
```json
"max_relations_per_quote": 2
```

**Hipótesis:** "Si capeo el fan-out por cita, Precision_rel sube porque los FP de abanico
comparten ancla textual, mientras las relaciones reales tienden a tener citas propias —
corta FP sin penalizar artículos densos legítimos como sí hace el cap global."

**Checklist de resultado** (llenar post-corrida):
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.227  |         |        | ⬜       |
| Recall_rel      | 0.381  |         |        | ⬜       |
| Precision_ent   | 0.748  |         |        | ⬜       |
| Polarity_acc    | 0.809  |         |        | ⬜       |
| fitness overall | 0.4385 |         |        | ⬜       |

**Modelo usado en la corrida:** —
**Veredicto:** ⏳ Pendiente
**Notas:** Riesgo: citas que sí reportan 3+ interacciones reales (votaciones, acuerdos
multipartito) pierden TPs. Interacciona con P-006 — probar aislada del cap global. Igual que
P-013, barrer offline sobre preds crudas antes de gastar en una corrida.

---

### [P-017] Puente de cobertura de act_types + cap6 (coverage_bridge)
**Propuesto por:** Claude — panel de expertos agent_run6 (METO-3, metodólogo de evaluación)
**Fecha:** 2026-06-10
**Artefacto:** A (prompt) + B (ValidationConfig)
**Motivación:** Muchos FN no son relaciones omitidas sino actos que el modelo descarta
porque "no calzan" textualmente con los 9 act_types (criticar gestión, querellarse,
militar en una coalición). Mapearlos al tipo más cercano sube recall sin inventar
relaciones. Ganador claro de agent_run6: df05 +0.180 vs control, único combo sobre el
piso de recall (R 0.218, +0.184), mejor Polarity_acc del run (0.842). cap6 sumó +0.057 P
gratis donde hubo volumen.

**Cambio concreto:**
- Artefacto A: appendear tras el schema JSON el bloque "Cobertura de actos: si observas
  un acto político explícito que no calza textualmente con los 9 tipos, NO lo omitas —
  mapéalo al tipo más cercano: criticar/reprochar→questions (attacks si descalificación
  frontal); investigar/querellarse→accuses; militar/integrar coalición→allies_with;
  respaldar postura→endorses; desmarcarse→distances_from. La única razón válida para
  omitir una relación es que el texto no la reporte explícitamente."
- Artefacto B: `max_relations_per_article: 6`
- Genoma exacto listo: `results/swarm/agent_run6/coverage_bridge/genome_b_cap6.json`

**Hipótesis:** "Sube Recall_rel (recupera actos fuera de esquema) sin bajar Precision_rel
(el puente reclasifica, no inventa); cap6 poda la cola de baja calidad."

**Checklist de resultado:**
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.400  |         |        | ⬜       |
| Recall_rel      | 0.417  |         |        | ⬜       |
| Precision_ent   | 1.000  |         |        | ⬜       |
| Polarity_acc    | 0.794  |         |        | ⬜       |
| fitness overall | 0.543  |         |        | ⬜       |

**Modelo usado:** — (señal haiku agent_run6; pendiente Gemini)
**Veredicto:** pendiente
**Notas:** Ver `results/swarm/agent_run6/REPORTE.md`. Cuidado: la magnitud haiku no
transfiere; el control del run estaba deprimido.

---

### [P-018] Bloque intensidad de act_type + reglas de polaridad (sin enforce B)
**Propuesto por:** Claude — panel de expertos agent_run6 (METO-2 + METO-1)
**Fecha:** 2026-06-10
**Artefacto:** A (prompt)
**Motivación:** Polarity_acc 0.794 y confusión entre tipos vecinos (attacks/questions/
accuses/calls_on). El bloque pide elegir el tipo MÁS DÉBIL compatible con la cita (sin
descartar la relación) + reglas de polaridad por tipo (attacks/accuses/questions/
distances_from/competes_with→negative; endorses/allies_with→positive; calls_on→negative
si es emplazamiento; no usar neutral como comodín). En haiku dio P 0.600 y Polarity_acc
1.000 pero deprimió la emisión (5 rels) — Gemini no colapsa volumen, justo donde
"reclasificar sin descartar" puede pagar.

**Cambio concreto:**
- Artefacto A: diff exacto en `results/swarm/agent_run6/intensity_plus_polarity/genome.json`
- NO activar `enforce_polarity_consistency=true`: el `_POLARITY_MAP` de validation.py
  mapea calls_on→neutral y cementaría el mismatch dominante.

**Hipótesis:** "Sube Polarity_acc y Precision_rel (menos confusión entre tipos) con
Recall_rel estable porque el bloque prohíbe descartar ante la duda."

**Checklist de resultado:**
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | 0.400  |         |        | ⬜       |
| Recall_rel      | 0.417  |         |        | ⬜       |
| Precision_ent   | 1.000  |         |        | ⬜       |
| Polarity_acc    | 0.794  |         |        | ⬜       |
| fitness overall | 0.543  |         |        | ⬜       |

**Modelo usado:** — (señal haiku agent_run6; pendiente Gemini)
**Veredicto:** pendiente
**Notas:** Riesgo: depresión de emisión en modelos débiles. Si P-017 y P-018 pagan por
separado, probar el cruce (ortogonales: existencia vs reclasificación). Re-encolar
P-015 (confidence): en agent_run6 el filtro `min_confidence` fue un no-op — verificar
que el prompt pida el campo y que la validación lo mapee antes de re-correr.

---

## Historial (propuestas cerradas)

> Mover aquí las propuestas con veredicto final. Incluir el fitness_delta del loop
> y si la mutación fue adoptada por el campeón.

*(vacío — primera corrida pendiente)*

---

## Cómo pedir propuestas a un modelo potente

Copiar este prompt a Claude / GPT-4 después de cada corrida:

```
Soy investigador de un sistema evolutivo que extrae relaciones políticas chilenas
de artículos de prensa. El extractor usa un LLM (Gemini Flash) con:

PROMPT ACTUAL:
[pegar prompt del campeón actual]

VALIDACIÓN ACTUAL (post-proceso determinista):
[pegar ValidationConfig del campeón]

MÉTRICAS ACTUALES:
- Precision_rel: X.XXX  ← CUELLO DE BOTELLA
- Recall_rel: X.XXX
- Precision_ent: X.XXX  (sano)
- Polarity_acc: X.XXX  (sano)
- fitness: X.XXXX

ERRORES MÁS FRECUENTES (FP):
[pegar top-10 FP del reporte]

ERRORES MÁS FRECUENTES (FN):
[pegar top-10 FN del reporte]

Propone 3 mutaciones concretas (cambios pequeños y testeables, uno por vez) que
ataquen el cuello de botella de Precision_rel sin colapsar Recall. Para cada una:
1. Qué cambiar exactamente (artefacto A o B, valor concreto)
2. Por qué creés que ayuda
3. Qué riesgo tiene (qué podría empeorar)
```
