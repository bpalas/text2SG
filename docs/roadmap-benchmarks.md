# Roadmap — superar los benchmarks de extracción de relaciones

**Fecha:** 2026-06-10 · **Estado de partida (held-out, test n=30, flash-lite, semilla p017):**
F1 0.335 / P 0.331 / R 0.338. Benchmark a batir: pilot1 (F1 0.300 / P 0.345 / R 0.265)
y pilot2 (F1 0.267 / **P 0.481** / R 0.184). Ya ganamos F1 y recall en held-out;
falta precisión: −0.02 vs pilot1, −0.15 vs pilot2.

**Principio rector (aprendido hoy):** toda mejora se valida en eval Y test; lo que solo
gana en eval es ruido (cross: 0.398 eval → 0.317 test). El gold v2-neutral es la regla
de medición oficial.

---

## Fase 0 — Protocolo de benchmark (½ día, $0) — PREREQUISITO

Sin esto, los números no son comparables y volvemos a perseguir fantasmas.

- [ ] Fijar protocolo: gold v2-neutral (`pilot_gold_v2_sin_adiciones.parquet`),
      eval n=20 seed 7 para iterar + test n=30 SOLO para decisiones de adopción.
- [ ] Regla anti-quemado del test: máximo una mirada al test por candidato a adopción
      (no por iteración). El loop evolutivo nunca ve el test.
- [ ] Alinear con el benchmark externo: confirmar si pilot1/pilot2 usan el mismo
      matcher (par dirigido, fuzzy 85) — si no, los números no son comparables y hay
      que correr nuestras configs con SU scorer (o viceversa) una vez.
- [ ] Script único `benchmark.py` que corre una config y reporta ambos splits + delta
      vs p017, para que cada experimento cueste un comando.

## Fase 1 — Sprint de precisión quirúrgica (1-2 días, ~$1-2)

Cerrar el −0.02 vs pilot1 atacando la taxonomía REAL de los 34 FP auditados,
una mutación por vez sobre p017, adopción solo si gana en eval y no cae en test.

| Mutación | Ataca | FPs | Costo |
|---|---|---|---|
| 1a. **Atribución de hablante** (prompt: "identifica quién habla en cada párrafo; el acto se atribuye a quien lo ejecuta, no a quien lo reporta ni a quien es citado por terceros") | atribución errada | ~6-9 | 1 corrida |
| 1b. **Filtro de dominio** (prompt: "solo relaciones entre actores de la política chilena; actores extranjeros solo si interactúan con un actor chileno") | política extranjera | ~6 | 1 corrida |
| 1c. **Endpoints = actores de la lista** (la regla strict de wave3; en flash-lite está sin probar) | conceptos/no-actores | ~7 | 1 corrida |
| 1d. **Dedup persona↔institución** (prompt: si el vocero actúa en nombre de la institución, emitir solo la institución) | redundancia | ~3 | 1 corrida |
| 1e. **Verify con verificador fuerte** (flash o pro verificando flash-lite; wave3: +0.06 P por −0.01 R) | FP residuales | — | 2x tokens |

Meta de fase: **P ≥ 0.345 en test sin que F1 baje de 0.32**. Con ~9-15 FP reales
evitables sobre ~80 emisiones, el headroom aritmético es +0.05-0.10 de precisión.

## Fase 2 — Dominar la curva, no el punto (1 día, ~$0.5)

pilot2 (P 0.481 @ R 0.184) no es "mejor precisión": es OTRO punto de operación
(ultra-conservador). En vez de perseguirlo con el mismo punto, publicar DOS modos
del mismo extractor:

- [ ] **Modo balanceado** (p017+fase1): F1 máximo, el que ya gana.
- [ ] **Modo alta-precisión** (solo artefacto B, $0 por variante): barrer
      `max_relations_per_article` ∈ {2,3,4} × `min_quote_len` ∈ {15,25} ×
      confidence-gate sobre las MISMAS predicciones guardadas, y elegir el punto
      con P > 0.481. Si nuestra curva pasa por encima de su punto, dominamos
      pilot2 sin una llamada extra.
- [ ] Entregable: curva P-R del extractor con los dos puntos del benchmark marcados.

## Fase 3 — Escalera de modelos (½ día, ~$2-5)

Todo lo anterior es flash-lite (el modelo del benchmark). Medido el techo barato:
- [ ] p017+fase1 en **flash** y (si paga) en **pro**, eval+test.
- [ ] Matriz extractor×verificador (lite/flash) — wave3 sugiere que el verificador
      fuerte es donde más paga el upgrade.
- [ ] Decidir el punto costo/calidad para producción con el fitness price-aware.

## Fase 4 — Gold v2.1 a escala (1 día de agentes, ~$2-3 en Sonnet)

La medición mejora los resultados "gratis" pero está auditada solo en 20/93 artículos.
- [ ] Enjambre de jueces (3 Sonnet por artículo, mayoría) sobre los 73 restantes:
      gold_miss + fantasmas + atribuciones erradas. Anti-circularidad: auditar FPs
      de DOS configs no emparentadas (baseline y pure_llm), no solo del campeón.
- [ ] Mapear la cola de 47 act_types no canónicos → 9 canónicos (puro pandas);
      desbloquea Act_acc como métrica real.
- [ ] Limpiar los 124 fantasmas de unions (afectan Recall_ent).
- [ ] Proceso: parche versionado + firma humana (dueño del gold) antes de adoptar v2.1.

## Fase 5 — Relanzar el loop evolutivo con lo aprendido (continuo, presupuesto)

Recién aquí vuelve la evolución automática — con el campo de juego arreglado:
- [ ] Semilla = p017 + mutaciones adoptadas de fase 1. Gold = v2-neutral (v2.1 cuando exista).
- [ ] **Gate anti-Goodhart**: championship en eval + veto si el test cae >ε
      (presupuestar el test para no quemarlo: solo en championships).
- [ ] Priors del meta-policy actualizados: deprioritizar mutaciones de dirección
      (techo +0.01) y polaridad (no mueve P_rel); priorizar existencia/atribución.
- [ ] Mantener el patrón de hoy: enjambre de expertos para PROPONER, agentes baratos
      para CRIBAR, Gemini para CONFIRMAR, test para ADOPTAR.

---

## Metas

| Horizonte | Métrica (test, flash-lite) | Hoy | Meta |
|---|---|---|---|
| Fase 1 | Precisión | 0.331 | ≥ 0.345 (bate pilot1 en los 3 ejes) |
| Fase 2 | P del modo alta-precisión | — | > 0.481 (domina pilot2) |
| Fase 3-5 | F1 | 0.335 | 0.40-0.45 |

## Riesgos

1. **Quemar el test** por mirarlo demasiado → protocolo fase 0.
2. **Circularidad del gold** (auditar con el mismo modelo que extrae) → jueces de
   familia distinta + firma humana.
3. **Benchmark no comparable** (otro gold/matcher) → resolver en fase 0 antes de
   declarar victoria.
4. n=20/30 con ruido ±0.03-0.05 → deltas chicos requieren replicación (2 seeds)
   antes de adoptar.
