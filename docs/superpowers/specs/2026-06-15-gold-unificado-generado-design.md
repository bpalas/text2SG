# Gold único v3 — 100 chunks generados (Opus 4.8, anclados) + fusión con el gold real

**Fecha:** 2026-06-15
**Rama:** `feat/evolutionary-optimizer-redesign`
**Estado:** Diseño en revisión (dirección: **generado-anclado, 100 chunks, gold único**)
**Specs relacionadas:** `2026-06-14-synthetic-subagent-eval-improve-design.md` (harness swarm + score)
**Supersede:** la idea de un v2 sintético como *side-benchmark separado* — este lo **unifica** al gold real.

## 1. Contexto y motivación

El **gold real vive en un repo hermano** `../gold_standard_v5/data/` (no en este repo):

- `pilot_gold_articles.parquet` — **93 art**; columnas `article_id` (sha256), `stratum`, `period`,
  `title`, `body`, `publish_date`, `n_elite_actors_matched`, `n_inst_actors_matched`, `body_chars`.
  8 estratos adversariales: S1_common_surname_ambiguity(15), S2_closed_world_non_roster(15),
  S3_sentinel_dyads(15), S5_long_form(15), S6_short_flash(10), S8_random_baseline(10),
  S4_roster_duplicate_stress(7), S7_period_edges(6).
- `pilot_gold_final.parquet` — **441 rels**; columnas `article_id, u_from, u_to, act_type, polarity,
  is_reactive, issue, evidence_quote, source, n_inclusion_votes, dispute_type`. `act_type` tiene cola
  larga (57 tipos: accuses 78, endorses 74, calls_on 51, allies_with 49, … **co_occurs 20** ← la
  clase FP espuria del diagnóstico).
- `pilot_entity_unions/<article_id>.yaml` — unions por artículo (`load_union` en `rubric.py:42`).
- `pilot_deliberations/<art>__U?-U?.json` — adjudicación humana por par.

El loop (`swarm_optimizer/loop.py:22-23`) carga `GOLD_ARTICLES`, `GOLD_PARQUET`,
`load_union_map`, `load_splits`. **Matching del scorer = por PAR dirigido `(u_from,u_to)`**
(loop.py:53); `act_type`/`polarity` se puntúan aparte (Act_acc/Polarity_acc). El campeón marca
**P_rel 0.40** sobre este gold; ~29% de los FP son omisiones (gold incompleto); ruido del test
(135 rel) ~±0.08.

**Decisión del usuario:** un **gold único** = los 93 reales **+ 100 chunks nuevos GENERADOS por
Opus 4.8, anclados** en noticias reales de los top-10, con **difficulty 1–10** (sesgo a 5–8),
**etiquetado conservador** (solo relaciones inequívocas — "no marcar antes que marcar mal"), schema
compatible con el gold real.

## 2. Objetivo y no-objetivos

**Objetivo:** (1) generar 100 chunks cortos anclados con verdad plantada **limpia y completa por
construcción**, difficulty 1–10; (2) **fusionarlos** con el gold real en un dataset único `gold_v3`
**local a este repo**, consumible por el loop (loaders repuntados vía switch de versión), con
provenance y un **re-baseline único** del campeón.

**No-objetivos:**
- No etiquetar texto real (descartado en la decisión).
- No reparar las omisiones del gold viejo (trabajo aparte; ver [[gold-incompleteness-audit]]).
- No tocar el split viejo *sin versionar* — se versiona (`gold_v3` convive con el actual vía switch).
- No modificar/sobreescribir los archivos del repo hermano `gold_standard_v5` (solo se copian).
- No versionar el parquet de 7.2 GB.

## 3. Esquema de fusión (cada chunk generado → filas del gold real)

| Destino | Cómo se llena para un chunk generado |
|---|---|
| `articles.parquet` | `article_id="gen_NNN"`, `stratum="G{banda}"` (por difficulty, §5), `period="generado"`, `title`, `body`, `publish_date`=fecha del ancla, `body_chars`, `n_*_matched`=contados de los actores plantados |
| `gold_final.parquet` | `article_id, u_from, u_to, act_type∈9 canónicos, polarity, is_reactive, issue`=tema del ancla, `evidence_quote` **poblado**, `source="opus_planted"`/`"opus_verifier_extra"`, `n_inclusion_votes/dispute_type`=NA |
| `pilot_entity_unions/gen_NNN.yaml` | `entities_union:[{union_id, type, canonical_names, surfaces}]` (convertido del `unions.json` del pipeline) |
| `splits.json` | añadir los `gen_NNN`, re-estratificar por `stratum` |

Los 9 `act_type` canónicos (del sintético): `endorses, accuses, calls_on, allies_with,
distances_from, attacks, questions, negotiates_with, competes_with`. **No** plantamos `co_occurs`
(es la clase FP que el extractor debe *evitar*). `id` con prefijo `gen_` = provenance visible y sin
colisión con los sha del gold real.

## 4. Pipeline de generación (Opus 4.8, anclado)

Reparto **Python (determinista)** vs **Workflow (LLM fan-out, sin FS/Python)**:

```
parquet (top-10, Downloads) ──filtro (Python)──▶ data/gold_v3/anchors.parquet (~250 candidatos)
                                                   │  args
                                                   ▼
                  [Workflow: Opus lee cada ancla] ──▶ andamiaje {actores, tema, registro, distractores, dificultad sugerida}
                                  controller escribe scaffolding.json
                                                   ▼
            composición de guion (Python) ──▶ guiones.json (verdad plantada + difficulty + stratum)
                                                   │  args
                                                   ▼
                  [Workflow: Opus 4.8 redacta chunk corto] ──▶ {body, evidence_quote por relación}
                  [Workflow: verificador NO-Opus] ──▶ dropa relación sin quote defendible, añade extras seguros
                                  controller escribe redacciones.json
                                                   ▼
              ensamblaje (Python) ──▶ articles/gold_final parquet · unions YAML · splits.json (schema gold real)
```

**"100% seguro" operacionalizado:** el verificador (modelo distinto a Opus) exige que **cada
relación del gold tenga un `evidence_quote` que la soporte sin ambigüedad**; si no → se **elimina**
del gold (no se marca). Preferimos un chunk con menos relaciones que con una dudosa. Como Opus
**genera** el texto, la verdad es **completa por construcción** para lo que queda → no hace falta
abstención en el scorer (eso era para texto real; aquí no aplica).

## 5. Escala de dificultad 1–10 (atada a la taxonomía de estrés del proyecto)

| Banda | Expresión | Tipo de estrés (alineado a los estratos del gold real) |
|---|---|---|
| **1–3** | Explícita, una oración, verbo conector directo | Actores roster, sin ambigüedad |
| **4–6** | Nominalización, relación en 2 oraciones, correferencia simple | Actores non-roster, atribución por vocero |
| **7–8** | Oblicua: discurso reportado, pasiva, correferencia por cargo, unir párrafos | Ambigüedad de apellido, *sentinel dyad* (co-mención tentadora **sin** relación), period edges |
| **9–10** | Implícita/inferencial (inequívoca para lector experto) | Closed-world non-roster + múltiples distractores cercanos |

**Distribución objetivo de los 100** (sesgo a medio-difícil como pediste): 1–3 ≈ 15, 4–6 ≈ 35,
7–8 ≈ 35, 9–10 ≈ 15. **~15 de los 100 son distractores** (0 relaciones; su `difficulty` = cuán
tentadora es la trampa). Cada chunk lleva campo `difficulty` (1–10) y su `stratum` derivado
(`G_d1-3`, `G_d4-6`, `G_d7-8`, `G_d9-10`).

## 6. Medios top-10 y anclaje

latercera, emol, cooperativa, biobiochile, elmostrador, t13, adnradio, 24horas, lanacion, cnnchile.
El anclaje toma **actores, tema y patrones de distractor** del artículo real (Opus lo lee); el chunk
es **nuevo** (no paráfrasis). Preprocesado obligatorio: **normalizar/reparar encoding** del corpus
(se observó mojibake) y descartar anclas irreparables.

## 7. Unificación, comparabilidad y re-baseline ⚠️

- **Gold v3 local:** copiar los 93 artículos+rels+unions del repo hermano a `data/gold_v3/` y
  **añadir** los 100 generados → dataset único, self-contained en este repo. El repo hermano
  `gold_standard_v5` queda **intacto**.
- **Switch de versión:** parametrizar `GOLD_ARTICLES`/`GOLD_PARQUET`/`UNIONS_DIR`/`SPLITS_PATH`
  (en `loop.py`, `splits.py`, `rubric.py`) por un `GOLD_VERSION` (constante o env) que apunte a
  `v_current` (93) o `v3` (193). Permite A/B y rollback.
- **Rompe comparabilidad** con corridas previas (regla de CLAUDE.md). → **re-baseline del campeón
  UNA vez** sobre v3, documentado. Se conservan los splits viejos para poder re-medir el
  *subconjunto-93* y confirmar que no se movió.
- **Provenance:** `source` (`opus_planted` vs el del gold real) y `stratum` (`G_*` vs `S*`)
  distinguen generado de real → filtrable en cualquier reporte.

## 8. Circularidad y calidad

Opus **genera y planta**; el extractor objetivo es **gemini-3.1-flash-lite** (familia distinta) →
circularidad limitada. Verificador **no-Opus**. `evidence_quote` obligatorio. **Spot-check humano**
de ~8 chunks. Naturaleza de la señal: el gold generado mide *“cuán bien el extractor recupera lo que
Opus expresó claramente en casos de dificultad diseñada”* — **complementa** (no reemplaza) los
estratos adversariales reales (S1–S8).

## 9. Criterios de éxito

1. ≥95% de relaciones con `evidence_quote` defendible (acuerdo del verificador).
2. Distribución de difficulty lograda (§5); distractores con 0 relaciones.
3. El loop corre sobre `gold_v3` sin errores (loaders + unions + splits OK).
4. Re-baseline documentado; el *subset-93* re-medido coincide con el baseline viejo (±ruido).
5. Spot-check humano del subconjunto OK.

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Escritura cross-repo | No se toca `gold_standard_v5`; v3 se construye **local** copiando |
| Comparabilidad rota | Versionado + switch `GOLD_VERSION` + re-baseline único documentado + re-check del subset-93 |
| Circularidad | Extractor cross-family (gemini); verificador no-Opus; quote obligatorio; spot-check humano |
| Recall saturado (problema del pilot) | Forzar oblicuas/implícitas en bandas 7–10; medir R por difficulty |
| Vocab de `act_type` | Usar solo los 9 canónicos; nunca plantar `co_occurs` |
| Encoding sucio del corpus | Normalizar (ftfy/charset) antes de anclar; descartar irreparables |
| Fuga del ancla (paráfrasis) | Redactar NUEVO; verificador marca body que copie el ancla |

## 11. Archivos a crear / modificar

- `scripts/synth_mine_anchors.py` *(nuevo)* — filtra parquet top-10 → `data/gold_v3/anchors.parquet`.
- `scripts/gen_gold_v3_workflow.js` *(nuevo)* — Workflow: mining → redacción → verificación.
- `scripts/build_gold_v3.py` *(nuevo)* — andamiaje→guion→ensamblaje al **schema del gold real**
  (articles + gold_final + unions YAML `gen_*` + splits), copiando los 93 reales y añadiendo los 100.
- `swarm_optimizer/{loop.py, splits.py, rubric.py}` *(modificar)* — `GOLD_VERSION` switch hacia
  `data/gold_v3/` sin romper `v_current`.
- `data/` *(nuevo)* — `articles_all.parquet` (gitignored) + `gold_v3/`.
- `.gitignore` — añadir el parquet grande / `data/raw`.

## 12. Validación post-build

Correr el campeón sobre `gold_v3`: métricas globales + **por `stratum`** (incl. los nuevos `G_*`) +
**por `difficulty`**; comparar el *subset-93* contra el baseline viejo para confirmar no-regresión de
la medición. Reporte a `results/swarm/`.
