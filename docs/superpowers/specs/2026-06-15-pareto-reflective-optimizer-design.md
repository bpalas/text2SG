# Optimizer Pareto-reflexivo (GEPA-adaptado) — diseño

**Fecha:** 2026-06-15
**Rama:** `feat/evolutionary-optimizer-redesign`
**Estado:** Diseño aprobado, pendiente plan de implementación
**Specs relacionadas:** `2026-06-14-synthetic-subagent-eval-improve-design.md` (harness sintético),
`2026-06-13-text2graph-rearchitecture-analysis-tool-design.md` (Analysis Tool).

## 1. Contexto y motivación

El repo ya tiene un clon estructural de RoboPhD en `loop.py` (ELO + 2/3 artefactos + evaluador +
`mutate.py` + `archive.py` + bandit), pero estaba **estancado**: campeón único congelado, ELO
ranqueando genomas muertos, reward ruidoso (gold real). Esta sesión construimos lo que faltaba —
**reward limpio (oráculo sintético) + executor confiable + agente de reflexión Opus** — y al medir
descubrimos que **el problema es multi-objetivo**: la precisión es alta en todos lados (0.91–0.95),
el diferenciador es el **recall**; el tool determinista ayuda al modelo fuerte y hunde al débil;
`verify` cambia recall por precisión. Colapsar eso a un escalar (f05) y a un campeón único (ELO)
tira la estructura que importa.

La línea de **GEPA (Agrawal et al., 2025)** — evolución reflexiva con selección de Pareto — calza
mejor que el ELO/campeón-único de RoboPhD para *nuestros* requerimientos. Este diseño reemplaza la
selección por un **frente de Pareto reflexivo**, reusando el harness confiable. No es un gemelo de
RoboPhD: es su corrección para un problema de tradeoff.

## 2. Objetivo y no-objetivos

**Objetivo:** Un loop de optimización que mantiene un **frente de Pareto sobre (precisión, recall)
por tier de modelo**, donde un agente Opus reflexiona sobre las trazas de error de un miembro del
frente y propone una mutación; el candidato se evalúa en el oráculo sintético y se agrega al frente
si no es dominado. Selección por dominancia de Pareto, **sin ELO**.

**No-objetivos (v1):**
- Especialistas por tipo de instancia (dominio/dureza/medio) en el frente — los objetivos son P/R.
- Automatización 100% desatendida — el driver corre K rondas con archivo persistido; el controller
  puede orquestar/checkpointear.
- Optimización conjunta multi-tier — un frente por modelo; el costo se maneja como tier separado.
- Tocar/resucitar el `loop.py` ELO — se deja como está; `pareto.py` es nuevo y enfocado.

## 3. Arquitectura — el ciclo cerrado (GEPA-adaptado)

```
miembro del frente ──[Opus reflexiona sobre sus FN/FP por tipo]──▶ genoma candidato
        ▲                                                              │
        │                                                   [evaluar en sintético]
  frente Pareto (por tier) ◀──[add: queda si NO es dominado]── (P, R, diagnóstico)
```

Reemplaza "campeón único por ELO" con "frente de Pareto reflexivo". El costo entra como **tier**
(un frente por modelo: `gemini-2.5-flash-lite`, `gpt-4o-mini`, Haiku, Sonnet, …). Deployment =
el tier más barato cuyo frente cruza el umbral requerido.

## 4. Componentes (unidades chicas, testeables)

### 4.1 `swarm_optimizer/pareto.py` — archivo de Pareto (nuevo, sin ELO)
```python
@dataclass
class ParetoEntry:
    id: int
    genome: dict        # Genome.to_dict()
    P: float
    R: float
    parent_id: int | None
    expansions: int = 0     # cuántas veces se mutó a partir de este (para elegir a quién expandir)

class ParetoArchive:
    def add(self, genome_dict, P, R, parent_id=None) -> ParetoEntry: ...
    def dominates(self, a, b) -> bool:   # a domina b si a.P>=b.P y a.R>=b.R y al menos uno estricto
    def frontier(self) -> list[ParetoEntry]:   # entradas no-dominadas
    def pick_to_expand(self, rng) -> ParetoEntry:   # del frente, el de menos expansions (desempate aleatorio)
    def to_json(self) / from_json(...)   # persistencia + linaje
```
Determinista; un archivo por tier (`results/synthetic/pareto/<model>.json`).

### 4.2 `diagnostics(preds, gold_df, union_map, articles_df)` — trazas de error
Formaliza el script inline ya usado. Devuelve, para un set de predicciones:
- `fn`: relaciones gold no extraídas (con `dureza`, `act_type`, `evidence_quote` del gold) — el hueco
  de recall.
- `fp`: relaciones predichas que no matchean gold (con flag `es_distractor`) — disciplina.
- agregados por `dureza`/`dominio`. Vive en `swarm_optimizer/diagnostics.py` (módulo nuevo enfocado).

### 4.3 Operador reflexivo (el Evolution Agent Opus)
Formaliza el despacho que ya hicimos: dado (genoma del miembro a expandir + sus FN/FP de
`diagnostics`), un subagente Opus reflexiona y devuelve un genoma nuevo (edita prompt A y/o flags de
`AnalysisConfig` C; condicionado al modelo objetivo — tool pesado solo si el modelo lo banca).
Contrato: entra genoma JSON + diagnóstico; sale genoma JSON validado (`Genome.from_json`). El driver
lo dispara; no es código Python puro (es un dispatch de agente), documentado en el runbook.

### 4.4 `scripts/synth_evolve.py` — driver del loop
Por ronda (K rondas, para un `--model` y `--split train`):
1. `pick_to_expand` del frente (menos expandido).
2. Reflexión Opus → genoma candidato (controller dispara el subagente).
3. Evaluar: extraer con el executor (API via `synth_run_model`, o swarm cero-API), `do_score` → P,R.
4. `archive.add` → queda si no-dominado; persistir; `expansions += 1` del padre.
5. Al final: regenerar el gráfico de Pareto y reportar el frente.
Semilla del frente: los genomas base que ya tenemos (seed/analysis/…) ya medidos.

### 4.5 Árbitro
Los mejores del frente se confirman en **synth-test (49 held-out)** y en el **test real** (anti-Goodhart).

## 5. Qué se evoluciona
Genoma = **prompt A + `AnalysisConfig` C** (condicionada al modelo). B se barre gratis (re-`score`
sin re-extraer). `model` = el tier (no se evoluciona).

## 6. Data / reuso
Evaluador: `do_score` (synth_agent_eval). Executor: `synth_run_model` (API) o swarm cero-API
(`dump→collect→score`). Artefactos: `analysis.py` + `genome`. `pareto.py` es nuevo (no toca el
`archive.py` acoplado al ELO).

## 7. Diferencias con RoboPhD/loop.py
- **Pareto** en vez de ELO/campeón-único → sin "campeón congelado" ni "ELO ranquea muertos".
- **Reflexión** (NL sobre trazas) en vez de churn de rollouts ELO → sample-efficient (importa: cada
  rollout cuesta subagentes/API).
- **Reward limpio** (sintético) en vez del gold ruidoso.
- **Costo como tier** (un frente por modelo) → decisión accuracy-cost estilo AstaBench.

## 8. Testing
- `pareto.py`: `dominates` (casos límite: empate, estricto), `frontier` (descarta dominados),
  `add` (no-dominado entra, dominado no desplaza al frente), `pick_to_expand` (menos expandido),
  roundtrip JSON. Con fixtures sintéticas.
- `diagnostics()`: FN/FP correctos sobre un mini-gold fijo (incluye un distractor con FP).
- Suite existente verde (158).

## 9. Alcance / secuencia
1. `pareto.py` + tests.
2. `diagnostics()` + tests.
3. `synth_evolve.py` driver + runbook del operador reflexivo Opus.
4. Sembrar el frente con los genomas ya medidos; correr 2-3 rondas reflexivas sobre un tier
   (ej. Haiku cero-API, o gemini-flash-lite con `--limit` barato).
5. Confirmar el frente en synth-test + test real.

## 10. Riesgos
- **Convergencia/colapso del frente:** si la reflexión propone siempre lo mismo, el frente no crece.
  Mitigación: `pick_to_expand` rota por menos-expandido; diagnóstico por-tipo da señal variada.
- **Goodhart al sintético:** synth-test + test real como árbitro.
- **Reflexión costosa (Opus):** 1 dispatch/ronda; rondas acotadas.
- **El operador reflexivo no es Python puro** (es dispatch de agente) → el driver lo orquesta con
  checkpoints; documentado en el runbook, no 100% automatizado en v1.

## 11. Archivos
- **Nuevo:** `swarm_optimizer/pareto.py`, `swarm_optimizer/diagnostics.py`,
  `swarm_optimizer/tests/test_pareto.py` (+ tests de diagnostics), `scripts/synth_evolve.py`.
- **Reuso sin cambios:** `synth_agent_eval.do_score`, `synth_run_model`, `analysis.py`, `genome`.
</content>
