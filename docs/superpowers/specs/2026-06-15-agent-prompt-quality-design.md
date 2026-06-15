# Diseño — Calidad de prompts por agente + system prompts robustos

**Fecha:** 2026-06-15
**Rama:** `feat/evolutionary-optimizer-redesign`
**Estado:** aprobado por el usuario, implementación directa autorizada

## Problema

El loop self-evolve no replica autónomamente lo que el humano hacía a mano: las
mejoras de f05 (0.886→0.928) vinieron porque el usuario **proponía buenos prompts y
mutaciones manualmente**, no porque los meta-agentes los generaran solos. Mirando el
código, la causa raíz es concreta:

1. **`cross_pollinate` y `fresh_genome` están ciegos** — reciben solo los dos prompts
   padres, sin métricas, sin diagnóstico, sin saber *por qué* cada padre destaca. Un
   LLM no puede "combinar lo mejor de ambos" sin esa señal.
2. **`diagnose`/`meta_review` no ven trayectorias** — reciben strings `from-act->to (id)`,
   sin el cuerpo del artículo ni la `evidence_quote`. GEPA (arXiv:2507.19457) insiste en
   reflexionar sobre la **trayectoria completa**, no sobre etiquetas sueltas.
3. **Ningún agente usa rol `system`** — todo va en un blob `contents=` (mensaje usuario).
   Los backends Anthropic/OpenAI soportan rol `system` pero no se aprovecha; con el
   cambio a Opus, un system prompt estable y robusto mejora el seguimiento de instrucciones.
4. **No hay tests que garanticen** que cada agente tenga un prompt bien estructurado ni
   que reciba toda la información necesaria.

## Objetivo

- Cada meta-agente tiene un **system prompt ultra-robusto** (rol + invariantes + contrato
  de salida) versionado en un paquete `prompts/`, separado del mensaje **user** dinámico.
- Tests parametrizados (`PromptSpec` rubric) garantizan estructura (en system) y completitud
  de contexto (en user) por agente — sin costo de API.
- `cross`/`fresh` dejan de ser ciegos; `diagnose` recibe trayectorias; `propose` se
  consolida a calidad-GEPA. (Los tests fallidos se arreglan en el mismo PR — opción B.)
- Estructura alineada a `gepa-ai/gepa` (`prompts/` co-localizado con su lógica), pensando
  en repo de producción. La migración completa del resto del código es follow-up.

## Alcance (decisión: Opción A)

**Dentro:** paquete `prompts/` en su ubicación final gepa-aligned; split system/user en los
5 meta-agentes; rubric + tests; arreglo de agentes ciegos. **Fuera (follow-up):** migración
del resto del código a `src/`, limpieza de las ~30 scripts scratch de la raíz.

## Inventario de agentes (qué participa en el self-evolve)

| # | Agente | Tipo | Cambio en este spec |
|---|--------|------|---------------------|
| 1 | `propose` | LLM meta ⭐ | system/user split; consolida básico+GEPA; centerpiece |
| 2 | `cross_pollinate` | LLM meta | split; **deja de ser ciego** (métricas + diagnosis) |
| 3 | `fresh_genome` | LLM meta | split; **deja de ser ciego** (diagnosis + meta_review) |
| 4 | `diagnose` | LLM meta | split; **trayectorias** (body + evidence_quote) |
| 5 | `meta_review` | LLM meta | split |
| 6 | `extract` (seed) | LLM tarea | rubric coverage únicamente (`runtime_split=False`) |
| 7 | `verify` | LLM tarea | rubric coverage únicamente |
| — | `MetaPolicy` | bandit no-LLM | sin cambio |

Los agentes-tarea (extract/verify) NO cambian su runtime: el extractor crea su propio
`genai.Client` vía `_client()` (separado del cliente del loop), y CLAUDE.md prohíbe romper
la comparabilidad del gold. Entran al rubric solo para cobertura estructural.

## Arquitectura

### Refactor de transporte (system/user split, retrocompatible)

`generate_content(model, contents, system=None)` en los 4 backends. `system=None` por
defecto → comportamiento idéntico al actual (los call-sites de extracción no pasan `system`).

- **Anthropic:** `messages.create(..., system=system)` solo si `system` no es None.
- **OpenAI:** anteponer `{"role":"system","content":system}` solo si `system`.
- **Ollama:** campo `"system"` en el payload de `/api/generate` solo si `system`.
- **Gemini (NUEVO `GeminiClient` en `llm_backends.py`):** wrapper sobre `genai.Client`;
  `system` → `config=types.GenerateContentConfig(system_instruction=system)`. `loop.py`
  cambia su default `genai.Client(...)` → `GeminiClient(...)`. El extractor queda con
  `genai.Client` crudo (no pasa system; sigue funcionando).

### Paquete `prompts/` (fuente única, runtime + test)

```
swarm_optimizer/prompts/
├── __init__.py      # ALL_SPECS = [DIAGNOSE_SPEC, META_REVIEW_SPEC, PROPOSE_SPEC, ...]
├── base.py          # PromptSpec dataclass + helpers del rubric
├── diagnose.py      # SYSTEM_DIAGNOSE + build_user_diagnose(...) + DIAGNOSE_SPEC
├── meta_review.py
├── propose.py
├── cross.py
├── fresh.py
├── extract.py       # describe SEED_PROMPT para cobertura (runtime_split=False)
└── verify.py        # describe _VERIFY_PROMPT para cobertura
```

```python
@dataclass(frozen=True)
class PromptSpec:
    agent: str
    system: str                          # system prompt robusto ("" si runtime_split=False)
    build_user: Callable[..., str]       # construye el mensaje user/contexto
    required_structure: tuple[str, ...]  # substrings que evidencian ROLE/TASK/OUTPUT/CONSTRAINTS
    required_context: tuple[str, ...]    # claves de probe cuyos sentinelas deben aparecer en user
    probe: dict                          # inputs representativos con sentinelas únicos
    runtime_split: bool = True           # False = solo cobertura (agente-tarea)
```

### `mutate.py` reconectado

Cada meta-agente importa `system` + `build_user` desde `prompts/` y llama:
```python
user = build_user_propose(genome=..., diagnosis=..., memory=..., examples=..., ...)
resp = client.models.generate_content(model=model, contents=user, system=SYSTEM_PROPOSE)
```
La lógica de parseo (SEARCH/REPLACE, patch, fail-safe noop) NO cambia. `cross_pollinate`
y `fresh_genome` ganan parámetros nuevos (`metrics`, `diagnosis`, `meta_review`) cableados
desde `loop.py`; el contexto de "por qué cada padre destaca" se arma desde el championship.

### Enriquecimiento de contexto (arreglo de agentes ciegos)

- **`_format_errors` en `loop.py`** pasa a producir, además de los strings FP/FN, una
  muestra de **trayectorias** `{body_excerpt, evidence_quote, pred, gold}` para `diagnose`.
- **`cross`/`fresh`** reciben desde `loop.py` las métricas de los top-2 (P/R/f05, ya
  calculadas en el championship) y el `current_meta_review`/diagnosis vigente.

## Tests (`swarm_optimizer/tests/test_prompt_quality.py`)

Parametrizado sobre `ALL_SPECS`:

1. `test_system_prompt_structure` — target = `spec.system` si `runtime_split` else
   `build_user(**probe)`. Cada token de `required_structure` debe aparecer.
2. `test_user_has_full_context` — renderiza `build_user(**spec.probe)` con **sentinelas
   únicos** por input; cada sentinela de `required_context` debe aparecer. Si un builder
   ignora un arg (ej. `cross` ignora `diagnosis`), el sentinela falta → falla → fuerza el cableado.
3. `test_agent_call_uses_system_and_user` — llama la función real (`propose`/`cross`/...)
   con un FakeClient que captura `contents` + `system`; verifica que el system robusto llegue
   como `system=` y el contexto como `contents=`.
4. `test_backends_accept_system_param` — los 4 backends aceptan `system=` y lo enrutan
   al canal correcto (param Anthropic, mensaje system OpenAI, campo Ollama, config Gemini).

Actualizaciones a tests existentes: `FakeModels`/`Boom` en `test_mutate.py` ganan
`system=None` y capturan `last_system`; las assertions de contexto (que chequean `last_prompt`)
siguen válidas porque el contexto sigue yendo en el user.

## Criterios de éxito

- Los 111 tests existentes siguen verdes + nuevos tests de `test_prompt_quality.py` pasan.
- `python -m swarm_optimizer.run --probe` corre sin regresión.
- Ningún `required_context` sentinel falta para ningún agente (cross/fresh ya no ciegos).
- Diff de `extractor.py` runtime: cero (solo cobertura de rubric).

## Riesgos

- **Romper comparabilidad del gold:** mitigado — extractor runtime intacto.
- **Romper los 4 backends:** mitigado — `system=None` retrocompatible + test por backend.
- **Prompts en español que el rubric no capture semánticamente:** el rubric usa substrings
  concretos (invariantes), no juicio semántico; el LLM-as-judge queda fuera de alcance.
