"""Genera propuestas de mutación basadas en el estado actual del archivo evolutivo.

Lee history.jsonl, extrae el contexto del campeón (prompt, ValidationConfig, métricas,
diagnósticos previos, meta-review) y produce un prompt rico para un modelo externo.

Modos:
  --print     Imprime el prompt a stdout para pegarlo en Claude/ChatGPT (default)
  --call      Llama a Gemini y escribe las propuestas en docs/mutation-proposals.md
  --n N       Número de propuestas a pedir (default: 3)
  --history   Ruta al history.jsonl (default: results/swarm/history.jsonl)

Ejemplos:
  python scripts/propose_mutations.py
  python scripts/propose_mutations.py --call --n 5
  python scripts/propose_mutations.py --history results/swarm/history.jsonl --call
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Forzar UTF-8 en stdout para evitar errores cp1252 en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROPOSALS_DOC = Path(__file__).parent.parent / "docs" / "mutation-proposals.md"
DEFAULT_HISTORY = Path(__file__).parent.parent / "results" / "swarm" / "history.jsonl"


# ── contexto del campeón ──────────────────────────────────────────── #

def load_champion_context(history_path: Path) -> dict | None:
    """Extrae todo el contexto relevante del campeón desde el archivo."""
    from swarm_optimizer.archive import Archive

    if not history_path.exists():
        print(f"ERROR: no existe {history_path}", file=sys.stderr)
        return None

    arc = Archive(history_path)
    champ = arc.champion()
    if not champ:
        print("ERROR: no hay campeón con championship_score en el archivo.", file=sys.stderr)
        print("  Sugerencia: corre al menos una iteración del loop primero.", file=sys.stderr)
        return None

    # métricas del campeón
    m = champ.metrics or {}

    # diagnósticos del linaje (FP/FN patterns que el Evolution AI ya identificó)
    lineage_diags = [
        e.diagnosis for e in arc.lineage(champ.id)
        if e.diagnosis and e.id != champ.id
    ]

    # meta-review más reciente (síntesis transversal)
    all_reviews = [e.meta_review for e in arc.all() if e.meta_review]
    meta_review = all_reviews[-1] if all_reviews else None

    # top mutaciones positivas y negativas del linaje
    tried = [e for e in arc.lineage(champ.id) if e.fitness_delta is not None]
    tried.sort(key=lambda e: e.fitness_delta, reverse=True)
    best = [(e.mutation_type, e.artifact_touched, e.fitness_delta, e.diagnosis)
            for e in tried[:3] if e.fitness_delta > 0]
    worst = [(e.mutation_type, e.artifact_touched, e.fitness_delta, e.diagnosis)
             for e in tried[-3:] if e.fitness_delta < 0]

    return {
        "champion_id": champ.id,
        "model": champ.genome.model,
        "architecture": champ.genome.architecture,
        "prompt": champ.genome.prompt_text,
        "validation": dataclasses.asdict(champ.genome.validation),
        "championship_score": champ.championship_score,
        "metrics": {
            "Precision_rel": m.get("Precision_rel"),
            "Recall_rel": m.get("Recall_rel"),
            "Precision_ent": m.get("Precision_ent"),
            "Polarity_acc": m.get("Polarity_acc"),
            "Act_acc": m.get("Act_acc"),
            "fitness": champ.championship_score,
            "eval_score": m.get("eval_score"),
            "test_score": m.get("test_score"),
        },
        "lineage_diagnoses": lineage_diags,
        "meta_review": meta_review,
        "best_mutations": best,
        "worst_mutations": worst,
        "total_entries": len(arc.all()),
    }


# ── construcción del prompt ───────────────────────────────────────── #

def build_proposal_prompt(ctx: dict, n: int = 3) -> str:
    m = ctx["metrics"]

    def fmt(v): return f"{v:.3f}" if v is not None else "—"

    # métricas formateadas
    metrics_block = f"""\
- Precision_rel : {fmt(m['Precision_rel'])}  ← {'⚠️  CUELLO DE BOTELLA' if (m['Precision_rel'] or 1) < 0.35 else 'OK'}
- Recall_rel    : {fmt(m['Recall_rel'])}
- Precision_ent : {fmt(m['Precision_ent'])}  {'(sano)' if (m['Precision_ent'] or 0) > 0.70 else ''}
- Polarity_acc  : {fmt(m['Polarity_acc'])}
- fitness       : {fmt(m['fitness'])}  (F0.5 price-aware, piso de recall 0.15)"""

    if m.get("eval_score") and m.get("test_score"):
        gap = (m["eval_score"] or 0) - (m["test_score"] or 0)
        metrics_block += f"\n- Gap Goodhart   : eval={fmt(m['eval_score'])} test={fmt(m['test_score'])} diff={gap:+.3f}"

    # diagnósticos previos del linaje
    diag_block = ""
    if ctx["lineage_diagnoses"]:
        diag_block = "\n\nDIAGNÓSTICOS PREVIOS DEL LINAJE (lo que el sistema ya identificó como causas de error):\n"
        for i, d in enumerate(ctx["lineage_diagnoses"][-4:], 1):
            diag_block += f"\n[{i}] {d.strip()[:400]}\n"

    # meta-review
    review_block = ""
    if ctx["meta_review"]:
        review_block = f"\n\nPATRONES SISTÉMICOS (meta-review del loop — síntesis transversal de errores):\n{ctx['meta_review'].strip()[:600]}\n"

    # historial de mutaciones
    hist_block = ""
    if ctx["best_mutations"] or ctx["worst_mutations"]:
        hist_block = "\n\nHISTORIAL DE MUTACIONES DEL LINAJE:\n"
        if ctx["best_mutations"]:
            hist_block += "Funcionaron:\n"
            for mtype, art, delta, diag in ctx["best_mutations"]:
                d_short = (diag or "")[:80].replace("\n", " ")
                hist_block += f"  - {mtype} sobre artefacto {art}: delta {delta:+.4f}  [{d_short}]\n"
        if ctx["worst_mutations"]:
            hist_block += "NO funcionaron:\n"
            for mtype, art, delta, diag in ctx["worst_mutations"]:
                d_short = (diag or "")[:80].replace("\n", " ")
                hist_block += f"  - {mtype} sobre artefacto {art}: delta {delta:+.4f}  [{d_short}]\n"

    prompt = f"""\
Soy investigador de un sistema evolutivo que extrae relaciones políticas chilenas de
artículos de prensa. El extractor usa Gemini ({ctx['model']}, arquitectura: {ctx['architecture']}).
El sistema evoluciona dos artefactos: el PROMPT (A) y una ValidationConfig determinista (B).

El archivo tiene {ctx['total_entries']} entradas (campeón id={ctx['champion_id']}).

═══════════════════════════════════════════════════════
PROMPT ACTUAL DEL CAMPEÓN (Artefacto A):
═══════════════════════════════════════════════════════
{ctx['prompt'].strip()}

═══════════════════════════════════════════════════════
VALIDATION CONFIG ACTUAL (Artefacto B, post-proceso $0):
═══════════════════════════════════════════════════════
{json.dumps(ctx['validation'], indent=2, ensure_ascii=False)}

═══════════════════════════════════════════════════════
MÉTRICAS DEL CAMPEÓN (sobre eval set completo):
═══════════════════════════════════════════════════════
{metrics_block}{diag_block}{review_block}{hist_block}

═══════════════════════════════════════════════════════
CONTEXTO DEL SISTEMA:
═══════════════════════════════════════════════════════
- act_types válidos: endorses, accuses, allies_with, calls_on, distances_from,
  attacks, co_occurs, questions, negotiates_with, competes_with
- El fitness usa F0.5 (pondera más precisión que recall, ratio 2:1)
- Piso de recall: si Recall_rel < 0.15, el fitness colapsa (penalización graduada)
- Cambios al Artefacto B son gratuitos ($0 tokens); cambios al A cuestan una llamada LLM
- El sistema hace diff SEARCH/REPLACE: cada mutación cambia UNA cosa pequeña

═══════════════════════════════════════════════════════
TAREA:
═══════════════════════════════════════════════════════
Propone {n} mutaciones CONCRETAS y TESTEABLES (una por vez, cambio mínimo) que ataquen
el cuello de botella de Precision_rel sin colapsar Recall_rel por debajo de 0.15.

Para cada propuesta devuelve EXACTAMENTE este JSON (sin markdown):
{{
  "proposals": [
    {{
      "id": "P-XXX",
      "name": "Nombre corto",
      "artifact": "A o B",
      "motivation": "Por qué creés que esto ataca el problema específico",
      "change": "Descripción precisa del cambio (qué campo, qué valor, o qué parte del prompt)",
      "diff_search": "Texto EXACTO a reemplazar en el prompt (solo si artifact=A, sino null)",
      "diff_replace": "Texto de reemplazo exacto (solo si artifact=A, sino null)",
      "patch": {{"campo": valor}},  // solo si artifact=B, sino null
      "hypothesis": "Si hago X, espero que Precision_rel [suba/baje] porque Y",
      "risk": "Qué podría empeorar y por qué",
      "priority": "alta / media / baja"
    }}
  ]
}}
"""
    return prompt


# ── parseo y escritura en el doc ──────────────────────────────────── #

def _next_proposal_id(doc_path: Path) -> str:
    """Lee el doc y devuelve el siguiente ID (P-005, P-006, etc.)."""
    if not doc_path.exists():
        return "P-001"
    content = doc_path.read_text(encoding="utf-8")
    ids = re.findall(r'\[P-(\d+)\]', content)
    if not ids:
        return "P-001"
    return f"P-{max(int(x) for x in ids) + 1:03d}"


def proposals_to_markdown(proposals: list[dict], ctx: dict, next_id_start: str) -> str:
    """Convierte las propuestas JSON en markdown para agregar al doc."""
    lines = []
    num = int(next_id_start.split("-")[1])
    today = date.today().isoformat()

    for p in proposals:
        pid = f"P-{num:03d}"
        num += 1
        m = ctx["metrics"]

        change_block = ""
        if p.get("artifact") == "B" and p.get("patch"):
            change_block = f"```json\n{json.dumps(p['patch'], indent=2, ensure_ascii=False)}\n```"
        elif p.get("artifact") == "A" and p.get("diff_search"):
            change_block = (
                f"Reemplazar en el prompt:\n"
                f"```\nBUSCAR:\n{p['diff_search']}\n\nREEMPLAZAR CON:\n{p.get('diff_replace', '')}\n```"
            )
        else:
            change_block = p.get("change", "Ver motivación")

        lines.append(f"""
### [{pid}] {p.get('name', 'Sin nombre')}
**Propuesto por:** Modelo externo (auto-generado por `scripts/propose_mutations.py`)
**Fecha:** {today}
**Artefacto:** {p.get('artifact', '?')}
**Prioridad:** {p.get('priority', 'media')}
**Motivación:** {p.get('motivation', '')}

**Cambio concreto:**
{change_block}

**Hipótesis:** {p.get('hypothesis', '')}

**Riesgo:** {p.get('risk', '')}

**Checklist de resultado:**
| Dimensión       | Antes  | Después | Delta  | ¿Mejoró? |
|-----------------|--------|---------|--------|----------|
| Precision_rel   | {f"{m['Precision_rel']:.3f}" if m.get('Precision_rel') else '—'}  |         |        | ⬜       |
| Recall_rel      | {f"{m['Recall_rel']:.3f}" if m.get('Recall_rel') else '—'}  |         |        | ⬜       |
| Precision_ent   | {f"{m['Precision_ent']:.3f}" if m.get('Precision_ent') else '—'}  |         |        | ⬜       |
| Polarity_acc    | {f"{m['Polarity_acc']:.3f}" if m.get('Polarity_acc') else '—'}  |         |        | ⬜       |
| fitness overall | {f"{m['fitness']:.4f}" if m.get('fitness') else '—'}  |         |        | ⬜       |

**Modelo usado en la corrida:** {ctx['model']}
**Veredicto:** ⏳ Pendiente
**Notas:** —
""")

    return "\n".join(lines)


def append_to_proposals_doc(proposals: list[dict], ctx: dict) -> None:
    """Agrega las propuestas al doc de mutation-proposals.md."""
    if not PROPOSALS_DOC.exists():
        print(f"WARN: no existe {PROPOSALS_DOC}", file=sys.stderr)
        return

    content = PROPOSALS_DOC.read_text(encoding="utf-8")
    next_id = _next_proposal_id(PROPOSALS_DOC)

    md = proposals_to_markdown(proposals, ctx, next_id)

    # insertar antes del "## Historial"
    if "## Historial" in content:
        content = content.replace("## Historial", md + "\n---\n\n## Historial")
    else:
        content += "\n" + md

    PROPOSALS_DOC.write_text(content, encoding="utf-8")
    print(f"[propose] {len(proposals)} propuestas agregadas a {PROPOSALS_DOC}")


# ── llamada a Gemini ──────────────────────────────────────────────── #

def call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> list[dict] | None:
    """Llama a Gemini y parsea el JSON de propuestas."""
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(model=model, contents=prompt)
        text = (resp.text or "").strip()
        # strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return data.get("proposals", [])
    except Exception as e:
        print(f"ERROR al llamar a Gemini: {e}", file=sys.stderr)
        return None


# ── main ──────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(description="Genera propuestas de mutación desde el archivo evolutivo")
    ap.add_argument("--history", default=str(DEFAULT_HISTORY), help="Ruta al history.jsonl")
    ap.add_argument("--call", action="store_true",
                    help="Llama a Gemini y agrega las propuestas al doc")
    ap.add_argument("--model", default="gemini-2.5-pro",
                    help="Modelo a usar para generar propuestas (default: gemini-2.5-pro)")
    ap.add_argument("--n", type=int, default=3, help="Número de propuestas a pedir (default: 3)")
    ap.add_argument("--print-only", action="store_true",
                    help="Solo imprime el prompt, sin llamar a ninguna API")
    args = ap.parse_args()

    ctx = load_champion_context(Path(args.history))
    if not ctx:
        sys.exit(1)

    prompt = build_proposal_prompt(ctx, n=args.n)

    if args.print_only or not args.call:
        print(prompt)
        print("\n" + "═" * 60)
        print("Para que Gemini genere las propuestas automáticamente:")
        print("  python scripts/propose_mutations.py --call")
        print("  python scripts/propose_mutations.py --call --model gemini-2.5-pro --n 5")
        return

    # llamar al modelo y agregar al doc
    print(f"[propose] Consultando {args.model} para {args.n} propuestas...")
    proposals = call_gemini(prompt, model=args.model)
    if not proposals:
        print("ERROR: no se obtuvieron propuestas válidas.", file=sys.stderr)
        print("Prompt generado (para pegar manualmente):")
        print(prompt)
        sys.exit(1)

    print(f"[propose] {len(proposals)} propuestas recibidas:")
    for p in proposals:
        print(f"  {p.get('id','?')} [{p.get('priority','?')}] {p.get('name','?')} (art={p.get('artifact','?')})")

    append_to_proposals_doc(proposals, ctx)


if __name__ == "__main__":
    main()
