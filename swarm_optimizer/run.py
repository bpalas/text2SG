# experiments/swarm_optimizer/run.py
"""
CLI del swarm optimizer.

Uso:
    python -m swarm_optimizer.run --probe              # verifica pipeline antes de gastar
    python -m swarm_optimizer.run --iterations 20 --budget 8.0
    python -m swarm_optimizer.run --multi-seed --iterations 20 --budget 8.0
    python -m swarm_optimizer.run --meta-policy --iterations 40 --budget 8.0
    python -m swarm_optimizer.run --splits-only
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Cargar .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # si no está instalado, el usuario debe setear GEMINI_API_KEY manualmente


def _run_probe(client, splits: dict, articles_df, gold_df, n: int = 3,
               model: str | None = None) -> bool:
    """Extrae n artículos con la semilla base y muestra métricas.

    Cuesta <$0.01 y confirma que el pipeline funciona antes de un loop completo.
    Devuelve True si la extracción parece funcional.
    """
    from swarm_optimizer.extractor import run_extraction
    from swarm_optimizer.fitness import fitness
    from swarm_optimizer.genome import Genome
    from swarm_optimizer.rubric import compute_metrics, load_union_map
    from swarm_optimizer.splits import subsample

    seed = Genome.from_seed()
    if model:
        seed.model = model
    ids = subsample(splits["eval"], min(n, len(splits["eval"])), seed=0)
    union_map = load_union_map(ids)

    print(f"[probe] Extrayendo {len(ids)} artículo(s) con semilla base...")
    try:
        preds, total_tokens = run_extraction(ids, articles_df, gold_df, union_map, seed, client)
    except Exception as exc:
        print(f"[probe] ERROR durante extracción: {exc}")
        return False

    n_rels = sum(len(p.get("relations", [])) for p in preds)
    print(f"[probe] {len(ids)} artículos -> {n_rels} relaciones, {total_tokens} tokens")

    if n_rels == 0:
        print("[probe] ERROR: 0 relaciones extraídas. Posibles causas:")
        print("  - GEMINI_API_KEY inválida o expirada")
        print("  - El modelo no está devolviendo JSON válido")
        print("  - La validación está filtrando todas las relaciones (ej: min_quote_len)")
        print("  Sugerencia: revisar stderr arriba por mensajes de extractor")
        return False

    metrics = compute_metrics(preds, ids, gold_df, union_map)
    tokens_per_art = total_tokens / max(len(ids), 1)
    score = fitness(metrics, tokens_per_art, model=seed.model)
    cost_est = total_tokens * 0.30 / 1_000_000

    print(f"  Precision_rel={metrics.get('Precision_rel', 0):.3f}  "
          f"Recall_rel={metrics.get('Recall_rel', 0):.3f}  "
          f"Precision_ent={metrics.get('Precision_ent', 0):.3f}")
    print(f"  fitness={score:.4f}  costo_aprox=${cost_est:.5f}")
    print("[probe] OK — el pipeline funciona. Puede lanzar el loop.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm optimizer — loop RL para extracción política"
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--budget", type=float, default=8.0, help="Presupuesto en USD")
    parser.add_argument("--subsample-k", type=int, default=12,
                        help="Artículos por iteración (undersampling RoboPhD)")
    parser.add_argument("--championship-every", type=int, default=5,
                        help="Cada cuántas iteraciones se corre el championship anclado")
    parser.add_argument("--cross-every", type=int, default=7,
                        help="Cada cuántas iteraciones se hace cross-pollination")
    parser.add_argument("--gate-k", type=int, default=4,
                        help="Artículos del gate 1 de la cascada (0 = sin cascada)")
    parser.add_argument("--gate-epsilon", type=float, default=0.05,
                        help="Margen eps del gate 1 (descarta si child < champ - eps)")
    parser.add_argument("--meta-policy", action="store_true",
                        help="Usa el bandit Thompson (meta-agente) en vez del calendario fijo")
    parser.add_argument("--meta-policy-path", type=str, default=None,
                        help="Ruta al JSON del posterior del bandit "
                             "(default: results/swarm/policy.json). "
                             "Acumula priors entre corridas.")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Siembra 3 variantes (base, verify, debate) y deja competir")
    parser.add_argument(
        "--splits-only", action="store_true", help="Solo genera splits.json y sale"
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Extrae 3 artículos con la semilla base y muestra métricas. "
             "Úsalo antes de un loop completo para confirmar que el pipeline funciona. "
             "Cuesta <$0.01 y no modifica el archivo evolutivo."
    )
    parser.add_argument("--probe-n", type=int, default=3,
                        help="Artículos a extraer en modo --probe (default: 3)")
    parser.add_argument("--llm", choices=["auto", "gemini", "ollama", "anthropic"],
                        default="auto",
                        help="Backend LLM: auto (Gemini si hay key, sino Ollama), "
                             "gemini, ollama, o anthropic (requiere ANTHROPIC_API_KEY)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override del modelo del genoma semilla (ej: "
                             "claude-haiku-4-5, claude-sonnet-4-6, gemini-2.5-flash)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from swarm_optimizer.splits import load_splits
    splits = load_splits()
    print(
        f"Splits: eval={len(splits['eval'])} artículos, "
        f"test={len(splits['test'])} artículos"
    )

    if args.splits_only:
        print("Splits guardados.")
        return

    # Seleccionar cliente LLM
    client = None
    if args.llm == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: --llm anthropic requiere ANTHROPIC_API_KEY en el entorno.")
            sys.exit(1)
        from swarm_optimizer.llm_backends import AnthropicClient
        client = AnthropicClient()
        print("Usando Anthropic API (Claude).")
        if not args.model:
            print("AVISO: el genoma semilla usa gemini-2.5-flash como `model`. "
                  "Pasa --model claude-haiku-4-5 (o claude-sonnet-4-6) para Claude.")
    elif args.llm == "gemini" or (args.llm == "auto" and os.environ.get("GEMINI_API_KEY")):
        if not os.environ.get("GEMINI_API_KEY"):
            print("ERROR: --llm gemini requiere GEMINI_API_KEY en el entorno.")
            sys.exit(1)
        print("Usando Gemini API.")
    else:
        if args.llm == "auto":
            print("GEMINI_API_KEY no definida. Intentando usar Ollama...")
        try:
            from swarm_optimizer.llm_backends import OllamaClient
            client = OllamaClient()
            print("Usando Ollama (http://localhost:11434)")
        except Exception as e:
            print(f"ERROR: No se pudo conectar a Ollama: {e}")
            print("  Opciones:")
            print("  1. Inicia Ollama: ollama serve")
            print("  2. O seteá GEMINI_API_KEY: $env:GEMINI_API_KEY = 'tu-key'")
            print("  3. O usa --llm anthropic con ANTHROPIC_API_KEY")
            sys.exit(1)

    # Override del modelo de las semillas (necesario al cambiar de backend)
    seed_model_override = args.model

    if args.probe:
        import pandas as pd
        from pathlib import Path as _P
        _ARTICLES = _P(__file__).parent.parent.parent / "gold_standard_v5/data/pilot_gold_articles.parquet"
        _GOLD = _P(__file__).parent.parent.parent / "gold_standard_v5/data/pilot_gold_final.parquet"
        articles_df = pd.read_parquet(_ARTICLES)
        gold_df = pd.read_parquet(_GOLD)
        ok = _run_probe(client, splits, articles_df, gold_df, n=args.probe_n,
                        model=seed_model_override)
        sys.exit(0 if ok else 1)

    from swarm_optimizer.genome import Genome
    from swarm_optimizer.loop import run_loop

    seed_genomes = Genome.seed_variants() if args.multi_seed else None
    if seed_model_override:
        if seed_genomes is None:
            seed_genomes = [Genome.from_seed()]
        for g in seed_genomes:
            g.model = seed_model_override

    default_policy_path = Path(__file__).parent.parent / "results/swarm/policy.json"
    policy_path = Path(args.meta_policy_path) if args.meta_policy_path else default_policy_path

    run_loop(
        max_iter=args.iterations,
        budget_usd=args.budget,
        subsample_k=args.subsample_k,
        championship_every=args.championship_every,
        cross_every=args.cross_every,
        gate_k=args.gate_k,
        gate_epsilon=args.gate_epsilon,
        use_meta_policy=args.meta_policy,
        meta_policy_path=policy_path if args.meta_policy else None,
        seed_genomes=seed_genomes,
        verbose=not args.quiet,
        client=client,
    )


if __name__ == "__main__":
    main()
