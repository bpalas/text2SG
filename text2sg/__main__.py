"""CLI entry point -- python -m text2sg <command> [options]

Commands:
    run      Extract relations from text or a file
    models   List available models (Ollama local + cloud backends)

Examples:
    # given_entities -- you supply the actors
    python -m text2sg run \\
        --extractor ollama:qwen2.5:7b \\
        --actors "Gabriel Boric" "Camila Vallejo" "Jose Antonio Kast" \\
        --text "Boric respaldo las propuestas de Vallejo, mientras Kast las ataco."

    # end2end -- NER + extraction in one go (no --actors needed)
    python -m text2sg run --mode end2end \\
        --extractor ollama:qwen2.5:14b \\
        --file articulo.txt

    # mixed backends: cheap local NER, cloud for extraction, Claude to verify
    # (--genome optional; defaults to the built-in seed genome)
    python -m text2sg run --mode end2end \\
        --ner       ollama:qwen2.5:7b \\
        --extractor gemini:gemini-2.5-flash-lite \\
        --verifier  anthropic:claude-haiku-4-5 \\
        --file      articulo.txt

    # list what's available
    python -m text2sg models
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m text2sg",
        description="Extract signed political graphs from Spanish-language news.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # -- run ----------------------------------------------------------------- #
    rp = sub.add_parser("run", help="Extract relations from text or a file")
    rp.add_argument(
        "--mode", choices=["given_entities", "end2end"], default="given_entities",
        help="given_entities: you supply --actors; end2end: NER runs first (default: given_entities)",
    )
    rp.add_argument(
        "--extractor", required=True, metavar="BACKEND:MODEL",
        help="Main extraction LLM. E.g.: ollama:qwen2.5:7b  gemini:gemini-2.5-flash-lite",
    )
    rp.add_argument(
        "--ner", metavar="BACKEND:MODEL", default=None,
        help="NER model for end2end (default: same as --extractor)",
    )
    rp.add_argument(
        "--verifier", metavar="BACKEND:MODEL", default=None,
        help="Optional agentic verify pass. Omit to skip (saves tokens).",
    )
    rp.add_argument(
        "--genome", metavar="PATH", default=None,
        help="Path to genome JSON (prompt + configs). Default: built-in seed prompt.",
    )
    rp.add_argument("--text", metavar="TEXT", help="Article text (inline)")
    rp.add_argument("--file", metavar="PATH", help="Path to article text file")
    rp.add_argument(
        "--actors", nargs="*", metavar="NAME",
        help="Known actor names for given_entities mode. E.g.: --actors 'Gabriel Boric' 'Kast'",
    )
    rp.add_argument(
        "--output", choices=["pretty", "json"], default="pretty",
        help="Output format (default: pretty)",
    )
    rp.add_argument(
        "--log-dir", metavar="DIR", default="results/runs",
        help="Directory for per-run JSONL traces (default: results/runs)",
    )
    rp.add_argument(
        "--no-log", action="store_true",
        help="Don't save the JSONL trace file (the stderr trace table still prints)",
    )

    # -- models -------------------------------------------------------------- #
    sub.add_parser("models", help="List available models (Ollama + cloud backends)")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "models":
        _cmd_models()
    else:
        parser.print_help()


# -- command implementations ------------------------------------------------- #

def _cmd_run(args: argparse.Namespace) -> None:
    import time
    from text2sg.genome import Genome
    from text2sg.observability import RunLogger, format_trace
    from text2sg.pipeline import PipelineConfig, extract_text

    # -- load text -- #
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        print("[text2sg] Reading article from stdin (Ctrl-D to finish)...", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("[text2sg] Error: no text provided.", file=sys.stderr)
        sys.exit(1)

    # -- load genome -- #
    if args.genome:
        with open(args.genome, encoding="utf-8") as f:
            genome = Genome.from_json(f.read())
    else:
        genome = Genome.from_seed()

    # -- validate given_entities requirements -- #
    if args.mode == "given_entities" and not args.actors:
        print(
            "[text2sg] Warning: --mode given_entities but no --actors supplied.\n"
            "          The model will receive no actor list; consider --mode end2end.",
            file=sys.stderr,
        )

    # -- build pipeline config -- #
    config = PipelineConfig.from_cli_args(
        mode=args.mode,
        extractor=args.extractor,
        ner=args.ner,
        verifier=args.verifier,
    )

    # -- run logger -- #
    _now = time.time()
    run_id = (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(_now))}"
        f"{int((_now % 1) * 1000):03d}-{os.getpid()}"
    )
    logger = RunLogger(
        run_id=run_id,
        out_dir=args.log_dir,
        enabled=not args.no_log,
    )

    # -- log plan -- #
    print(f"[text2sg] mode      = {config.mode}", file=sys.stderr)
    print(f"[text2sg] extractor = {config.extractor}", file=sys.stderr)
    if config.mode == "end2end":
        print(f"[text2sg] ner       = {config.ner}", file=sys.stderr)
    if config.verifier:
        print(f"[text2sg] verifier  = {config.verifier}", file=sys.stderr)
    print(f"[text2sg] genome    = {args.genome or 'seed (default)'}", file=sys.stderr)
    print(file=sys.stderr)

    # -- run -- #
    result = extract_text(
        text, genome, config,
        actors=args.actors,
        article_id=args.file or "stdin",
        logger=logger,
    )

    # -- output -- #
    if args.output == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _pretty_print(result)

    # -- trace -- #
    print(format_trace(logger.events), file=sys.stderr)
    if logger.path:
        print(f"[text2sg] trace saved to {logger.path}", file=sys.stderr)


def _pretty_print(result: dict) -> None:
    rels = result.get("relations", [])
    tok = result.get("tokens", {})

    print(f"\n{'=' * 64}")
    print(f"  Relations: {len(rels)}")
    t_str = "  |  ".join(
        f"{k}: {v}" for k, v in tok.items() if k != "total" and v > 0
    )
    print(f"  Tokens:    total={tok.get('total', 0)}  ({t_str})")
    print(f"{'=' * 64}")
    if not rels:
        print("  (no relations extracted)")
    for i, r in enumerate(rels, 1):
        pol_sym = {"positive": "(+)", "negative": "(-)", "neutral": "(~)"}.get(
            r.get("polarity", ""), "(?)"
        )
        print(
            f"  {i:2}. {r.get('from_entity', '?')} "
            f"--[{r.get('act_type', '?')} {pol_sym}]--> "
            f"{r.get('to_entity', '?')}"
        )
        quote = r.get("evidence_quote", "")
        if quote:
            snippet = quote[:90] + ("..." if len(quote) > 90 else "")
            print(f"       -> \"{snippet}\"")
    print()


def _cmd_models() -> None:
    import urllib.request

    print("\n-- Ollama (local, $0) " + "-" * 42)
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
        models = data.get("models", [])
        if models:
            for m in models:
                size_gb = m.get("size", 0) / 1e9
                print(f"  ollama:{m['name']:<30} {size_gb:.1f} GB")
        else:
            print("  (no models pulled yet -- run: ollama pull qwen2.5:7b)")
    except Exception:
        print("  Ollama not running.  Start with: ollama serve")

    print("\n-- Cloud APIs " + "-" * 49)
    print("  gemini:gemini-2.5-flash-lite    GEMINI_API_KEY")
    print("  gemini:gemini-2.5-flash         GEMINI_API_KEY")
    print("  anthropic:claude-haiku-4-5      ANTHROPIC_API_KEY")
    print("  anthropic:claude-sonnet-4-6     ANTHROPIC_API_KEY")
    print("  openai:gpt-4o-mini              OPENAI_API_KEY")
    print()
    print("Usage:  python -m text2sg run --extractor BACKEND:MODEL --text '...'")
    print()


if __name__ == "__main__":
    main()
