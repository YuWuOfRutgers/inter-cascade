"""Command-line interface.

The full workflow, end to end::

    inter-cascade init                          # write a starter config
    # edit inter_cascade.yaml, export API keys
    inter-cascade prepare --dataset gsm8k --output-dir data   # or bring your own JSON
    inter-cascade calibrate --config inter_cascade.yaml       # derive thresholds
    inter-cascade run --config inter_cascade.yaml \
        --input data/test_gsm8k.json --output results.json    # batch evaluation
    inter-cascade query --config inter_cascade.yaml "your question here"
    inter-cascade stats --results results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLE_CONFIG = """\
# Inter-Cascade configuration. API keys come from environment variables only.

weak:
  provider: openai            # openai | gemini; openai also covers vLLM/Ollama etc.
  model: gpt-3.5-turbo
  api_key_env: OPENAI_API_KEY
  # base_url: http://localhost:8000/v1   # uncomment for a local open-source model
  temperature: 0.0

strong:
  provider: gemini
  model: gemini-2.0-flash
  api_key_env: GEMINI_API_KEY
  # project_id: your-gcp-project        # use Vertex AI instead of an API key

# Deferral rules: give alpha/delta and `calibrate` derives the threshold with
# a provable risk bound; or pin `threshold` directly; or leave all null to
# always accept (common for the strong model).
weak_deferral:
  alpha: 0.6
  delta: 0.6

strong_deferral:
  alpha: null
  delta: null

repository:
  dir: strategy_repo
  embedding_model: all-MiniLM-L6-v2
  top_k: 2
  similarity_threshold: null   # keep retrieved docs with cosine score > this
  retrieval: similarity        # similarity | random (ablation) | none (baseline cascade)
  include_strategy: true       # false = no-strategy ablation
  write: true                  # false = frozen repository
  seed_from_calibration: true  # false = cold start

task:
  answer_type: numeric         # numeric | multiple_choice | freeform
  judge: strong                # grade answers with the strong model when needed

calibration_file: data/calibration_gsm8k.json
artifacts_dir: artifacts
"""


def _load(config_path: str):
    from inter_cascade.config import load_config

    return load_config(config_path)


def cmd_init(args) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(f"Wrote starter config to {path}.")
    print("Next: export OPENAI_API_KEY / GEMINI_API_KEY, prepare a calibration set,")
    print("then run: inter-cascade calibrate --config", path)
    return 0


def cmd_prepare(args) -> int:
    from inter_cascade.datasets import prepare

    kwargs = {}
    if args.calib_ratio is not None:
        kwargs["calib_ratio"] = args.calib_ratio
    if args.seed is not None:
        kwargs["seed"] = args.seed
    prepare(args.dataset, args.output_dir, **kwargs)
    return 0


def cmd_calibrate(args) -> int:
    from inter_cascade.runner import run_calibration

    cfg = _load(args.config)
    roles = ["weak", "strong"] if args.role == "both" else [args.role]
    for role in roles:
        run_calibration(cfg, role, limit=args.limit, scan=args.scan)
    return 0


def cmd_run(args) -> int:
    from inter_cascade.runner import run_eval

    cfg = _load(args.config)
    run_eval(
        cfg,
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        resume=args.resume,
        seed=args.seed,
    )
    return 0


def cmd_query(args) -> int:
    from inter_cascade.runner import build_cascade

    cfg = _load(args.config)
    cascade = build_cascade(cfg)
    result = cascade.answer(args.question)

    if args.json:
        payload = {
            "answer": result.answer,
            "source": result.source,
            "weak_confidence": result.weak.confidence if result.weak else None,
            "strong_confidence": result.strong.confidence if result.strong else None,
            "retrieved_strategies": len(result.retrieved),
            "repository_size": len(cascade.repository) if cascade.repository else 0,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"answer : {result.answer}")
        print(f"source : {result.source}")
        if result.weak:
            print(f"weak   : answer={result.weak.answer!r} confidence={result.weak.confidence}")
        if result.strong:
            print(f"strong : answer={result.strong.answer!r} confidence={result.strong.confidence}")
        print(f"retrieved strategies: {len(result.retrieved)}")
    return 0 if result.source != "rejected" else 2


def cmd_stats(args) -> int:
    from inter_cascade.metrics import compute_metrics, format_metrics

    with open(args.results, "r", encoding="utf-8") as f:
        records = json.load(f)
    metrics = compute_metrics(records)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_metrics(metrics))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="inter-cascade",
        description="Inter-Cascade: an online, interactive LLM cascade with a strategy repository.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter YAML config")
    p.add_argument("--path", default="inter_cascade.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("prepare", help="download a benchmark into calibration/test JSON")
    p.add_argument(
        "--dataset",
        required=True,
        help="gsm_symbolic | gsm_plus | metamath | nasa_history | "
        "gsm8k | bigbench | barexam | medmcqa",
    )
    p.add_argument("--output-dir", default="data")
    p.add_argument("--calib-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("calibrate", help="run the calibration set and derive deferral thresholds")
    p.add_argument("--config", required=True)
    p.add_argument("--role", choices=["weak", "strong", "both"], default="both")
    p.add_argument("--limit", type=int, default=None, help="use only the first N calibration items")
    p.add_argument("--scan", action="store_true", help="also record an alpha/delta grid scan")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run", help="stream a test set through the cascade and grade it")
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True, help="JSON list of {question, ground_truth}")
    p.add_argument("--output", required=True, help="where per-query result records are written")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--seed", type=int, default=None, help="seed for the random-retrieval ablation")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("query", help="answer a single question through the cascade")
    p.add_argument("question")
    p.add_argument("--config", required=True)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("stats", help="compute paper metrics from a results file")
    p.add_argument("--results", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
