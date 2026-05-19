"""Squad B CLI — run classification experiments.

Usage:
    python -m squad_b.main --task tfidf
    python -m squad_b.main --task embedding
    python -m squad_b.main --task llm-zero
    python -m squad_b.main --task llm-few
    python -m squad_b.main --task cross-domain
    python -m squad_b.main --task ablations
    python -m squad_b.main --task llm-zero --limit 10
    python -m squad_b.main --task llm-zero --model-preset gpt5-mini --json-output
    python -m squad_b.main --task all
"""
import argparse

from .data_loader import INPUT_VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Squad B classification experiments")
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "tfidf", "embedding", "llm-zero", "llm-few", "llm-calibration",
            "cross-domain", "ablations", "all-local", "all",
        ],
        help="Which experiment to run",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max test samples for LLM tasks (cost control)")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit OpenRouter model ID for LLM tasks")
    parser.add_argument("--model-preset", type=str, default="default",
                        choices=["default", "gpt5-mini", "gemini-flash-lite"],
                        help="Named LLM model preset")
    parser.add_argument("--k-per-class", type=int, default=1,
                        help="Few-shot examples per class (default: 1)")
    parser.add_argument("--json-output", action="store_true",
                        help="Ask LLM tasks for JSON action/confidence output")
    parser.add_argument("--calibration-cache", type=str, default=None,
                        help="Cached LLM prediction JSON to calibrate")
    parser.add_argument("--input-variant", type=str, default="full_trace",
                        choices=INPUT_VARIANTS,
                        help="Input view for TF-IDF/embedding/LLM tasks")
    parser.add_argument("--no-structured", action="store_true",
                        help="TF-IDF only, skip structured features")
    parser.add_argument("--structured-ablations", action="store_true",
                        help="Run input ablations with structured features stacked")
    parser.add_argument("--min-domain-samples", type=int, default=10,
                        help="Minimum examples required for cross-domain held-out evaluation")
    args = parser.parse_args()

    if args.task == "all-local":
        tasks = ["tfidf", "ablations", "cross-domain"]
    elif args.task == "all":
        tasks = ["tfidf", "embedding", "llm-zero", "llm-few", "cross-domain"]
    else:
        tasks = [args.task]

    for task in tasks:
        print(f"\n{'#' * 60}")
        print(f"# Running: {task}")
        print(f"{'#' * 60}\n")

        if task == "tfidf":
            from .tfidf_baseline import run_tfidf_baseline
            run_tfidf_baseline(
                use_structured=not args.no_structured,
                input_variant=args.input_variant,
            )

        elif task == "embedding":
            from .embedding_pipeline import run_embedding_pipeline
            run_embedding_pipeline(input_variant=args.input_variant)

        elif task == "llm-zero":
            from .llm_classifier import run_llm_classification
            run_llm_classification(
                mode="zero",
                limit=args.limit,
                model=args.model,
                model_preset=args.model_preset,
                json_output=args.json_output,
                input_variant=args.input_variant,
            )

        elif task == "llm-few":
            from .llm_classifier import run_llm_classification
            run_llm_classification(mode="few", limit=args.limit, model=args.model,
                                   model_preset=args.model_preset,
                                   k_per_class=args.k_per_class,
                                   json_output=args.json_output,
                                   input_variant=args.input_variant)

        elif task == "llm-calibration":
            from .llm_calibration import run_llm_calibration
            run_llm_calibration(cache_path=args.calibration_cache)

        elif task == "cross-domain":
            from .cross_domain import run_cross_domain_evaluation
            run_cross_domain_evaluation(
                use_structured=not args.no_structured,
                min_domain_samples=args.min_domain_samples,
            )

        elif task == "ablations":
            from .ablations import run_input_ablations
            run_input_ablations(use_structured=args.structured_ablations)

    print("\nDone. Results saved to squad_b/results/")


if __name__ == "__main__":
    main()
