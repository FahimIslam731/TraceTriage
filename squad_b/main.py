"""Squad B CLI — run classification experiments.

Usage:
    python -m squad_b.main --task tfidf
    python -m squad_b.main --task embedding
    python -m squad_b.main --task llm-zero
    python -m squad_b.main --task llm-few
    python -m squad_b.main --task llm-zero --limit 10
    python -m squad_b.main --task all
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Squad B classification experiments")
    parser.add_argument(
        "--task",
        required=True,
        choices=["tfidf", "embedding", "llm-zero", "llm-few", "all"],
        help="Which experiment to run",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max test samples for LLM tasks (cost control)")
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview",
                        help="OpenRouter model for LLM tasks")
    parser.add_argument("--k-per-class", type=int, default=1,
                        help="Few-shot examples per class (default: 1)")
    parser.add_argument("--no-structured", action="store_true",
                        help="TF-IDF only, skip structured features")
    args = parser.parse_args()

    tasks = [args.task] if args.task != "all" else ["tfidf", "embedding", "llm-zero", "llm-few"]

    for task in tasks:
        print(f"\n{'#' * 60}")
        print(f"# Running: {task}")
        print(f"{'#' * 60}\n")

        if task == "tfidf":
            from .tfidf_baseline import run_tfidf_baseline
            run_tfidf_baseline(use_structured=not args.no_structured)

        elif task == "embedding":
            from .embedding_pipeline import run_embedding_pipeline
            run_embedding_pipeline()

        elif task == "llm-zero":
            from .llm_classifier import run_llm_classification
            run_llm_classification(mode="zero", limit=args.limit, model=args.model)

        elif task == "llm-few":
            from .llm_classifier import run_llm_classification
            run_llm_classification(mode="few", limit=args.limit, model=args.model,
                                   k_per_class=args.k_per_class)

    print("\nDone. Results saved to squad_b/results/")


if __name__ == "__main__":
    main()
