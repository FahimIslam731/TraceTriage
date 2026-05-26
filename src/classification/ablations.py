from __future__ import annotations

"""Input-variant ablations for Squad B Experiment 2."""

from .data_loader import INPUT_VARIANTS
from .evaluator import save_results
from .tfidf_baseline import run_tfidf_baseline


def run_input_ablations(use_structured: bool = False) -> dict:
    """Run TF-IDF baselines across all requested input variants.

    The paper brief's input ablations are intentionally text/view ablations, so
    ``use_structured`` defaults to False. Pass ``--structured-ablations`` from
    the CLI if you also want the same variants with structured features stacked.
    """
    summary = []
    by_variant = {}

    for variant in INPUT_VARIANTS:
        print(f"\n{'#' * 60}")
        print(f"# Input ablation: {variant}")
        print(f"{'#' * 60}\n")
        results = run_tfidf_baseline(
            use_structured=use_structured,
            input_variant=variant,
            result_prefix=f"ablation_{variant}",
        )
        by_variant[variant] = results
        logreg = results["tfidf_logreg"]
        rf = results["tfidf_rf"]
        domain_only = results["domain_only"]
        summary.append({
            "input_variant": variant,
            "logreg_macro_f1": logreg["macro_f1"],
            "rf_macro_f1": rf["macro_f1"],
            "domain_only_macro_f1": domain_only["macro_f1"],
            "best_macro_f1": max(logreg["macro_f1"], rf["macro_f1"]),
            "best_model": "tfidf_logreg" if logreg["macro_f1"] >= rf["macro_f1"] else "tfidf_rf",
            "n_samples": logreg["n_samples"],
            "use_structured": use_structured,
        })

    output = {
        "use_structured": use_structured,
        "summary": summary,
        "by_variant": by_variant,
    }
    save_results(output, "input_ablation_summary")

    print("\nInput ablation summary:")
    for row in summary:
        print(
            f"  {row['input_variant']:<24} "
            f"best={row['best_model']:<12} "
            f"best_macro_f1={row['best_macro_f1']:.4f} "
            f"domain_only={row['domain_only_macro_f1']:.4f}"
        )

    return output
