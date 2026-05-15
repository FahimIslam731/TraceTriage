from __future__ import annotations

"""Standardized evaluation utilities for Squad B classifiers.

All methods produce the same metrics dict structure so results
are directly comparable.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Compute standard classification metrics.

    Returns a dict with:
        accuracy, macro_f1, weighted_f1,
        per_class: {class: {precision, recall, f1, support}}
    """
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0,
    )

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "n_samples": len(y_true),
    }


def confusion_matrix_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> str:
    """Return a formatted confusion matrix string."""
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Header
    col_width = max(len(c) for c in class_names) + 2
    header = " " * col_width + "".join(c.rjust(col_width) for c in class_names)
    lines = [header]

    for i, cls in enumerate(class_names):
        row = cls.ljust(col_width) + "".join(
            str(cm[i, j]).rjust(col_width) for j in range(len(class_names))
        )
        lines.append(row)

    return "\n".join(lines)


def print_report(results: dict, method_name: str) -> None:
    """Print a formatted evaluation report to console."""
    print(f"\n{'=' * 60}")
    print(f"  {method_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro F1:    {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    print(f"  N samples:   {results['n_samples']}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<16} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-' * 48}")
    for cls, m in results["per_class"].items():
        print(
            f"  {cls:<16} {m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['support']:>8}"
        )
    print()


def save_results(results: dict, method_name: str, extra: Optional[dict] = None) -> Path:
    """Save results dict to squad_b/results/<method>_results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "method": method_name,
        "metrics": results,
    }
    if extra:
        output["extra"] = extra

    path = RESULTS_DIR / f"{method_name}_results.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
