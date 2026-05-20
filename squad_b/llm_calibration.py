from __future__ import annotations

"""Calibration utilities for cached LLM predictions.

This module does not call any LLM APIs. It reads cached predictions produced by
``llm_classifier.py`` when run with JSON/confidence output enabled.
"""

import json
from pathlib import Path

import numpy as np

from .data_loader import RANDOM_SEED, build_dataset, get_train_test_indices
from .evaluator import save_results
from .llm_classifier import CACHE_DIR


def _ece(rows: list[dict], num_bins: int = 10) -> dict:
    if not rows:
        return {"ece": None, "bins": [], "n": 0}

    bins = []
    ece = 0.0
    n = len(rows)
    for bin_idx in range(num_bins):
        lo = bin_idx / num_bins
        hi = (bin_idx + 1) / num_bins
        if bin_idx == num_bins - 1:
            selected = [r for r in rows if lo <= r["confidence"] <= hi]
        else:
            selected = [r for r in rows if lo <= r["confidence"] < hi]
        if not selected:
            bins.append({
                "bin_start": lo,
                "bin_end": hi,
                "count": 0,
                "accuracy": None,
                "confidence": None,
            })
            continue

        accuracy = float(np.mean([r["correct"] for r in selected]))
        confidence = float(np.mean([r["confidence"] for r in selected]))
        ece += (len(selected) / n) * abs(accuracy - confidence)
        bins.append({
            "bin_start": lo,
            "bin_end": hi,
            "count": len(selected),
            "accuracy": round(accuracy, 4),
            "confidence": round(confidence, 4),
        })

    return {"ece": round(float(ece), 4), "bins": bins, "n": n}


def run_llm_calibration(cache_path: str | None = None, num_bins: int = 10) -> dict:
    """Compute ECE from cached LLM predictions with confidence values."""
    if cache_path is None:
        candidates = sorted(CACHE_DIR.glob("llm_*_json_predictions.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No JSON-confidence LLM cache found in {CACHE_DIR}. "
                "Run an LLM task with --json-output after setting an API key."
            )
        path = candidates[-1]
    else:
        path = Path(cache_path)

    cached = json.loads(path.read_text(encoding="utf-8"))
    ds = build_dataset(use_sqlite_features=False)
    labels = ds["labels"]
    trace_ids = ds["trace_ids"]
    _, test_idx, _ = get_train_test_indices(ds, labels, seed=RANDOM_SEED)
    gold_by_id = {trace_ids[i]: labels[i] for i in test_idx}

    rows = []
    skipped = 0
    for trace_id, pred in cached.items():
        if trace_id not in gold_by_id:
            continue
        confidence = pred.get("confidence")
        if confidence is None:
            skipped += 1
            continue
        rows.append({
            "trace_id": trace_id,
            "gold": gold_by_id[trace_id],
            "prediction": pred.get("action"),
            "confidence": float(confidence),
            "correct": pred.get("action") == gold_by_id[trace_id],
        })

    result = _ece(rows, num_bins=num_bins)
    result.update({
        "cache_path": str(path),
        "skipped_without_confidence": skipped,
    })
    save_results(result, "llm_calibration")

    print(f"Calibration cache: {path}")
    print(f"Examples with confidence: {result['n']}")
    print(f"Skipped without confidence: {skipped}")
    print(f"ECE: {result['ece']}")
    return result
