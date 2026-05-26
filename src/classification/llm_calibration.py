from __future__ import annotations

"""Calibration utilities for cached LLM predictions.

This module does not call any LLM APIs. It reads cached predictions produced by
``llm_classifier.py`` when run with JSON/confidence output enabled.
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .data_loader import RANDOM_SEED, build_dataset, get_train_test_indices
from .evaluator import RESULTS_DIR, save_results
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


def _save_reliability_svg(result: dict, output_path: Path, title: str) -> None:
    """Save a simple reliability diagram as SVG without extra dependencies."""
    width, height = 720, 520
    margin_left, margin_right = 72, 32
    margin_top, margin_bottom = 64, 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x(value: float) -> float:
        return margin_left + value * plot_w

    def y(value: float) -> float:
        return margin_top + (1.0 - value) * plot_h

    bins = result.get("bins", [])
    num_bins = max(1, len(bins))
    bar_gap = 5
    bar_w = max(4, plot_w / num_bins - bar_gap)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">{escape(title)}</text>',
        f'<text x="{width / 2}" y="52" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#555">ECE={result.get("ece")} · n={result.get("n")}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#888" stroke-width="2" stroke-dasharray="6 5"/>',
    ]

    for tick in np.linspace(0, 1, 6):
        tx, ty = x(float(tick)), y(float(tick))
        parts.append(f'<line x1="{tx:.1f}" y1="{height - margin_bottom}" x2="{tx:.1f}" y2="{height - margin_bottom + 6}" stroke="#222"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{tick:.1f}</text>')
        parts.append(f'<line x1="{margin_left - 6}" y1="{ty:.1f}" x2="{margin_left}" y2="{ty:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{tick:.1f}</text>')
        if 0 < tick < 1:
            parts.append(f'<line x1="{margin_left}" y1="{ty:.1f}" x2="{width - margin_right}" y2="{ty:.1f}" stroke="#eee"/>')

    for idx, bin_info in enumerate(bins):
        accuracy = bin_info.get("accuracy")
        count = bin_info.get("count", 0)
        if accuracy is None or count == 0:
            continue
        left = margin_left + idx * (plot_w / num_bins) + bar_gap / 2
        top = y(float(accuracy))
        bottom = y(0)
        parts.append(
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bottom - top:.1f}" '
            'fill="#4C78A8" opacity="0.82"/>'
        )
        parts.append(
            f'<text x="{left + bar_w / 2:.1f}" y="{top - 6:.1f}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="10" fill="#333">'
            f'{count}</text>'
        )

    parts.extend([
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Mean confidence bin</text>',
        f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Accuracy</text>',
        '<text x="540" y="92" font-family="Arial, sans-serif" font-size="12" fill="#555">Dashed line = perfect calibration</text>',
        '</svg>',
    ])
    output_path.write_text("\n".join(parts), encoding="utf-8")


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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_DIR / f"llm_calibration_{path.stem}.svg"
    _save_reliability_svg(
        result,
        plot_path,
        title=f"LLM Calibration: {path.stem}",
    )
    result.update({
        "cache_path": str(path),
        "plot_path": str(plot_path),
        "skipped_without_confidence": skipped,
    })
    save_results(result, "llm_calibration")

    print(f"Calibration cache: {path}")
    print(f"Examples with confidence: {result['n']}")
    print(f"Skipped without confidence: {skipped}")
    print(f"ECE: {result['ece']}")
    print(f"Reliability plot: {plot_path}")
    return result
