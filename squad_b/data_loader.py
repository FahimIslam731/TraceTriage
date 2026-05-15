from __future__ import annotations

"""Data loading and feature engineering for Squad B classification.

Loads the 638 GPT-labeled failed traces from JSONL + SQLite,
produces text representations and structured features for classifiers.

Reproducibility: all splits use RANDOM_SEED = 42.
"""
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "causal_runs.sqlite"
GPT_LABELS_PATH = PROJECT_ROOT / "data" / "labeling_exports" / "gpt_auto_labels_kavin.jsonl"
LLAMA_LABELS_PATH = PROJECT_ROOT / "data" / "labeling_exports" / "llama_auto_labels_kavin.jsonl"
FAILED_TRACES_PATH = PROJECT_ROOT / "data" / "labeling_exports" / "failed_traces.jsonl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TARGET_CLASSES = ["RETRIEVE_MORE", "REPLAN", "TOOL_FIX", "RETRY", "ESCALATE"]
TOOL_NAMES = [
    "web_search", "web_fetch", "calculator",
    "docker_code_execution", "llm_code_generation",
    "docker_code_execution_result",
]


# ---------------------------------------------------------------------------
# Core data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labeled_traces() -> list[dict]:
    """Load 638 traces with GPT labels joined to full trace data.

    Returns a list of dicts, each containing:
        - All fields from failed_traces.jsonl (trace_id, steps, problem_statement, …)
        - 'label': the GPT-assigned action (ground truth)
        - 'llama_label': the Llama-assigned action (for agreement analysis)
        - 'gpt_rationale', 'gpt_confidence'
    """
    # Load labels keyed by trace_id
    gpt_labels = {r["trace_id"]: r for r in _load_jsonl(GPT_LABELS_PATH)}
    llama_labels = {r["trace_id"]: r for r in _load_jsonl(LLAMA_LABELS_PATH)}

    # Load full trace data
    failed_traces = _load_jsonl(FAILED_TRACES_PATH)

    merged = []
    for trace in failed_traces:
        tid = trace["trace_id"]
        gpt = gpt_labels.get(tid)
        if gpt is None:
            continue  # skip traces without GPT labels

        llama = llama_labels.get(tid, {})
        trace["label"] = gpt["action"]
        trace["llama_label"] = llama.get("action")
        trace["gpt_rationale"] = gpt.get("rationale", "")
        trace["gpt_confidence"] = gpt.get("confidence")
        merged.append(trace)

    return merged


# ---------------------------------------------------------------------------
# Feature engineering — text
# ---------------------------------------------------------------------------

def flatten_trace_to_text(trace: dict) -> str:
    """Convert a trace dict into a single text string for TF-IDF / embeddings.

    Format:
        [DOMAIN] <domain>
        [PROBLEM] <problem_statement>
        [STEP 0] type=<step_type> tool=<tool_name>
          <step text (truncated)>
        ...
        [FINAL_ANSWER] <final_answer>
    """
    parts = [f"[DOMAIN] {trace.get('domain', '')}"]
    parts.append(f"[PROBLEM] {trace.get('problem_statement', '')}")

    for step in trace.get("steps", []):
        header = f"[STEP {step.get('step_index', '?')}] type={step.get('step_type', '')}"
        if step.get("tool_name"):
            header += f" tool={step['tool_name']}"
        parts.append(header)

        text = step.get("text") or ""
        if text:
            parts.append(f"  {text[:600]}")

        tool_out = step.get("tool_output_json") or ""
        if tool_out:
            parts.append(f"  output: {str(tool_out)[:400]}")

    parts.append(f"[FINAL_ANSWER] {trace.get('final_answer', '')}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Feature engineering — structured (numerical / categorical)
# ---------------------------------------------------------------------------

def extract_structured_features(trace: dict) -> dict[str, float]:
    """Extract numerical/categorical features from a single trace.

    Returns a flat dict of floats suitable for sklearn.
    """
    steps = trace.get("steps", [])
    num_steps = len(steps)
    tool_calls = [s for s in steps if s.get("has_tool") or s.get("tool_name")]
    num_tool_calls = len(tool_calls)
    tool_failures = [s for s in steps if s.get("tool_call_result") == 0]
    num_tool_failures = len(tool_failures)

    # Tool presence flags
    tools_used = {s.get("tool_name") for s in steps if s.get("tool_name")}

    features: dict[str, float] = {
        "num_steps": float(num_steps),
        "num_tool_calls": float(num_tool_calls),
        "num_tool_failures": float(num_tool_failures),
        "tool_failure_rate": (
            num_tool_failures / num_tool_calls if num_tool_calls > 0 else 0.0
        ),
        "problem_length": float(len(trace.get("problem_statement", "") or "")),
        "final_answer_length": float(len(trace.get("final_answer", "") or "")),
        "is_local_repairable": float(trace.get("is_local_repairable", 0) or 0),
        "num_successful_repair_steps": float(
            trace.get("num_successful_repair_steps", 0) or 0
        ),
    }

    # One-hot encode domain
    for domain in ["GSM8K", "MBPP", "MedBrowseComp", "SealQA", "BrowseComp"]:
        features[f"domain_{domain}"] = 1.0 if trace.get("domain") == domain else 0.0

    # One-hot encode tool usage
    for tool in TOOL_NAMES:
        features[f"tool_{tool}"] = 1.0 if tool in tools_used else 0.0

    # Step type counts
    step_types = [s.get("step_type", "") for s in steps]
    for stype in ["reasoning", "tool_call", "tool_response", "final_answer", "llm_response"]:
        features[f"steptype_{stype}"] = float(step_types.count(stype))

    return features


def extract_structured_features_from_sqlite(trace_ids: list[str]) -> dict[str, dict[str, float]]:
    """Bulk-load trace_metrics features from SQLite for the given trace_ids.

    Returns {trace_id: {feature_name: value}}.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join(["?"] * len(trace_ids))
    rows = conn.execute(
        f"""SELECT trace_id, minimality_average, num_identified_causal_steps,
                   attribution_precision, attribution_recall, attribution_f1,
                   repairs_attempted, repairs_successful, repairs_failed,
                   repair_success_rate, num_successful_repair_steps,
                   num_consensus_steps
            FROM trace_metrics
            WHERE trace_id IN ({placeholders})""",
        trace_ids,
    ).fetchall()
    conn.close()

    result = {}
    for row in rows:
        result[row["trace_id"]] = {
            "minimality_avg": float(row["minimality_average"] or 0),
            "num_causal_steps": float(row["num_identified_causal_steps"] or 0),
            "attr_precision": float(row["attribution_precision"] or 0),
            "attr_recall": float(row["attribution_recall"] or 0),
            "attr_f1": float(row["attribution_f1"] or 0),
            "repairs_attempted": float(row["repairs_attempted"] or 0),
            "repairs_successful": float(row["repairs_successful"] or 0),
            "repairs_failed": float(row["repairs_failed"] or 0),
            "repair_success_rate": float(row["repair_success_rate"] or 0),
            "num_successful_repairs": float(row["num_successful_repair_steps"] or 0),
            "num_consensus_steps": float(row["num_consensus_steps"] or 0),
        }
    return result


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    use_sqlite_features: bool = True,
) -> dict[str, Any]:
    """Build the full classification dataset.

    Returns dict with keys:
        traces: list[dict]          — raw trace dicts
        texts:  list[str]           — flattened text per trace
        structured: np.ndarray      — (N, D) structured feature matrix
        feature_names: list[str]    — column names for structured matrix
        labels: np.ndarray          — (N,) string labels
        llama_labels: np.ndarray    — (N,) llama labels for agreement
        trace_ids: list[str]        — trace IDs in same order
    """
    traces = load_labeled_traces()

    texts = [flatten_trace_to_text(t) for t in traces]
    labels = np.array([t["label"] for t in traces])
    llama_labels = np.array([t.get("llama_label", "") for t in traces])
    trace_ids = [t["trace_id"] for t in traces]

    # Structured features from the trace data itself
    struct_features = [extract_structured_features(t) for t in traces]

    # Optionally add SQLite trace_metrics
    if use_sqlite_features:
        sqlite_features = extract_structured_features_from_sqlite(trace_ids)
        for i, tid in enumerate(trace_ids):
            sf = sqlite_features.get(tid, {})
            struct_features[i].update(sf)

    # Convert to matrix
    feature_names = sorted(struct_features[0].keys())
    structured = np.array(
        [[f.get(k, 0.0) for k in feature_names] for f in struct_features],
        dtype=np.float64,
    )

    return {
        "traces": traces,
        "texts": texts,
        "structured": structured,
        "feature_names": feature_names,
        "labels": labels,
        "llama_labels": llama_labels,
        "trace_ids": trace_ids,
    }


def get_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split with fixed random seed.

    For tiny classes (ESCALATE), we use stratify with a fallback
    that groups ESCALATE with RETRY if stratification fails.
    """
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y,
        )
    except ValueError:
        # Stratification fails if a class has fewer samples than n_splits.
        # Fallback: group ESCALATE into RETRY for splitting purposes only.
        y_grouped = np.where(y == "ESCALATE", "RETRY", y)
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y_grouped,
        )


def get_index_splits(
    n: int,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) for consistent splitting across methods."""
    indices = np.arange(n)
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y,
        )
    except ValueError:
        y_grouped = np.where(y == "ESCALATE", "RETRY", y)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y_grouped,
        )
    return train_idx, test_idx
