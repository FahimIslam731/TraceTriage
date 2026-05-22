from __future__ import annotations

"""Data loading and feature engineering for Squad B classification.

Loads the full failed-trace triage dataset from SQLite plus LLM label exports,
produces text representations and structured features for classifiers.

Reproducibility: all splits use RANDOM_SEED = 42.
"""
import json
import sqlite3
import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "causal_runs.sqlite"
GPT_LABELS_PATH = PROJECT_ROOT / "data" / "labeling_exports" / "gpt_auto_labels_P1.jsonl"
LLAMA_LABELS_PATH = PROJECT_ROOT / "data" / "labeling_exports" / "llama_auto_labels_P1.jsonl"
FAILED_TRACES_PATH = PROJECT_ROOT / "data" / "labeling_exports" / "failed_traces.jsonl"
SQUAD_A_DIR = PROJECT_ROOT / "squad_a"
FROZEN_SPLIT_PATHS = {
    "train": SQUAD_A_DIR / "train.csv",
    "dev": SQUAD_A_DIR / "dev.csv",
    "test": SQUAD_A_DIR / "test.csv",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TARGET_CLASSES = [
    "LOCAL_REPAIR",
    "RETRIEVE_MORE",
    "REPLAN",
    "TOOL_FIX",
    "RETRY",
    "ESCALATE",
]
INPUT_VARIANTS = [
    "full_trace",
    "final_answer_only",
    "verifier_feedback_only",
    "trace_stats_only",
    "causal_neighborhood",
]
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
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_frozen_split_labels() -> dict[str, dict[str, str]]:
    """Load Squad A frozen split labels keyed by trace_id.

    Returns {trace_id: {"split": split_name, "label": human_majority}}.
    """
    split_labels = {}
    for split, path in FROZEN_SPLIT_PATHS.items():
        if not path.exists():
            return {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                trace_id = row.get("trace_id")
                label = row.get("human_majority")
                if trace_id and label:
                    split_labels[trace_id] = {"split": split, "label": label}
    return split_labels


def _load_failed_traces_from_sqlite() -> list[dict]:
    """Load failing traces, triage metadata, and steps from SQLite."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    trace_rows = conn.execute(
        """SELECT t.trace_id, t.run_id, t.problem_id, t.benchmark, t.domain,
                  t.model_used, t.is_ablation, t.ablation_type, t.timestamp,
                  t.success, t.problem_statement, t.gold_answer, t.final_answer,
                  t.num_steps, t.causal_flow_analysis_minutes, t.is_passing_trace,
                  t.is_failing_trace, t.answer_exact_match,
                  tl.action_label, tl.label_source, tl.is_auto_labeled,
                  tl.needs_labeling, tl.is_local_repairable,
                  tl.num_successful_repair_steps, tl.applicable_actions_json,
                  tl.llm_1_action, tl.llm_1_rationale, tl.llm_2_action,
                  tl.llm_2_rationale, tl.human_action, tl.human_rationale,
                  tl.split
           FROM traces t
           JOIN triage_labels tl ON t.trace_id = tl.trace_id
           WHERE t.is_failing_trace = 1
           ORDER BY t.trace_id"""
    ).fetchall()

    step_rows = conn.execute(
        """SELECT step_uid, trace_id, run_id, problem_id, step_id, step_index,
                  step_type, dependencies_json, text, tool_name, tool_args_json,
                  tool_output_json, tool_call_result, state_snapshot_json,
                  trace_success, has_tool, is_reasoning_step, is_tool_call,
                  is_tool_response, is_final_answer, text_length
           FROM steps
           ORDER BY trace_id, step_index"""
    ).fetchall()

    consensus_rows = conn.execute(
        """SELECT trace_id, step_id, final_critic_summary, text,
                  consensus_score, is_repairable_step, has_successful_repair
           FROM consensus_steps
           ORDER BY trace_id, step_id"""
    ).fetchall()
    conn.close()

    steps_by_trace: dict[str, list[dict]] = {}
    for row in step_rows:
        steps_by_trace.setdefault(row["trace_id"], []).append(dict(row))

    consensus_by_trace: dict[str, list[dict]] = {}
    for row in consensus_rows:
        consensus_by_trace.setdefault(row["trace_id"], []).append(dict(row))

    traces = []
    for row in trace_rows:
        trace = dict(row)
        trace["steps"] = steps_by_trace.get(trace["trace_id"], [])
        trace["consensus_steps"] = consensus_by_trace.get(trace["trace_id"], [])
        traces.append(trace)
    return traces


def load_labeled_traces() -> list[dict]:
    """Load traces with 6-action labels joined to full trace data.

    Returns a list of dicts, each containing:
        - Full trace fields and steps from SQLite when available
        - 'label': LOCAL_REPAIR for CausalFlow-repairable traces, otherwise GPT label
        - 'llama_label': the Llama-assigned action (for agreement analysis)
        - 'gpt_rationale', 'gpt_confidence'
    """
    # Load labels keyed by trace_id
    gpt_labels = {r["trace_id"]: r for r in _load_jsonl(GPT_LABELS_PATH)}
    llama_labels = {r["trace_id"]: r for r in _load_jsonl(LLAMA_LABELS_PATH)}

    # Prefer SQLite because it contains both LOCAL_REPAIR and needs-labeling traces.
    # Fall back to the JSONL export if the database is unavailable.
    failed_traces = _load_failed_traces_from_sqlite() or _load_jsonl(FAILED_TRACES_PATH)

    merged = []
    for trace in failed_traces:
        tid = trace["trace_id"]
        gpt = gpt_labels.get(tid)
        is_local_repair = (
            trace.get("action_label") == "LOCAL_REPAIR"
            or int(trace.get("is_local_repairable") or 0) == 1
            or int(trace.get("num_successful_repair_steps") or 0) > 0
        )

        if is_local_repair:
            label = "LOCAL_REPAIR"
            gpt_rationale = "CausalFlow found at least one successful local counterfactual repair for this failed trace."
            gpt_confidence = 1.0
        elif gpt is not None:
            label = gpt["action"]
            gpt_rationale = gpt.get("rationale", "")
            gpt_confidence = gpt.get("confidence")
        else:
            continue  # skip non-LOCAL_REPAIR traces without a model/human label

        llama = llama_labels.get(tid, {})
        trace = dict(trace)
        trace["label"] = label
        trace["llama_label"] = llama.get("action")
        trace["gpt_rationale"] = gpt_rationale
        trace["gpt_confidence"] = gpt_confidence
        merged.append(trace)

    return merged


# ---------------------------------------------------------------------------
# Feature engineering — text
# ---------------------------------------------------------------------------

def _step_to_text(step: dict, max_text_chars: int = 600, max_output_chars: int = 400) -> list[str]:
    """Render one trace step as compact text."""
    lines = []
    header = f"[STEP {step.get('step_index', '?')}] type={step.get('step_type', '')}"
    if step.get("tool_name"):
        header += f" tool={step['tool_name']}"
    lines.append(header)

    text = step.get("text") or ""
    if text:
        lines.append(f"  {text[:max_text_chars]}")

    tool_out = step.get("tool_output_json") or ""
    if tool_out:
        lines.append(f"  output: {str(tool_out)[:max_output_chars]}")

    return lines


def _verifier_feedback_text(trace: dict) -> str:
    """Build a cheap verifier-feedback view from available outcome fields."""
    return "\n".join([
        f"[DOMAIN] {trace.get('domain', '')}",
        f"[PROBLEM_ID] {trace.get('problem_id', '')}",
        f"[GOLD_ANSWER] {trace.get('gold_answer', '')}",
        f"[AGENT_FINAL_ANSWER] {trace.get('final_answer', '')}",
        f"[ANSWER_EXACT_MATCH] {trace.get('answer_exact_match', '')}",
    ])


def _trace_stats_text(trace: dict) -> str:
    """Build a text representation containing only trace statistics."""
    steps = trace.get("steps", [])
    tool_calls = [s for s in steps if s.get("has_tool") or s.get("tool_name")]
    tool_failures = [s for s in steps if s.get("tool_call_result") == 0]
    step_types = [s.get("step_type", "") for s in steps]
    tools_used = {s.get("tool_name") for s in steps if s.get("tool_name")}
    features = {
        "num_steps": float(len(steps)),
        "num_tool_calls": float(len(tool_calls)),
        "num_tool_failures": float(len(tool_failures)),
        "tool_failure_rate": len(tool_failures) / len(tool_calls) if tool_calls else 0.0,
        "problem_length": float(len(trace.get("problem_statement", "") or "")),
        "final_answer_length": float(len(trace.get("final_answer", "") or "")),
    }
    for stype in ["reasoning", "tool_call", "tool_response", "final_answer", "llm_response"]:
        features[f"steptype_{stype}"] = float(step_types.count(stype))
    for tool in TOOL_NAMES:
        features[f"tool_{tool}"] = 1.0 if tool in tools_used else 0.0

    lines = [
        f"[DOMAIN] {trace.get('domain', '')}",
        f"[NUM_STEPS] {trace.get('num_steps', len(trace.get('steps', [])))}",
    ]
    for key in sorted(features):
        lines.append(f"[{key.upper()}] {features[key]}")
    return "\n".join(lines)


def _causal_neighborhood_text(trace: dict, window: int = 2) -> str:
    """Render two steps before/after CausalFlow's suspicious consensus steps."""
    steps = trace.get("steps", [])
    consensus_steps = trace.get("consensus_steps", [])
    suspicious_ids = {
        int(row["step_id"]) for row in consensus_steps
        if row.get("step_id") is not None
    }

    parts = [
        f"[DOMAIN] {trace.get('domain', '')}",
        f"[PROBLEM] {trace.get('problem_statement', '')}",
        f"[FINAL_ANSWER] {trace.get('final_answer', '')}",
    ]

    if not suspicious_ids:
        parts.append("[CAUSAL_NEIGHBORHOOD] unavailable")
        return "\n".join(parts)

    step_ids_to_keep = set()
    for step_id in suspicious_ids:
        step_ids_to_keep.update(range(step_id - window, step_id + window + 1))

    consensus_by_step = {row.get("step_id"): row for row in consensus_steps}
    for step in steps:
        step_id = step.get("step_id")
        if step_id not in step_ids_to_keep:
            continue
        if step_id in suspicious_ids:
            consensus = consensus_by_step.get(step_id, {})
            parts.append(
                "[SUSPICIOUS_STEP] "
                f"score={consensus.get('consensus_score', '')} "
                f"repairable={consensus.get('is_repairable_step', '')} "
                f"successful_repair={consensus.get('has_successful_repair', '')}"
            )
            summary = consensus.get("final_critic_summary") or ""
            if summary:
                parts.append(f"[CRITIC_SUMMARY] {summary[:800]}")
        parts.extend(_step_to_text(step))

    return "\n".join(parts)


def flatten_trace_to_text(trace: dict, input_variant: str = "full_trace") -> str:
    """Convert a trace dict into a single text string for TF-IDF / embeddings.

    Format:
        [DOMAIN] <domain>
        [PROBLEM] <problem_statement>
        [STEP 0] type=<step_type> tool=<tool_name>
          <step text (truncated)>
        ...
        [FINAL_ANSWER] <final_answer>
    """
    if input_variant == "final_answer_only":
        return f"[FINAL_ANSWER] {trace.get('final_answer', '')}"
    if input_variant == "verifier_feedback_only":
        return _verifier_feedback_text(trace)
    if input_variant == "trace_stats_only":
        return _trace_stats_text(trace)
    if input_variant == "causal_neighborhood":
        return _causal_neighborhood_text(trace)
    if input_variant != "full_trace":
        raise ValueError(f"Unknown input_variant={input_variant!r}; expected one of {INPUT_VARIANTS}")

    parts = [f"[DOMAIN] {trace.get('domain', '')}"]
    parts.append(f"[PROBLEM] {trace.get('problem_statement', '')}")

    for step in trace.get("steps", []):
        parts.extend(_step_to_text(step))

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
    use_frozen_splits: bool = True,
    input_variant: str = "full_trace",
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
        split_indices: dict[str, np.ndarray] — frozen Squad A split indices, if available
    """
    traces = load_labeled_traces()
    split_labels = load_frozen_split_labels() if use_frozen_splits else {}

    if split_labels:
        filtered_traces = []
        for trace in traces:
            split_label = split_labels.get(trace["trace_id"])
            if split_label is None:
                continue
            trace = dict(trace)
            trace["label"] = split_label["label"]
            trace["split"] = split_label["split"]
            trace["label_source"] = "squad_a_human_majority"
            filtered_traces.append(trace)
        traces = filtered_traces

    texts = [flatten_trace_to_text(t, input_variant=input_variant) for t in traces]
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

    split_indices: dict[str, np.ndarray] = {}
    if split_labels:
        for split in FROZEN_SPLIT_PATHS:
            split_indices[split] = np.array(
                [i for i, trace in enumerate(traces) if trace.get("split") == split],
                dtype=int,
            )

    return {
        "traces": traces,
        "texts": texts,
        "structured": structured,
        "feature_names": feature_names,
        "labels": labels,
        "llama_labels": llama_labels,
        "trace_ids": trace_ids,
        "split_indices": split_indices,
        "input_variant": input_variant,
    }


def get_frozen_train_dev_test_indices(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return frozen (train, dev, test) indices if Squad A splits are loaded."""
    split_indices = dataset.get("split_indices") or {}
    if all(split in split_indices and len(split_indices[split]) > 0 for split in ("train", "dev", "test")):
        return split_indices["train"], split_indices["dev"], split_indices["test"]
    return None


def get_train_test_indices(
    dataset: dict[str, Any],
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    include_dev_in_train: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return train/test indices, preferring frozen Squad A splits when present."""
    frozen = get_frozen_train_dev_test_indices(dataset)
    if frozen is not None:
        train_idx, dev_idx, test_idx = frozen
        if include_dev_in_train:
            train_idx = np.concatenate([train_idx, dev_idx])
        return train_idx, test_idx, dev_idx

    train_idx, test_idx = get_index_splits(len(labels), labels, test_size, seed)
    return train_idx, test_idx, None


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
