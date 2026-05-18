"""Policy analysis for Experiment 4 — Stage A decision gate and full comparison.

Run after recover_results.jsonl is populated by run_recovery.py:

    # Stage A: check pilot results and apply decision gate
    python -m squad_c.analyze_policies --stage a

    # Stage B: full policy comparison (only run if Stage A passed)
    python -m squad_c.analyze_policies --stage b

Outputs
-------
squad_c/results/policy_comparison.json   — per-policy metrics table
squad_c/results/decision_gate.json       — Stage A gate result (pass/fail + margin)

7 Policies (from paper Experiment 4)
-------------------------------------
always_retry          — apply RETRY to every failure
always_local_repair   — apply LOCAL_REPAIR to every failure
always_replan         — apply REPLAN to every failure
always_retrieve_more  — RETRIEVE_MORE where applicable, LOCAL_REPAIR fallback
domain_policy         — modal action per domain (computed from Squad A labels)
trace_triage          — action from Squad A human_majority labels (all_1212_labels.csv)
oracle                — whichever action actually succeeded (upper bound)
"""
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path("data/causal_runs.sqlite")
LABELS_CSV = Path("squad_c/all_1212_labels.csv")   # Squad A's human majority labels
RESULTS_DIR = Path("squad_c/results")
RESULTS_JSONL = RESULTS_DIR / "recovery_results.jsonl"
POLICY_JSON = RESULTS_DIR / "policy_comparison.json"
GATE_JSON = RESULTS_DIR / "decision_gate.json"

# Stage A decision gate threshold (paper §5, Experiment 4 Stage B condition)
GATE_THRESHOLD_POINTS = 5.0   # Trace Triage must beat Domain Policy by this many ppt
GATE_METRIC = "utility_lambda_1_0"  # which utility column to use for the gate (matches key format)

# Cost-adjusted utility lambdas to sweep (paper: "vary lambda")
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0]

# Fallback domain modal actions when triage_labels are incomplete
# (updated once Squad A finalizes labels)
DOMAIN_MODAL_FALLBACK: dict[str, str] = {
    "GSM8K":        "LOCAL_REPAIR",
    "MBPP":         "LOCAL_REPAIR",
    "SealQA":       "RETRIEVE_MORE",
    "MedBrowseComp":"RETRIEVE_MORE",
    "BrowseComp":   "RETRIEVE_MORE",
}

RETRIEVE_MORE_DOMAINS = {"SealQA", "MedBrowseComp", "BrowseComp"}


# ---------------------------------------------------------------------------
# Load raw results
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict[str, dict[str, dict]]:
    """Load recovery_results.jsonl into {trace_id: {action: result_dict}}."""
    if not path.exists():
        sys.exit(f"Results file not found: {path}\nRun run_recovery.py first.")
    by_trace: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_trace[rec["trace_id"]][rec["action"]] = rec
    return dict(by_trace)


def load_squad_a_labels(csv_path: Path) -> dict[str, str]:
    """Load human majority-vote labels from Squad A's all_1212_labels.csv."""
    if not csv_path.exists():
        sys.exit(f"Squad A labels not found: {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        return {row["trace_id"]: row["human_majority"] for row in csv.DictReader(f)}


def load_trace_metadata(db_path: Path) -> dict[str, dict]:
    """Load domain and action_label for each trace.

    action_label comes from Squad A's all_1212_labels.csv (human majority-vote).
    For traces not in the CSV the label is None (excluded from trace_triage evaluation).
    """
    squad_a_labels = load_squad_a_labels(LABELS_CSV)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT l.trace_id, t.domain
        FROM triage_labels l
        JOIN traces t ON l.trace_id = t.trace_id
        WHERE t.is_failing_trace = 1 AND t.is_ablation = 0
    """).fetchall()
    conn.close()

    return {
        row["trace_id"]: {
            "domain": row["domain"],
            "action_label": squad_a_labels.get(row["trace_id"]),
        }
        for row in rows
    }


# ---------------------------------------------------------------------------
# Domain modal action (for Domain Policy)
# ---------------------------------------------------------------------------

def compute_domain_modal(metadata: dict[str, dict]) -> dict[str, str]:
    """Compute modal triage label per domain from Squad A labels."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for info in metadata.values():
        if info["action_label"]:
            counts[info["domain"]][info["action_label"]] += 1

    modal: dict[str, str] = {}
    for domain, counter in counts.items():
        modal[domain] = counter.most_common(1)[0][0] if counter else DOMAIN_MODAL_FALLBACK.get(domain, "ESCALATE")

    # Fill in any domains with no labeled data
    for domain, fallback in DOMAIN_MODAL_FALLBACK.items():
        if domain not in modal:
            modal[domain] = fallback
    return modal


# ---------------------------------------------------------------------------
# Policy selectors — each returns (action, result_dict | None) for one trace
# ---------------------------------------------------------------------------

def _pick(trace_results: dict[str, dict], action: str) -> dict | None:
    return trace_results.get(action)


def policy_always_retry(trace_id: str, domain: str, trace_results: dict) -> dict | None:
    return _pick(trace_results, "RETRY")


def policy_always_local_repair(trace_id: str, domain: str, trace_results: dict) -> dict | None:
    return _pick(trace_results, "LOCAL_REPAIR")


def policy_always_replan(trace_id: str, domain: str, trace_results: dict) -> dict | None:
    return _pick(trace_results, "REPLAN")


def policy_always_retrieve_more(trace_id: str, domain: str, trace_results: dict) -> dict | None:
    # RETRIEVE_MORE where applicable, LOCAL_REPAIR fallback
    if domain in RETRIEVE_MORE_DOMAINS and "RETRIEVE_MORE" in trace_results:
        return _pick(trace_results, "RETRIEVE_MORE")
    return _pick(trace_results, "LOCAL_REPAIR")


def policy_domain(trace_id: str, domain: str, trace_results: dict, modal: dict[str, str]) -> dict | None:
    action = modal.get(domain, "ESCALATE")
    result = _pick(trace_results, action)
    # If modal action wasn't run for this trace, try LOCAL_REPAIR as fallback
    return result or _pick(trace_results, "LOCAL_REPAIR")


def policy_trace_triage(trace_id: str, domain: str, trace_results: dict, action_label: str | None) -> dict | None:
    if not action_label:
        return None  # trace not yet labeled — excluded from Trace Triage evaluation
    return _pick(trace_results, action_label)


def policy_oracle(trace_id: str, domain: str, trace_results: dict) -> dict | None:
    """Pick whichever action succeeded. If multiple succeeded, pick lowest cost. If none, pick ESCALATE."""
    successes = [r for r in trace_results.values() if r.get("success")]
    if successes:
        return min(successes, key=lambda r: r.get("cost_usd", 0.0))
    return _pick(trace_results, "ESCALATE") or next(iter(trace_results.values()), None)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_utility(success_rate: float, avg_cost: float, lam: float) -> float:
    """Cost-adjusted utility: success_rate - lambda * avg_cost_per_trace."""
    return round(success_rate - lam * avg_cost, 6)


def evaluate_policy(
    policy_results: list[dict | None],
    policy_name: str,
) -> dict:
    """Compute all metrics for one policy across all traces."""
    valid = [r for r in policy_results if r is not None]
    n_total = len(policy_results)
    n_valid = len(valid)

    if n_valid == 0:
        return {"policy": policy_name, "n_traces": 0, "coverage": 0.0,
                "note": "no labeled traces for this policy"}

    n_success = sum(1 for r in valid if r.get("success"))
    total_cost = sum(r.get("cost_usd", 0.0) for r in valid)
    total_tokens = sum(r.get("total_tokens", 0) for r in valid)
    total_latency = sum(r.get("latency_seconds", 0.0) for r in valid)

    success_rate = n_success / n_valid
    avg_cost = total_cost / n_valid

    metrics = {
        "policy": policy_name,
        "n_traces": n_valid,
        "coverage": round(n_valid / n_total, 4) if n_total else 0,
        "n_success": n_success,
        "recovery_rate": round(success_rate, 4),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_trace": round(avg_cost, 8),
        "total_tokens": total_tokens,
        "avg_latency_seconds": round(total_latency / n_valid, 3),
        "cost_per_success": round(total_cost / n_success, 6) if n_success else None,
    }

    # Cost-adjusted utility at each lambda
    for lam in LAMBDAS:
        key = f"utility_lambda_{lam}".replace(".", "_")
        metrics[key] = compute_utility(success_rate, avg_cost, lam)

    return metrics


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------

def check_decision_gate(policy_metrics: list[dict]) -> dict:
    """Compare Trace Triage vs Domain Policy on GATE_METRIC.

    Returns a gate result dict with pass/fail and the margin.
    Paper condition: Trace Triage must beat Domain Policy by >= 5 ppt.
    """
    by_name = {m["policy"]: m for m in policy_metrics}
    triage = by_name.get("trace_triage")
    domain = by_name.get("domain_policy")

    if not triage or not domain:
        return {"passed": False, "reason": "trace_triage or domain_policy metrics missing"}

    triage_score = triage.get(GATE_METRIC, 0.0)
    domain_score = domain.get(GATE_METRIC, 0.0)
    margin = round((triage_score - domain_score) * 100, 2)  # in percentage points

    passed = margin >= GATE_THRESHOLD_POINTS
    return {
        "passed": passed,
        "metric": GATE_METRIC,
        "trace_triage_score": triage_score,
        "domain_policy_score": domain_score,
        "margin_ppt": margin,
        "threshold_ppt": GATE_THRESHOLD_POINTS,
        "recommendation": "Proceed to Stage B" if passed else "Escalate to PI — margin below threshold",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(stage: str) -> None:
    raw = load_results(RESULTS_JSONL)
    metadata = load_trace_metadata(DB_PATH)
    modal = compute_domain_modal(metadata)

    print(f"Loaded {len(raw)} traces from results")
    print(f"Domain modal actions: {modal}")

    # Collect per-policy picks for every trace
    policy_picks: dict[str, list[dict | None]] = {
        "always_retry": [],
        "always_local_repair": [],
        "always_replan": [],
        "always_retrieve_more": [],
        "domain_policy": [],
        "trace_triage": [],
        "oracle": [],
    }

    for trace_id, trace_results in raw.items():
        info = metadata.get(trace_id, {})
        domain = info.get("domain", "")
        action_label = info.get("action_label")

        policy_picks["always_retry"].append(policy_always_retry(trace_id, domain, trace_results))
        policy_picks["always_local_repair"].append(policy_always_local_repair(trace_id, domain, trace_results))
        policy_picks["always_replan"].append(policy_always_replan(trace_id, domain, trace_results))
        policy_picks["always_retrieve_more"].append(policy_always_retrieve_more(trace_id, domain, trace_results))
        policy_picks["domain_policy"].append(policy_domain(trace_id, domain, trace_results, modal))
        policy_picks["trace_triage"].append(policy_trace_triage(trace_id, domain, trace_results, action_label))
        policy_picks["oracle"].append(policy_oracle(trace_id, domain, trace_results))

    policy_metrics = [
        evaluate_policy(picks, name)
        for name, picks in policy_picks.items()
    ]

    # Print table
    print(f"\n{'Policy':<24} {'N':>5} {'RecovRate':>10} {'AvgCost':>10} {'Util(λ=1)':>10} {'Util(λ=2)':>10}")
    print("-" * 75)
    for m in policy_metrics:
        print(
            f"{m['policy']:<24} {m.get('n_traces', 0):>5} "
            f"{m.get('recovery_rate', 0):>10.4f} "
            f"{m.get('avg_cost_per_trace', 0):>10.6f} "
            f"{m.get('utility_lambda_1_0', 0):>10.4f} "
            f"{m.get('utility_lambda_2_0', 0):>10.4f}"
        )

    POLICY_JSON.write_text(json.dumps(policy_metrics, indent=2), encoding="utf-8")
    print(f"\nPolicy comparison written to {POLICY_JSON}")

    # Stage A: apply decision gate
    if stage == "a":
        gate = check_decision_gate(policy_metrics)
        GATE_JSON.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(f"\n=== Stage A Decision Gate ===")
        print(json.dumps(gate, indent=2))
        if gate["passed"]:
            print("\n✓ GATE PASSED — proceed to Stage B (run_recovery.py --full)")
        else:
            print("\n✗ GATE FAILED — escalate to PI before running Stage B")

    elif stage == "b":
        print("\nStage B: full simulation results analyzed.")
        print("Check policy_comparison.json for the complete table.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 4 policy analysis")
    parser.add_argument("--stage", choices=["a", "b"], required=True,
                        help="'a' = pilot + decision gate, 'b' = full comparison")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_analysis(args.stage)
