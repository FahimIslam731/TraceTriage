"""Recovery simulation runner — Squad C, Experiment 4.

Usage (pilot, 100 traces per domain):
    python -m squad_c.run_recovery --pilot

Usage (full run, all 1299 traces):
    python -m squad_c.run_recovery --full

Usage (single action on a specific trace for testing):
    python -m squad_c.run_recovery --test-action RETRY --trace-id <trace_id>

Outputs
-------
squad_c/results/recovery_results.jsonl   — one RecoveryResult per action per trace
squad_c/results/cost_log.jsonl           — one CallRecord per model call (audit trail)
squad_c/results/summary.json            — aggregate stats after run completes

Environment
-----------
OPENROUTER_API_KEY   required for any action that calls a model
"""
import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads OPENROUTER_API_KEY and SERPER_API_KEY from .env

from .cost_tracker import CostTracker
from .recovery_actions import FailedTrace, RecoveryResult, run_recovery

DB_PATH = Path("data/causal_runs.sqlite")
RESULTS_DIR = Path("squad_c/results")
RESULTS_JSONL = RESULTS_DIR / "recovery_results.jsonl"
COST_LOG = RESULTS_DIR / "cost_log.jsonl"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Pilot: 100 traces per domain (use all SealQA if < 100 available)
PILOT_PER_DOMAIN = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_failed_traces(
    db_path: Path,
    domain_filter: list[str] | None = None,
    limit_per_domain: int | None = None,
) -> list[FailedTrace]:
    """Load failed non-ablation traces with repair data from SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            t.trace_id, t.problem_id, t.domain, t.model_used,
            t.problem_statement, t.gold_answer, t.final_answer,
            l.applicable_actions_json, l.is_local_repairable,
            -- best successful repair (if any) for LOCAL_REPAIR
            ra.repaired_text, ra.step_id AS repair_step_id
        FROM traces t
        JOIN triage_labels l ON t.trace_id = l.trace_id
        LEFT JOIN (
            SELECT trace_id, repaired_text, step_id
            FROM repair_attempts
            WHERE repair_succeeded = 1
            GROUP BY trace_id           -- one representative repair per trace
        ) ra ON ra.trace_id = t.trace_id
        WHERE t.is_failing_trace = 1
          AND t.is_ablation = 0
        ORDER BY t.domain, t.trace_id
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    traces_by_domain: dict[str, list] = {}
    for row in rows:
        d = row["domain"]
        traces_by_domain.setdefault(d, []).append(row)

    result = []
    for domain, domain_rows in traces_by_domain.items():
        if domain_filter and domain not in domain_filter:
            continue
        subset = domain_rows[:limit_per_domain] if limit_per_domain else domain_rows
        for row in subset:
            applicable = json.loads(row["applicable_actions_json"] or "[]")
            result.append(FailedTrace(
                trace_id=row["trace_id"],
                problem_id=row["problem_id"] or "",
                domain=domain,
                model_used=row["model_used"] or "",
                problem_statement=row["problem_statement"] or "",
                gold_answer=row["gold_answer"] or "",
                final_answer=row["final_answer"] or "",
                steps=_load_steps(row["trace_id"]),
                applicable_actions=applicable,
                repaired_text=row["repaired_text"],
                repair_step_id=row["repair_step_id"],
            ))
    return result


def _load_steps(trace_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT step_index, step_type, text, tool_name, tool_args_json, "
        "tool_output_json, tool_call_result "
        "FROM steps WHERE trace_id = ? ORDER BY step_index",
        (trace_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def load_completed(results_path: Path) -> set[tuple[str, str]]:
    """Return set of (trace_id, action) pairs already in the results file."""
    done = set()
    if not results_path.exists():
        return done
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done.add((rec["trace_id"], rec["action"]))
    return done


def append_result(path: Path, result: RecoveryResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    traces: list[FailedTrace],
    client: OpenAI,
    tracker: CostTracker,
    actions_to_run: list[str] | None = None,
) -> list[RecoveryResult]:
    """Run all applicable recovery actions on each trace. Skips already-completed pairs."""
    completed = load_completed(RESULTS_JSONL)
    results = []
    total = sum(len(t.applicable_actions) for t in traces)
    done_count = 0

    for trace in traces:
        for action in trace.applicable_actions:
            if actions_to_run and action not in actions_to_run:
                continue
            if (trace.trace_id, action) in completed:
                done_count += 1
                continue

            try:
                result = run_recovery(trace, action, tracker, client)
            except Exception as exc:
                # Record failure without crashing the whole run
                result = RecoveryResult(
                    trace_id=trace.trace_id, action=action, success=False,
                    recovered_answer=None, input_tokens=0, output_tokens=0,
                    total_tokens=0, cost_usd=0.0, latency_seconds=0.0,
                    model_used="", error=repr(exc),
                )

            append_result(RESULTS_JSONL, result)
            results.append(result)
            done_count += 1

            status = "OK" if result.success else ("ERR" if result.error else "FAIL")
            print(
                f"[{done_count}/{total}] {trace.domain:<14} {action:<15} "
                f"{status}  cost=${result.cost_usd:.5f}  lat={result.latency_seconds:.1f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(results: list[RecoveryResult], tracker: CostTracker) -> dict:
    from collections import defaultdict

    by_action: dict[str, dict] = defaultdict(lambda: {"calls": 0, "successes": 0, "cost": 0.0, "tokens": 0})
    by_domain: dict[str, dict] = defaultdict(lambda: {"calls": 0, "successes": 0, "cost": 0.0})

    for r in results:
        ba = by_action[r.action]
        ba["calls"] += 1
        ba["successes"] += int(r.success)
        ba["cost"] += r.cost_usd
        ba["tokens"] += r.total_tokens

        bd = by_domain[r.action + "::" + (r.metadata.get("domain", ""))]
        bd["calls"] += 1
        bd["successes"] += int(r.success)
        bd["cost"] += r.cost_usd

    action_summary = {
        action: {
            "calls": d["calls"],
            "recovery_rate": round(d["successes"] / d["calls"], 4) if d["calls"] else 0,
            "total_cost_usd": round(d["cost"], 6),
            "total_tokens": d["tokens"],
            "cost_per_success": (
                round(d["cost"] / d["successes"], 6) if d["successes"] else None
            ),
        }
        for action, d in by_action.items()
    }

    return {
        "total_traces_attempted": len({r.trace_id for r in results}),
        "total_action_calls": len(results),
        "overall_recovery_rate": round(
            sum(r.success for r in results) / len(results), 4
        ) if results else 0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "by_action": action_summary,
        "session": tracker.session_summary(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENROUTER_API_KEY environment variable")
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "TraceTriage Recovery Simulation",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceTriage recovery simulation runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help=f"Run pilot: {PILOT_PER_DOMAIN} traces per domain")
    mode.add_argument("--full", action="store_true", help="Run full simulation on all traces")
    mode.add_argument("--test-action", metavar="ACTION", help="Test a single action")
    parser.add_argument("--trace-id", help="Specific trace_id for --test-action")
    parser.add_argument("--domain", nargs="+", help="Restrict to specific domain(s)")
    parser.add_argument("--dry-run", action="store_true", help="Load data only, no model calls")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(COST_LOG)
    client = build_client() if not args.dry_run else None

    if args.test_action:
        traces = load_failed_traces(DB_PATH, domain_filter=args.domain)
        if args.trace_id:
            traces = [t for t in traces if t.trace_id == args.trace_id]
        if not traces:
            sys.exit("No matching traces found")
        trace = traces[0]
        print(f"Testing {args.test_action} on {trace.trace_id} ({trace.domain})")
        result = run_recovery(trace, args.test_action, tracker, client)
        print(json.dumps(asdict(result), indent=2))
        return

    limit = PILOT_PER_DOMAIN if args.pilot else None
    traces = load_failed_traces(DB_PATH, domain_filter=args.domain, limit_per_domain=limit)
    print(f"Loaded {len(traces)} traces {'(pilot)' if args.pilot else '(full)'}")

    if args.dry_run:
        for d in sorted({t.domain for t in traces}):
            count = sum(1 for t in traces if t.domain == d)
            print(f"  {d}: {count} traces")
        return

    results = run_simulation(traces, client, tracker)

    summary = compute_summary(results, tracker)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
