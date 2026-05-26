"""Quick OpenRouter connectivity test — no DB or finalized data needed.

Tests that the API key works, the model responds, and token/cost tracking
records correctly. Uses a hardcoded GSM8K trace.

Usage:
    OPENROUTER_API_KEY=sk-or-... python -m squad_c.test_openrouter
"""
import json
import os
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

from ..cost_tracker import CostTracker, DOMAIN_MODELS
from ..recovery_actions import FailedTrace, run_retry, run_replan, run_escalate

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Hardcoded GSM8K trace — no DB needed
SAMPLE_TRACE = FailedTrace(
    trace_id="test_openrouter_gsm8k_1",
    problem_id="gsm8k_sample",
    domain="GSM8K",
    model_used=DOMAIN_MODELS["GSM8K"],
    problem_statement=(
        "Josh decides to try flipping a house. He buys a house for $80,000 "
        "and then puts in $50,000 in repairs. This increased the value of the "
        "house by 150%. How much profit did he make?"
    ),
    gold_answer="70000",
    final_answer="50000",
    steps=[
        {"step_index": 0, "step_type": "reasoning",
         "text": "The house value increased by 150% of $80,000 = $120,000. New value = $200,000. Cost = $130,000. Profit = $70,000.",
         "tool_name": None, "tool_args_json": None, "tool_output_json": None, "tool_call_result": None},
        {"step_index": 1, "step_type": "final_answer",
         "text": "50000",
         "tool_name": None, "tool_args_json": None, "tool_output_json": None, "tool_call_result": None},
    ],
    applicable_actions=["LOCAL_REPAIR", "RETRY", "REPLAN", "ESCALATE"],
    repaired_text=None,
    repair_step_id=None,
)


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY before running this test.")
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "TraceTriage OpenRouter Test",
        },
    )


def main() -> None:
    client = build_client()

    with tempfile.TemporaryDirectory() as tmp:
        tracker = CostTracker(Path(tmp) / "test_cost_log.jsonl")

        print(f"Model : {SAMPLE_TRACE.model_used}")
        print(f"Problem: {SAMPLE_TRACE.problem_statement[:80]}...")
        print(f"Gold answer: {SAMPLE_TRACE.gold_answer}")
        print(f"Wrong answer: {SAMPLE_TRACE.final_answer}")
        print()

        # Test 1: ESCALATE (no API call — sanity check)
        print("--- ESCALATE (no API call) ---")
        r = run_escalate(SAMPLE_TRACE, tracker)
        print(f"success={r.success}  cost=${r.cost_usd}  tokens={r.total_tokens}")
        assert not r.success and r.cost_usd == 0.0
        print("PASS\n")

        # Test 2: RETRY (real API call)
        print("--- RETRY (real API call) ---")
        r = run_retry(SAMPLE_TRACE, client, tracker)
        if r.error:
            print(f"ERROR: {r.error}")
            sys.exit(1)
        print(f"success={r.success}")
        print(f"tokens : in={r.input_tokens}  out={r.output_tokens}  total={r.total_tokens}")
        print(f"cost   : ${r.cost_usd:.6f}")
        print(f"latency: {r.latency_seconds:.2f}s")
        print(f"answer : {(r.recovered_answer or '')[:120]}")
        assert r.total_tokens > 0, "No tokens recorded — usage not returned by model"
        assert r.latency_seconds > 0
        print("PASS\n")

        # Test 3: REPLAN (real API call)
        print("--- REPLAN (real API call) ---")
        r = run_replan(SAMPLE_TRACE, client, tracker)
        if r.error:
            print(f"ERROR: {r.error}")
            sys.exit(1)
        print(f"success={r.success}")
        print(f"tokens : in={r.input_tokens}  out={r.output_tokens}  total={r.total_tokens}")
        print(f"cost   : ${r.cost_usd:.6f}")
        print(f"latency: {r.latency_seconds:.2f}s")
        print(f"answer : {(r.recovered_answer or '')[:120]}")
        print("PASS\n")

        # Session summary
        summary = tracker.session_summary()
        print("--- Session summary ---")
        print(f"Total calls   : {summary['total_calls']}")
        print(f"Total tokens  : {summary['total_tokens']}")
        print(f"Total cost    : ${summary['total_cost_usd']:.6f}")
        print(f"Total latency : {summary['total_latency_seconds']:.2f}s")
        print(f"By action     : {json.dumps(summary['by_action'], indent=2)}")
        print()
        print("All OpenRouter tests passed.")


if __name__ == "__main__":
    main()
