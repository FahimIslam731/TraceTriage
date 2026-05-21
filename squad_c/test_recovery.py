"""Tests for Squad C recovery action functions.

Run all tests (no API key needed):
    python -m squad_c.test_recovery

Run live integration test (needs OPENROUTER_API_KEY + real DB):
    python -m squad_c.test_recovery --live
"""
import argparse
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .cost_tracker import CostTracker, DOMAIN_MODELS
from .recovery_actions import (
    FailedTrace,
    run_escalate,
    run_local_repair,
    run_replan,
    run_retrieve_more,
    run_retry,
    run_tool_fix,
    run_recovery,
)
from .verify import verify_answer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_trace(**overrides) -> FailedTrace:
    defaults = dict(
        trace_id="trace_test_001",
        problem_id="gsm8k_1",
        domain="GSM8K",
        model_used="google/gemini-2.0-flash-lite-001",
        problem_statement="Josh buys a house for $80,000 and spends $50,000 on repairs. The value rises 150%. What is his profit?",
        gold_answer="70000",
        final_answer="50000",
        steps=[
            {"step_index": 0, "step_type": "reasoning", "text": "Let me calculate...", "tool_name": None, "tool_args_json": None, "tool_output_json": None, "tool_call_result": None},
            {"step_index": 1, "step_type": "final_answer", "text": "50000", "tool_name": None, "tool_args_json": None, "tool_output_json": None, "tool_call_result": None},
        ],
        applicable_actions=["LOCAL_REPAIR", "RETRY", "REPLAN", "ESCALATE"],
        repaired_text="The house was worth $80,000 * 2.5 = $200,000. Profit = $200,000 - $130,000 = $70,000.",
        repair_step_id=1,
    )
    defaults.update(overrides)
    return FailedTrace(**defaults)


def _make_tool_trace(**overrides) -> FailedTrace:
    """Trace with a failed tool call, for TOOL_FIX / RETRIEVE_MORE tests."""
    defaults = dict(
        trace_id="trace_tool_001",
        problem_id="mbpp_7",
        domain="MBPP",
        model_used="",
        problem_statement="Write a Python function that returns the nth Fibonacci number.",
        gold_answer="def fib(n): ...",
        final_answer="def fib(n): return n",
        steps=[
            {"step_index": 0, "step_type": "tool_call", "text": "Running code", "tool_name": "code_execution", "tool_args_json": '{"code": "def fib(n): return n"}', "tool_output_json": '{"error": "AssertionError: expected 55 got 10"}', "tool_call_result": 0},
            {"step_index": 1, "step_type": "final_answer", "text": "def fib(n): return n", "tool_name": None, "tool_args_json": None, "tool_output_json": None, "tool_call_result": None},
        ],
        applicable_actions=["LOCAL_REPAIR", "RETRY", "REPLAN", "TOOL_FIX", "ESCALATE"],
        repaired_text=None,
        repair_step_id=None,
    )
    defaults.update(overrides)
    return FailedTrace(**defaults)


def _make_tracker(tmp_path: Path) -> CostTracker:
    return CostTracker(tmp_path / "cost_log.jsonl")


def _mock_openai_client(answer: str = "The answer is 70000.", input_tokens: int = 150, output_tokens: int = 50) -> MagicMock:
    """Return a mock OpenAI client whose chat.completions.create returns a fixed answer."""
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    message = SimpleNamespace(content=answer)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice], usage=usage)

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Tests: ESCALATE
# ---------------------------------------------------------------------------

class TestEscalate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))
        self.trace = _make_trace()

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_is_false(self):
        r = run_escalate(self.trace, self.tracker)
        self.assertFalse(r.success)

    def test_zero_cost(self):
        r = run_escalate(self.trace, self.tracker)
        self.assertEqual(r.cost_usd, 0.0)
        self.assertEqual(r.total_tokens, 0)
        self.assertEqual(r.latency_seconds, 0.0)

    def test_no_recovered_answer(self):
        r = run_escalate(self.trace, self.tracker)
        self.assertIsNone(r.recovered_answer)

    def test_action_label(self):
        r = run_escalate(self.trace, self.tracker)
        self.assertEqual(r.action, "ESCALATE")

    def test_records_to_cost_log(self):
        run_escalate(self.trace, self.tracker)
        summary = self.tracker.session_summary()
        self.assertEqual(summary["total_calls"], 1)
        self.assertEqual(summary["total_cost_usd"], 0.0)


# ---------------------------------------------------------------------------
# Tests: LOCAL_REPAIR
# ---------------------------------------------------------------------------

class TestLocalRepair(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_when_repaired_text_present(self):
        trace = _make_trace(repaired_text="The profit is $70,000.")
        r = run_local_repair(trace, self.tracker)
        self.assertTrue(r.success)
        self.assertIsNotNone(r.recovered_answer)

    def test_failure_when_no_repaired_text(self):
        trace = _make_trace(repaired_text=None)
        r = run_local_repair(trace, self.tracker)
        self.assertFalse(r.success)
        self.assertIsNone(r.recovered_answer)

    def test_zero_cost(self):
        trace = _make_trace()
        r = run_local_repair(trace, self.tracker)
        self.assertEqual(r.cost_usd, 0.0)
        self.assertEqual(r.total_tokens, 0)

    def test_action_label(self):
        trace = _make_trace()
        r = run_local_repair(trace, self.tracker)
        self.assertEqual(r.action, "LOCAL_REPAIR")

    def test_repair_step_id_in_metadata(self):
        trace = _make_trace(repair_step_id=42)
        r = run_local_repair(trace, self.tracker)
        self.assertEqual(r.metadata.get("repair_step_id"), 42)


# ---------------------------------------------------------------------------
# Tests: RETRY
# ---------------------------------------------------------------------------

class TestRetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_when_model_returns_correct_answer(self):
        trace = _make_trace()
        client = _mock_openai_client(answer="The profit is $70,000.")
        r = run_retry(trace, client, self.tracker)
        self.assertTrue(r.success)
        self.assertEqual(r.action, "RETRY")

    def test_failure_when_model_returns_wrong_answer(self):
        trace = _make_trace()
        client = _mock_openai_client(answer="The profit is $50,000.")
        r = run_retry(trace, client, self.tracker)
        self.assertFalse(r.success)

    def test_records_tokens_and_cost(self):
        trace = _make_trace()
        client = _mock_openai_client(input_tokens=200, output_tokens=80)
        r = run_retry(trace, client, self.tracker)
        self.assertEqual(r.input_tokens, 200)
        self.assertEqual(r.output_tokens, 80)
        self.assertEqual(r.total_tokens, 280)
        self.assertGreater(r.cost_usd, 0.0)

    def test_uses_temperature_1(self):
        trace = _make_trace()
        client = _mock_openai_client()
        run_retry(trace, client, self.tracker)
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 1.0)

    def test_error_captured_on_api_failure(self):
        trace = _make_trace()
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("connection error")
        r = run_retry(trace, client, self.tracker)
        self.assertFalse(r.success)
        self.assertIn("connection error", r.error)
        self.assertIsNone(r.recovered_answer)

    def test_latency_recorded(self):
        trace = _make_trace()
        client = _mock_openai_client()
        r = run_retry(trace, client, self.tracker)
        self.assertGreaterEqual(r.latency_seconds, 0.0)


# ---------------------------------------------------------------------------
# Tests: REPLAN
# ---------------------------------------------------------------------------

class TestReplan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_on_correct_answer(self):
        trace = _make_trace()
        client = _mock_openai_client(answer="Using a different method: profit = 200000 - 130000 = 70000")
        r = run_replan(trace, client, self.tracker)
        self.assertTrue(r.success)

    def test_prompt_includes_previous_wrong_answer(self):
        trace = _make_trace()
        client = _mock_openai_client()
        run_replan(trace, client, self.tracker)
        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn(trace.final_answer, user_msg)

    def test_prompt_instructs_different_approach(self):
        trace = _make_trace()
        client = _mock_openai_client()
        run_replan(trace, client, self.tracker)
        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn("different", user_msg.lower())

    def test_uses_temperature_07(self):
        trace = _make_trace()
        client = _mock_openai_client()
        run_replan(trace, client, self.tracker)
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)


# ---------------------------------------------------------------------------
# Tests: RETRIEVE_MORE
# ---------------------------------------------------------------------------

class TestRetrieveMore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_not_applicable_for_gsm8k(self):
        trace = _make_trace(domain="GSM8K")
        client = _mock_openai_client()
        r = run_retrieve_more(trace, client, self.tracker)
        self.assertFalse(r.success)
        self.assertIsNotNone(r.error)
        self.assertIn("not applicable", r.error)
        client.chat.completions.create.assert_not_called()

    def test_not_applicable_for_mbpp(self):
        trace = _make_trace(domain="MBPP")
        client = _mock_openai_client()
        r = run_retrieve_more(trace, client, self.tracker)
        client.chat.completions.create.assert_not_called()

    def test_applicable_for_sealqa(self):
        trace = _make_trace(
            domain="SealQA",
            model_used="google/gemini-3-flash-preview",
            gold_answer="Paris",
            final_answer="London",
            applicable_actions=["RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX", "ESCALATE"],
        )
        client = _mock_openai_client(answer="After extensive research, the answer is Paris.")
        r = run_retrieve_more(trace, client, self.tracker)
        self.assertTrue(r.success)
        self.assertEqual(r.action, "RETRIEVE_MORE")

    def test_applicable_for_medbrowsecomp(self):
        trace = _make_trace(
            domain="MedBrowseComp",
            model_used="google/gemini-3-flash-preview",
            gold_answer="aspirin",
            final_answer="ibuprofen",
            applicable_actions=["RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX", "ESCALATE"],
        )
        client = _mock_openai_client(answer="Based on more thorough research: aspirin is the answer.")
        r = run_retrieve_more(trace, client, self.tracker)
        self.assertTrue(r.success)

    def test_zero_cost_when_not_applicable(self):
        trace = _make_trace(domain="GSM8K")
        client = _mock_openai_client()
        r = run_retrieve_more(trace, client, self.tracker)
        self.assertEqual(r.cost_usd, 0.0)
        self.assertEqual(r.total_tokens, 0)


# ---------------------------------------------------------------------------
# Tests: TOOL_FIX
# ---------------------------------------------------------------------------

class TestToolFix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_not_applicable_for_gsm8k(self):
        trace = _make_trace(domain="GSM8K")
        client = _mock_openai_client()
        r = run_tool_fix(trace, client, self.tracker)
        self.assertFalse(r.success)
        self.assertIsNotNone(r.error)
        client.chat.completions.create.assert_not_called()

    def test_applicable_for_mbpp(self):
        trace = _make_tool_trace()
        client = _mock_openai_client(answer="def fib(n): a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a")
        r = run_tool_fix(trace, client, self.tracker)
        self.assertEqual(r.action, "TOOL_FIX")
        client.chat.completions.create.assert_called_once()

    def test_tool_errors_injected_into_prompt(self):
        trace = _make_tool_trace()
        client = _mock_openai_client()
        run_tool_fix(trace, client, self.tracker)
        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        # Failed tool call from fixture has tool_name=code_execution and output with AssertionError
        self.assertIn("code_execution", user_msg)
        self.assertIn("AssertionError", user_msg)

    def test_no_tool_errors_message_when_steps_clean(self):
        # Trace with no failed tools — message should still be sent, just note "no explicit errors"
        trace = _make_tool_trace(steps=[
            {"step_index": 0, "step_type": "reasoning", "text": "thinking", "tool_name": None, "tool_args_json": None, "tool_output_json": None, "tool_call_result": None},
        ])
        client = _mock_openai_client()
        run_tool_fix(trace, client, self.tracker)
        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn("No explicit tool errors", user_msg)

    def test_uses_temperature_0(self):
        trace = _make_tool_trace()
        client = _mock_openai_client()
        run_tool_fix(trace, client, self.tracker)
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.0)


# ---------------------------------------------------------------------------
# Tests: run_recovery dispatch
# ---------------------------------------------------------------------------

class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_dispatch_escalate(self):
        trace = _make_trace()
        r = run_recovery(trace, "ESCALATE", self.tracker)
        self.assertEqual(r.action, "ESCALATE")

    def test_dispatch_local_repair(self):
        trace = _make_trace()
        r = run_recovery(trace, "LOCAL_REPAIR", self.tracker)
        self.assertEqual(r.action, "LOCAL_REPAIR")

    def test_dispatch_retry(self):
        trace = _make_trace()
        client = _mock_openai_client()
        r = run_recovery(trace, "RETRY", self.tracker, client)
        self.assertEqual(r.action, "RETRY")

    def test_raises_if_action_not_applicable(self):
        trace = _make_trace(applicable_actions=["RETRY", "ESCALATE"])
        with self.assertRaises(ValueError):
            run_recovery(trace, "RETRIEVE_MORE", self.tracker)

    def test_raises_if_client_missing_for_model_action(self):
        trace = _make_trace()
        with self.assertRaises(ValueError):
            run_recovery(trace, "RETRY", self.tracker, client=None)

    def test_raises_on_unknown_action(self):
        trace = _make_trace(applicable_actions=["BADACTION"])
        with self.assertRaises(ValueError):
            run_recovery(trace, "BADACTION", self.tracker)


# ---------------------------------------------------------------------------
# Tests: CostTracker
# ---------------------------------------------------------------------------

class TestCostTracker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = _make_tracker(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_cost_calculation_gemini(self):
        cost = self.tracker.compute_cost("google/gemini-2.0-flash-lite-001", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.375, places=3)

    def test_cost_calculation_gpt(self):
        cost = self.tracker.compute_cost("openai/gpt-5-chat", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 11.25, places=3)

    def test_unknown_model_zero_cost(self):
        cost = self.tracker.compute_cost("unknown/model", 999999, 999999)
        self.assertEqual(cost, 0.0)

    def test_records_persisted_to_jsonl(self):
        trace = _make_trace()
        run_escalate(trace, self.tracker)
        run_escalate(trace, self.tracker)
        log_path = Path(self.tmp.name) / "cost_log.jsonl"
        lines = log_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        rec = json.loads(lines[0])
        self.assertEqual(rec["action"], "ESCALATE")
        self.assertEqual(rec["trace_id"], trace.trace_id)

    def test_session_summary_aggregates_correctly(self):
        trace = _make_trace()
        client = _mock_openai_client(input_tokens=100, output_tokens=50)
        run_escalate(trace, self.tracker)
        run_retry(trace, client, self.tracker)
        summary = self.tracker.session_summary()
        self.assertEqual(summary["total_calls"], 2)
        self.assertIn("ESCALATE", summary["by_action"])
        self.assertIn("RETRY", summary["by_action"])


# ---------------------------------------------------------------------------
# Tests: verify_answer
# ---------------------------------------------------------------------------

class TestVerifyAnswer(unittest.TestCase):
    def test_gsm8k_correct(self):
        self.assertTrue(verify_answer("GSM8K", "70000", "The profit is $70,000."))

    def test_gsm8k_wrong(self):
        self.assertFalse(verify_answer("GSM8K", "70000", "The profit is $50,000."))

    def test_gsm8k_comma_format(self):
        self.assertTrue(verify_answer("GSM8K", "1200", "Answer: 1,200"))

    def test_gsm8k_negative(self):
        self.assertTrue(verify_answer("GSM8K", "-5", "the result is -5 degrees"))

    def test_sealqa_substring_match(self):
        self.assertTrue(verify_answer("SealQA", "Paris", "The capital is Paris, France."))

    def test_sealqa_wrong(self):
        self.assertFalse(verify_answer("SealQA", "Paris", "The capital is Berlin."))

    def test_medbrowse_token_overlap(self):
        # Gold tokens {"aspirin", "reduces", "fever"} all appear in recovered text
        self.assertTrue(verify_answer("MedBrowseComp", "aspirin reduces fever", "aspirin reduces fever in patients"))

    def test_empty_inputs_false(self):
        self.assertFalse(verify_answer("GSM8K", "", "70000"))
        self.assertFalse(verify_answer("GSM8K", "70000", ""))

    def test_mbpp_short_output(self):
        self.assertTrue(verify_answer("MBPP", "True", "return True"))

    def test_mbpp_wrong(self):
        self.assertFalse(verify_answer("MBPP", "True", "return False"))


# ---------------------------------------------------------------------------
# Small verbose test — 1 trace per domain, all actions, prints like traces_to_label.txt
# ---------------------------------------------------------------------------

def run_small_test():
    """Load 1 trace per domain, run all applicable recovery actions, print everything."""
    import os
    import tempfile
    from dataclasses import asdict
    from dotenv import load_dotenv
    from openai import OpenAI
    from .run_recovery import load_failed_traces

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in .env"); return

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "TraceTriage-SmallTest"},
    )

    db = Path("data/causal_runs.sqlite")
    if not db.exists():
        print(f"ERROR: {db} not found"); return

    domains = ["GSM8K", "MBPP", "SealQA", "MedBrowseComp"]
    traces  = []
    for domain in domains:
        batch = load_failed_traces(db, domain_filter=[domain], limit_per_domain=1)
        if batch:
            traces.append(batch[0])
        else:
            print(f"[WARN] No trace found for {domain}")

    SEP = "=" * 80

    with tempfile.TemporaryDirectory() as tmp:
        tracker = CostTracker(Path(tmp) / "small_test_cost.jsonl")

        for i, trace in enumerate(traces, 1):
            # ── Header ──────────────────────────────────────────────────────
            print(SEP)
            print(f"TRACE {i} / {len(traces)}")
            print(f"TRACE ID:  {trace.trace_id}")
            print(f"DOMAIN:    {trace.domain} | BENCHMARK: {trace.domain}")
            print(SEP)
            print()
            print("[PROBLEM STATEMENT]")
            print(trace.problem_statement[:500])
            print()
            print("[GOLD ANSWER]")
            print(trace.gold_answer)
            print()
            print("[AGENT FINAL ANSWER]")
            print(str(trace.final_answer)[:300])
            print()

            # ── Original steps ───────────────────────────────────────────────
            print(SEP)
            print("AGENT EXECUTION STEPS")
            print(SEP)
            for step in trace.steps:
                stype = step.get("step_type", "").upper()
                tool  = step.get("tool_name", "")
                print(f"\n--- STEP {step.get('step_index')} : {stype} ---")
                if tool:
                    print(f"Tool Used: {tool}")
                if step.get("tool_args_json"):
                    print(f"Args: {str(step['tool_args_json'])[:200]}")
                if step.get("tool_output_json"):
                    print(f"Output: {str(step['tool_output_json'])[:300]}")
                if step.get("text") and not tool:
                    print(f"Text: {str(step['text'])[:300]}")
            print()

            # ── Recovery actions ─────────────────────────────────────────────
            print(SEP)
            print(f"RECOVERY ACTIONS  (applicable: {', '.join(trace.applicable_actions)})")
            print(SEP)

            for action in trace.applicable_actions:
                print(f"\n--- ACTION: {action} ---")
                try:
                    needs_client = action in ("RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX")
                    result = run_recovery(trace, action, tracker, client if needs_client else None)
                    print(f"SUCCESS:  {result.success}")
                    print(f"ANSWER:   {str(result.recovered_answer or '')[:300]}")
                    print(f"COST:     ${result.cost_usd:.6f}")
                    print(f"TOKENS:   {result.total_tokens} (in={result.input_tokens} out={result.output_tokens})")
                    print(f"LATENCY:  {result.latency_seconds:.2f}s")
                    if result.error:
                        print(f"ERROR:    {result.error}")
                    if result.metadata:
                        print(f"METADATA: {result.metadata}")
                except Exception as exc:
                    print(f"  [EXCEPTION] {exc}")

            print()

        # ── Summary ─────────────────────────────────────────────────────────
        summary = tracker.session_summary()
        print(SEP)
        print("SESSION SUMMARY")
        print(SEP)
        print(f"Total calls:    {summary.get('total_calls', 0)}")
        print(f"Total cost:     ${summary.get('total_cost_usd', 0):.4f}")
        print(f"Total tokens:   {summary.get('total_tokens', 0)}")
        print(f"Success rate:   {summary.get('success_rate', 0):.1%}")
        print()
        print("By action:")
        for action, stats in summary.get("by_action", {}).items():
            rate = stats["successes"] / stats["calls"] if stats["calls"] else 0
            print(f"  {action:<15} calls={stats['calls']}  success={rate:.0%}  cost=${stats['cost_usd']:.4f}")


# ---------------------------------------------------------------------------
# Live integration test (optional, needs OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------

def run_live_test():
    """Call the real API with one GSM8K trace to verify the full pipeline end-to-end."""
    import os
    from openai import OpenAI
    from .run_recovery import load_failed_traces

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("SKIP live test: OPENROUTER_API_KEY not set")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "TraceTriage Test"},
    )

    db = Path("data/causal_runs.sqlite")
    if not db.exists():
        print(f"SKIP live test: {db} not found")
        return

    traces = load_failed_traces(db, domain_filter=["GSM8K"], limit_per_domain=1)
    if not traces:
        print("SKIP live test: no GSM8K traces found")
        return

    trace = traces[0]
    print(f"\n=== Live Test: RETRY on {trace.trace_id} ===")
    print(f"Problem: {trace.problem_statement[:100]}")
    print(f"Gold:    {trace.gold_answer}")
    print(f"Wrong:   {trace.final_answer}")

    with tempfile.TemporaryDirectory() as tmp:
        tracker = CostTracker(Path(tmp) / "live_cost_log.jsonl")
        result = run_retry(trace, client, tracker)

    print(f"\nResult:")
    print(f"  success:   {result.success}")
    print(f"  answer:    {(result.recovered_answer or '')[:120]}")
    print(f"  tokens:    {result.total_tokens} (in={result.input_tokens} out={result.output_tokens})")
    print(f"  cost:      ${result.cost_usd:.6f}")
    print(f"  latency:   {result.latency_seconds:.2f}s")
    if result.error:
        print(f"  error:     {result.error}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",  action="store_true", help="Run live API test on 1 GSM8K trace")
    parser.add_argument("--small", action="store_true", help="Run small verbose test: 1 trace per domain, all actions")
    args, remaining = parser.parse_known_args()

    if args.small:
        run_small_test()
        sys.exit(0)

    if args.live:
        run_live_test()

    # Pass remaining args (e.g. -v) through to unittest
    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
