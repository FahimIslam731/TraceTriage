"""6 recovery action functions for Trace Triage (Squad C, Week 1).

Each function accepts a FailedTrace and returns a RecoveryResult.
All model-calling actions record token usage and latency via CostTracker.

Domain -> model mapping matches the original CausalFlow runs:
  GSM8K        -> google/gemini-2.0-flash-lite-001
  MBPP         -> openai/gpt-oss-120b
  SealQA       -> google/gemini-3-flash-preview
  MedBrowseComp-> google/gemini-3-flash-preview
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .cost_tracker import DOMAIN_MODELS, CallRecord, CostTracker
from .verify import verify_answer

# Actions restricted to specific domains (matches paper taxonomy)
_RETRIEVE_MORE_DOMAINS = {"SealQA", "MedBrowseComp", "BrowseComp"}
_TOOL_FIX_DOMAINS = {"MBPP", "SealQA", "MedBrowseComp", "BrowseComp"}

_MAX_STEP_CHARS = 600
_MAX_PROBLEM_CHARS = 2000
_MAX_ANSWER_CHARS = 600


@dataclass
class FailedTrace:
    trace_id: str
    problem_id: str
    domain: str
    model_used: str          # original agent model (may be empty; falls back to DOMAIN_MODELS)
    problem_statement: str
    gold_answer: str
    final_answer: str        # the wrong answer from the original run
    steps: list[dict]
    applicable_actions: list[str]
    # CausalFlow repair data (populated only for LOCAL_REPAIR candidates)
    repaired_text: Optional[str] = None
    repair_step_id: Optional[int] = None


@dataclass
class RecoveryResult:
    trace_id: str
    action: str
    success: bool
    recovered_answer: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float
    model_used: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_for(trace: FailedTrace) -> str:
    if trace.model_used:
        return trace.model_used
    return DOMAIN_MODELS.get(trace.domain, "google/gemini-3-flash-preview")


def _trunc(value, max_chars: int) -> str:
    if not value:
        return ""
    text = str(value).strip()
    return text[:max_chars] + "\n...[truncated]" if len(text) > max_chars else text


def _compact_steps(steps: list[dict], max_steps: int = 20) -> str:
    lines = []
    for step in steps[:max_steps]:
        header = (f"Step {step.get('step_index')} | type={step.get('step_type')}"
                  f"{' | tool=' + step['tool_name'] if step.get('tool_name') else ''}")
        lines.append(header)
        if step.get("tool_args_json"):
            lines.append("  args: " + _trunc(step["tool_args_json"], 300))
        if step.get("text"):
            lines.append("  text: " + _trunc(step["text"], _MAX_STEP_CHARS))
        if step.get("tool_output_json"):
            lines.append("  output: " + _trunc(step["tool_output_json"], _MAX_STEP_CHARS))
    if len(steps) > max_steps:
        lines.append(f"...[{len(steps) - max_steps} more steps omitted]")
    return "\n".join(lines)


def _extract_tool_errors(steps: list[dict]) -> str:
    """Collect failed tool calls from the trace to feed into TOOL_FIX."""
    errors = []
    for step in steps:
        if step.get("tool_call_result") == 0 and step.get("tool_name"):
            args = _trunc(step.get("tool_args_json", ""), 300)
            out = _trunc(step.get("tool_output_json", ""), 300)
            errors.append(
                f"Tool '{step['tool_name']}' failed.\n  Args: {args}\n  Output: {out}"
            )
    return "\n\n".join(errors) if errors else "No explicit tool errors recorded in trace."


def _system_prompt(domain: str) -> str:
    prompts = {
        "GSM8K": (
            "You are a math problem solver. Work through the problem step by step. "
            "State your final numerical answer clearly at the end."
        ),
        "MBPP": (
            "You are an expert Python programmer. Write correct, complete Python code. "
            "Output only the function implementation, no extra explanation."
        ),
    }
    return prompts.get(domain, (
        "You are a research assistant. Search thoroughly and provide an accurate, "
        "specific final answer. Be concise and direct."
    ))


def _call_model(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
) -> tuple[str, int, int]:
    """Call the model and return (content, input_tokens, output_tokens)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    return content, usage.prompt_tokens, usage.completion_tokens


def _build_result_and_record(
    trace: FailedTrace,
    action: str,
    model: str,
    recovered_answer: str,
    input_tokens: int,
    output_tokens: int,
    latency: float,
    tracker: CostTracker,
    error: Optional[str] = None,
    metadata: dict | None = None,
) -> RecoveryResult:
    cost = tracker.compute_cost(model, input_tokens, output_tokens)
    success = (
        error is None
        and verify_answer(trace.domain, trace.gold_answer, recovered_answer)
    )
    rec = CallRecord(
        trace_id=trace.trace_id,
        action=action,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost,
        latency_seconds=round(latency, 3),
        success=success,
        error=error,
        metadata=metadata or {},
    )
    tracker.record(rec)
    return RecoveryResult(
        trace_id=trace.trace_id,
        action=action,
        success=success,
        recovered_answer=recovered_answer if not error else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost,
        latency_seconds=round(latency, 3),
        model_used=model,
        error=error,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# 6 Recovery Action Functions
# ---------------------------------------------------------------------------

def run_local_repair(trace: FailedTrace, tracker: CostTracker) -> RecoveryResult:
    """Return the CausalFlow repair already computed — no new model call needed."""
    model = _model_for(trace)
    # repaired_text comes from repair_attempts joined at load time
    recovered = trace.repaired_text or ""
    success = bool(recovered)  # CausalFlow validated this repair already
    rec = tracker.make_zero_cost_record(
        trace_id=trace.trace_id,
        action="LOCAL_REPAIR",
        model=model,
        success=success,
        metadata={"repair_step_id": trace.repair_step_id, "repaired_text": recovered[:300]},
    )
    tracker.record(rec)
    return RecoveryResult(
        trace_id=trace.trace_id,
        action="LOCAL_REPAIR",
        success=success,
        recovered_answer=recovered or None,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        latency_seconds=0.0,
        model_used=model,
        metadata={"repair_step_id": trace.repair_step_id},
    )


def run_retry(trace: FailedTrace, client: OpenAI, tracker: CostTracker) -> RecoveryResult:
    """Resample the agent at temperature=1.0 with the same problem, no strategy change."""
    model = _model_for(trace)
    user_msg = (
        f"Problem:\n{_trunc(trace.problem_statement, _MAX_PROBLEM_CHARS)}\n\n"
        "Solve this problem and provide your final answer."
    )
    t0 = time.perf_counter()
    try:
        answer, in_tok, out_tok = _call_model(client, model, _system_prompt(trace.domain), user_msg, temperature=1.0)
        latency = time.perf_counter() - t0
        return _build_result_and_record(trace, "RETRY", model, answer, in_tok, out_tok, latency, tracker)
    except Exception as exc:
        latency = time.perf_counter() - t0
        return _build_result_and_record(trace, "RETRY", model, "", 0, 0, latency, tracker, error=repr(exc))


def run_replan(trace: FailedTrace, client: OpenAI, tracker: CostTracker) -> RecoveryResult:
    """Restart with an explicit instruction to use a completely different strategy."""
    model = _model_for(trace)
    user_msg = (
        f"Your previous attempt at this problem was WRONG.\n"
        f"Previous (incorrect) answer: {_trunc(trace.final_answer, _MAX_ANSWER_CHARS)}\n\n"
        "Try a completely different approach or strategy. Do NOT repeat the same steps.\n\n"
        f"Problem:\n{_trunc(trace.problem_statement, _MAX_PROBLEM_CHARS)}\n\n"
        "Provide your final answer using the new approach."
    )
    t0 = time.perf_counter()
    try:
        answer, in_tok, out_tok = _call_model(client, model, _system_prompt(trace.domain), user_msg, temperature=0.7)
        latency = time.perf_counter() - t0
        return _build_result_and_record(trace, "REPLAN", model, answer, in_tok, out_tok, latency, tracker)
    except Exception as exc:
        latency = time.perf_counter() - t0
        return _build_result_and_record(trace, "REPLAN", model, "", 0, 0, latency, tracker, error=repr(exc))


def run_retrieve_more(trace: FailedTrace, client: OpenAI, tracker: CostTracker) -> RecoveryResult:
    """Allow extended retrieval reasoning before answering (SealQA, MedBrowse only).

    The prompt explicitly asks the model to reason through what additional
    sources it would consult, then provide a final answer based on that extended
    research. Full tool-call integration (Serper) can be wired in later.
    """
    if trace.domain not in _RETRIEVE_MORE_DOMAINS:
        rec = tracker.make_zero_cost_record(
            trace.trace_id, "RETRIEVE_MORE", _model_for(trace), success=False,
            metadata={"skip_reason": f"domain {trace.domain} not applicable"},
        )
        tracker.record(rec)
        return RecoveryResult(
            trace_id=trace.trace_id, action="RETRIEVE_MORE", success=False,
            recovered_answer=None, input_tokens=0, output_tokens=0, total_tokens=0,
            cost_usd=0.0, latency_seconds=0.0, model_used=_model_for(trace),
            error=f"RETRIEVE_MORE not applicable to domain {trace.domain}",
        )

    model = _model_for(trace)
    trace_summary = _compact_steps(trace.steps, max_steps=15)
    user_msg = (
        f"The previous agent attempt FAILED to answer this question correctly.\n"
        f"Previous (wrong) answer: {_trunc(trace.final_answer, _MAX_ANSWER_CHARS)}\n\n"
        "The failure appears to be due to insufficient or missing information. "
        "Reason through what additional sources, searches, or evidence would be needed, "
        "then provide your best final answer based on that extended research.\n\n"
        f"Problem:\n{_trunc(trace.problem_statement, _MAX_PROBLEM_CHARS)}\n\n"
        f"What the previous agent did (steps summary):\n{trace_summary}\n\n"
        "Based on more thorough research, what is the correct answer?"
    )
    t0 = time.perf_counter()
    try:
        answer, in_tok, out_tok = _call_model(client, model, _system_prompt(trace.domain), user_msg, temperature=0.3)
        latency = time.perf_counter() - t0
        return _build_result_and_record(trace, "RETRIEVE_MORE", model, answer, in_tok, out_tok, latency, tracker)
    except Exception as exc:
        latency = time.perf_counter() - t0
        return _build_result_and_record(trace, "RETRIEVE_MORE", model, "", 0, 0, latency, tracker, error=repr(exc))


def run_tool_fix(trace: FailedTrace, client: OpenAI, tracker: CostTracker) -> RecoveryResult:
    """Re-run the agent with explicit tool error feedback prepended (MBPP, SealQA, MedBrowse only)."""
    if trace.domain not in _TOOL_FIX_DOMAINS:
        rec = tracker.make_zero_cost_record(
            trace.trace_id, "TOOL_FIX", _model_for(trace), success=False,
            metadata={"skip_reason": f"domain {trace.domain} not applicable"},
        )
        tracker.record(rec)
        return RecoveryResult(
            trace_id=trace.trace_id, action="TOOL_FIX", success=False,
            recovered_answer=None, input_tokens=0, output_tokens=0, total_tokens=0,
            cost_usd=0.0, latency_seconds=0.0, model_used=_model_for(trace),
            error=f"TOOL_FIX not applicable to domain {trace.domain}",
        )

    model = _model_for(trace)
    tool_errors = _extract_tool_errors(trace.steps)
    user_msg = (
        f"The previous agent attempt FAILED due to tool errors.\n\n"
        f"Tool errors from the previous attempt:\n{tool_errors}\n\n"
        "Fix the tool usage errors and solve the problem correctly.\n\n"
        f"Problem:\n{_trunc(trace.problem_statement, _MAX_PROBLEM_CHARS)}\n\n"
        f"Previous (wrong) answer: {_trunc(trace.final_answer, _MAX_ANSWER_CHARS)}\n\n"
        "Provide the correct final answer, avoiding the tool errors above."
    )
    t0 = time.perf_counter()
    try:
        answer, in_tok, out_tok = _call_model(client, model, _system_prompt(trace.domain), user_msg, temperature=0.0)
        latency = time.perf_counter() - t0
        return _build_result_and_record(
            trace, "TOOL_FIX", model, answer, in_tok, out_tok, latency, tracker,
            metadata={"tool_errors_found": tool_errors[:200]},
        )
    except Exception as exc:
        latency = time.perf_counter() - t0
        return _build_result_and_record(trace, "TOOL_FIX", model, "", 0, 0, latency, tracker, error=repr(exc))


def run_escalate(trace: FailedTrace, tracker: CostTracker) -> RecoveryResult:
    """Mark trace as unrepairable — no model call, cost=0, success=0."""
    model = _model_for(trace)
    rec = tracker.make_zero_cost_record(
        trace_id=trace.trace_id,
        action="ESCALATE",
        model=model,
        success=False,
        metadata={"reason": "flagged for human review"},
    )
    tracker.record(rec)
    return RecoveryResult(
        trace_id=trace.trace_id,
        action="ESCALATE",
        success=False,
        recovered_answer=None,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        latency_seconds=0.0,
        model_used=model,
        metadata={"reason": "flagged for human review"},
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ACTION_REQUIRES_CLIENT = {"RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX"}


def run_recovery(
    trace: FailedTrace,
    action: str,
    tracker: CostTracker,
    client: Optional[OpenAI] = None,
) -> RecoveryResult:
    """Dispatch to the correct recovery function by action name."""
    if action not in trace.applicable_actions:
        raise ValueError(f"Action {action} not in applicable_actions for {trace.domain}: {trace.applicable_actions}")
    if action in ACTION_REQUIRES_CLIENT and client is None:
        raise ValueError(f"Action {action} requires an OpenAI client")

    dispatch = {
        "LOCAL_REPAIR":   lambda: run_local_repair(trace, tracker),
        "RETRY":          lambda: run_retry(trace, client, tracker),
        "REPLAN":         lambda: run_replan(trace, client, tracker),
        "RETRIEVE_MORE":  lambda: run_retrieve_more(trace, client, tracker),
        "TOOL_FIX":       lambda: run_tool_fix(trace, client, tracker),
        "ESCALATE":       lambda: run_escalate(trace, tracker),
    }
    if action not in dispatch:
        raise ValueError(f"Unknown action: {action}")
    return dispatch[action]()
