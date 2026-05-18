"""6 recovery action functions for Trace Triage (Squad C, Week 1).

Each function accepts a FailedTrace and returns a RecoveryResult.
All model-calling actions record token usage and latency via CostTracker.

Domain -> model mapping matches the original CausalFlow runs:
  GSM8K        -> google/gemini-2.0-flash-lite-001
  MBPP         -> openai/gpt-oss-120b
  SealQA       -> google/gemini-3-flash-preview
  MedBrowseComp-> google/gemini-3-flash-preview

Environment variables:
  OPENROUTER_API_KEY   required for RETRY, REPLAN, RETRIEVE_MORE, TOOL_FIX
  SERPER_API_KEY       required for RETRIEVE_MORE (web search augmentation)
"""
import html.parser
import json
import os
import re
import time
import urllib.request
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

# Serper settings (paper spec: 5 extra search/open/extract steps per recovery)
_SERPER_URL = "https://google.serper.dev/search"
_SERPER_MAX_QUERIES = 5       # matches paper: "5 extra search/open/extract steps"
_SERPER_RESULTS_PER_QUERY = 5 # snippets per search


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


def _extract_prior_search_queries(steps: list[dict]) -> list[str]:
    """Pull search queries the original agent already used."""
    queries = []
    for step in steps:
        if step.get("tool_name") == "web_search":
            try:
                args = json.loads(step.get("tool_args_json") or "{}")
                q = args.get("query") or args.get("q") or args.get("input", "")
                if q and str(q).strip() not in queries:
                    queries.append(str(q).strip())
            except (json.JSONDecodeError, AttributeError):
                pass
    return queries


class _TextExtractor(html.parser.HTMLParser):
    """Minimal HTML-to-text extractor using stdlib only."""
    _SKIP = frozenset(("script", "style", "nav", "footer", "header", "aside", "noscript"))

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._depth = 0  # skip-tag nesting depth

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth > 0:
            self._depth -= 1

    def handle_data(self, data):
        if self._depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def _web_fetch(url: str, max_chars: int = 2000) -> str:
    """Fetch a URL and return cleaned page text (stdlib only, no extra deps)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; TraceTriage/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(1_000_000)  # cap at 1 MB
            content_type = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            page_html = raw.decode(charset, errors="replace")
    except Exception as exc:
        return f"[fetch failed: {exc}]"

    extractor = _TextExtractor()
    extractor.feed(page_html)
    text = extractor.get_text()
    return text[:max_chars] + "...[truncated]" if len(text) > max_chars else text


def _serper_search(queries: list[str], api_key: str) -> list[dict]:
    """Run up to _SERPER_MAX_QUERIES through Serper.

    Returns a list of dicts: {query, results: [{title, snippet, url}]}
    """
    output = []
    for query in queries[:_SERPER_MAX_QUERIES]:
        payload = json.dumps({"q": query, "num": _SERPER_RESULTS_PER_QUERY}).encode()
        req = urllib.request.Request(
            _SERPER_URL,
            data=payload,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            continue
        results = [
            {
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url":     item.get("link", ""),
            }
            for item in data.get("organic", [])[:_SERPER_RESULTS_PER_QUERY]
            if item.get("title") or item.get("snippet")
        ]
        if results:
            output.append({"query": query, "results": results})
    return output


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
    """Fetch real web evidence via Serper, then answer (SealQA, MedBrowse only).

    Strategy:
      1. Build search queries: problem statement + prior agent queries (deduplicated).
      2. Run up to 5 searches through Serper (paper spec: 5 extra search/open/extract steps).
      3. Feed retrieved snippets to the model to synthesize a final answer.
    Falls back to reasoning-only if SERPER_API_KEY is not set.
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
    serper_key = os.environ.get("SERPER_API_KEY", "")

    # Build search queries: problem statement first, then prior agent queries
    prior_queries = _extract_prior_search_queries(trace.steps)
    primary_query = _trunc(trace.problem_statement, 200).replace("\n", " ")
    all_queries = [primary_query] + [q for q in prior_queries if q != primary_query]

    queries_run: list[str] = []

    if serper_key:
        # Step 1: Search — get results including URLs
        search_data = _serper_search(all_queries, serper_key)

        # Step 2: Fetch top URL per query (open + extract, matching paper spec)
        evidence_parts = []
        for entry in search_data:
            queries_run.append(entry["query"])
            top = entry["results"][0] if entry["results"] else None
            snippet_lines = "\n".join(
                f"  - {r['title']}: {r['snippet']}" for r in entry["results"]
            )
            section = f'Search "{entry["query"]}":\n{snippet_lines}'

            if top and top["url"]:
                page_text = _web_fetch(top["url"])
                section += f'\n  Full page ({top["url"]}):\n  {page_text}'

            evidence_parts.append(section)

        evidence_section = "Web search results (search → fetch → extract):\n\n" + \
                           "\n\n".join(evidence_parts) if evidence_parts else "No results retrieved."
    else:
        evidence_section = (
            "No web search available. Reason through what additional sources "
            "would be needed and provide your best answer from existing knowledge."
        )

    user_msg = (
        f"The previous agent attempt FAILED to answer this question correctly.\n"
        f"Previous (wrong) answer: {_trunc(trace.final_answer, _MAX_ANSWER_CHARS)}\n\n"
        f"Problem:\n{_trunc(trace.problem_statement, _MAX_PROBLEM_CHARS)}\n\n"
        f"{evidence_section}\n\n"
        "Based on the evidence above, what is the correct answer? "
        "Be specific and direct."
    )
    t0 = time.perf_counter()
    try:
        answer, in_tok, out_tok = _call_model(
            client, model, _system_prompt(trace.domain), user_msg, temperature=0.3,
        )
        latency = time.perf_counter() - t0
        return _build_result_and_record(
            trace, "RETRIEVE_MORE", model, answer, in_tok, out_tok, latency, tracker,
            metadata={"serper_used": bool(serper_key), "queries_run": queries_run},
        )
    except Exception as exc:
        latency = time.perf_counter() - t0
        return _build_result_and_record(
            trace, "RETRIEVE_MORE", model, "", 0, 0, latency, tracker, error=repr(exc),
        )


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
