"""Cost and latency tracking for recovery action calls.

Every model call records tokens used, cost, and latency to a JSONL file.
The CostTracker is thread-safe for concurrent recovery runs.
"""
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# USD per million tokens. Update if OpenRouter pricing changes.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "google/gemini-2.0-flash-lite-001": {"input": 0.075, "output": 0.30},
    "google/gemini-3-flash-preview":    {"input": 0.10,  "output": 0.40},
    "openai/gpt-5-chat":                {"input": 1.25,  "output": 10.00},
    "openai/gpt-4o":                   {"input": 5.00,  "output": 15.00},
}

# Maps domain name to the model used in the original CausalFlow runs.
DOMAIN_MODELS: dict[str, str] = {
    "GSM8K":        "google/gemini-2.0-flash-lite-001",
    "MBPP":         "openai/gpt-5-chat",
    "SealQA":       "google/gemini-3-flash-preview",
    "MedBrowseComp":"google/gemini-3-flash-preview",
    "BrowseComp":   "google/gemini-3-flash-preview",
}


@dataclass
class CallRecord:
    trace_id: str
    action: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    metadata: dict = field(default_factory=dict)


class CostTracker:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._session: list[CallRecord] = []

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self._session.append(rec)
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def make_zero_cost_record(
        self,
        trace_id: str,
        action: str,
        model: str,
        success: bool,
        metadata: dict | None = None,
    ) -> CallRecord:
        """Create a cost-free record for actions that skip model calls (ESCALATE, LOCAL_REPAIR)."""
        return CallRecord(
            trace_id=trace_id,
            action=action,
            model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_seconds=0.0,
            success=success,
            metadata=metadata or {},
        )

    def session_summary(self) -> dict:
        if not self._session:
            return {}
        total_cost = sum(r.cost_usd for r in self._session)
        total_tokens = sum(r.total_tokens for r in self._session)
        successes = sum(1 for r in self._session if r.success)
        return {
            "total_calls": len(self._session),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "total_latency_seconds": round(sum(r.latency_seconds for r in self._session), 2),
            "success_rate": round(successes / len(self._session), 4),
            "by_action": _group_records(self._session, "action"),
            "by_model": _group_records(self._session, "model"),
        }


def _group_records(records: list[CallRecord], attr: str) -> dict:
    groups: dict[str, dict] = {}
    for r in records:
        key = getattr(r, attr)
        if key not in groups:
            groups[key] = {"calls": 0, "cost_usd": 0.0, "tokens": 0, "successes": 0}
        groups[key]["calls"] += 1
        groups[key]["cost_usd"] += r.cost_usd
        groups[key]["tokens"] += r.total_tokens
        groups[key]["successes"] += int(r.success)
    return groups
