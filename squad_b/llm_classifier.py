from __future__ import annotations

"""Zero-shot and few-shot LLM classification for Squad B.

Uses OpenRouter (same pattern as squad_c) to classify traces
into recovery actions via prompting.

Reproducibility: fixed few-shot example selection via RANDOM_SEED,
deterministic temperature=0 for generation, cached predictions.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from .data_loader import (
    RANDOM_SEED,
    TARGET_CLASSES,
    build_dataset,
    flatten_trace_to_text,
    get_index_splits,
)
from .evaluator import confusion_matrix_str, evaluate, print_report, save_results

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"

CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ACTION_DEFINITIONS = """Recovery Action Definitions:
- RETRIEVE_MORE: The agent failed due to insufficient or missing information. It needs to fetch additional data, run more searches, or consult extra sources.
- REPLAN: The agent's overall strategy was wrong. It needs to try a completely different approach rather than tweaking the current one.
- TOOL_FIX: A tool call failed or returned an error. The agent should fix the tool usage (correct arguments, handle errors) and retry.
- RETRY: The agent made a minor mistake (e.g., arithmetic error, off-by-one). Simply re-running with the same approach may succeed.
- ESCALATE: The problem is beyond the agent's capabilities. Flag for human review."""


def _get_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL,
                  default_headers={"HTTP-Referer": "http://localhost",
                                   "X-Title": "TraceTriage Squad B LLM"})


def _truncate_trace_text(text: str, max_chars: int = 6000) -> str:
    """Truncate trace text to fit within token limits."""
    if len(text) <= max_chars:
        return text
    # Keep beginning and end
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _parse_action(response: str) -> str:
    """Extract an action label from LLM response text."""
    response_upper = response.upper()
    # Try exact match first
    for action in TARGET_CLASSES:
        if action in response_upper:
            return action
    # Regex fallback
    match = re.search(r"(RETRIEVE_MORE|REPLAN|TOOL_FIX|RETRY|ESCALATE)", response_upper)
    if match:
        return match.group(1)
    return "UNKNOWN"


def _build_zero_shot_prompt(trace_text: str) -> str:
    return f"""{ACTION_DEFINITIONS}

Given the following failed agent trace, classify the best recovery action.
Respond with ONLY the action name (one of: RETRIEVE_MORE, REPLAN, TOOL_FIX, RETRY, ESCALATE).

Trace:
{_truncate_trace_text(trace_text)}

Best recovery action:"""


def _build_few_shot_prompt(trace_text: str, examples: list[dict]) -> str:
    example_strs = []
    for ex in examples:
        ex_text = _truncate_trace_text(ex["text"], max_chars=1500)
        example_strs.append(f"Trace:\n{ex_text}\nBest recovery action: {ex['label']}")

    examples_block = "\n\n---\n\n".join(example_strs)
    return f"""{ACTION_DEFINITIONS}

Here are some labeled examples:

{examples_block}

---

Now classify this trace. Respond with ONLY the action name.

Trace:
{_truncate_trace_text(trace_text)}

Best recovery action:"""


def _select_few_shot_examples(
    texts: list[str],
    labels: np.ndarray,
    train_idx: np.ndarray,
    k_per_class: int = 1,
) -> list[dict]:
    """Select k examples per class from the training set (deterministic)."""
    rng = np.random.RandomState(RANDOM_SEED)
    examples = []
    for cls in TARGET_CLASSES:
        cls_indices = [i for i in train_idx if labels[i] == cls]
        if not cls_indices:
            continue
        chosen = rng.choice(cls_indices, size=min(k_per_class, len(cls_indices)),
                            replace=False)
        for idx in chosen:
            examples.append({"text": texts[idx], "label": cls})
    return examples


def run_llm_classification(
    mode: str = "zero",
    test_size: float = 0.2,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    k_per_class: int = 1,
) -> dict:
    """Run LLM-based classification (zero-shot or few-shot).

    Args:
        mode: "zero" for zero-shot, "few" for few-shot.
        limit: Max test samples to classify (for cost control).
        model: OpenRouter model ID.
        k_per_class: Examples per class for few-shot mode.
    """
    client = _get_client()
    if client is None:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    print("Loading dataset...")
    ds = build_dataset(use_sqlite_features=False)
    texts, labels = ds["texts"], ds["labels"]
    trace_ids = ds["trace_ids"]

    train_idx, test_idx = get_index_splits(len(labels), labels, test_size, RANDOM_SEED)

    # For few-shot, select examples from training set
    examples = []
    if mode == "few":
        examples = _select_few_shot_examples(texts, labels, train_idx, k_per_class)
        print(f"  Selected {len(examples)} few-shot examples "
              f"({k_per_class} per class)")

    # Limit test set if requested
    eval_idx = test_idx[:limit] if limit else test_idx
    y_true = labels[eval_idx]

    method_name = f"llm_{mode}shot"
    cache_path = CACHE_DIR / f"{method_name}_predictions.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load any cached predictions
    cached_preds = {}
    if cache_path.exists():
        cached_preds = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(cached_preds)} cached predictions")

    predictions = []
    total_cost = 0.0
    total_tokens = 0

    for i, idx in enumerate(eval_idx):
        tid = trace_ids[idx]
        if tid in cached_preds:
            predictions.append(cached_preds[tid]["action"])
            continue

        if mode == "few":
            prompt = _build_few_shot_prompt(texts[idx], examples)
        else:
            prompt = _build_zero_shot_prompt(texts[idx])

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            content = resp.choices[0].message.content or ""
            action = _parse_action(content)
            usage = resp.usage
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
            total_tokens += tokens

            cached_preds[tid] = {
                "action": action, "raw": content.strip(),
                "tokens": tokens,
            }
            predictions.append(action)
            status = "OK" if action != "UNKNOWN" else "PARSE_FAIL"
            print(f"  [{i+1}/{len(eval_idx)}] {tid[:40]}... -> {action} ({status})")

        except Exception as exc:
            predictions.append("UNKNOWN")
            cached_preds[tid] = {"action": "UNKNOWN", "raw": "", "error": repr(exc)}
            print(f"  [{i+1}/{len(eval_idx)}] {tid[:40]}... -> ERROR: {exc}")

        # Rate limiting
        time.sleep(0.5)

        # Save cache periodically
        if (i + 1) % 10 == 0:
            cache_path.write_text(json.dumps(cached_preds, indent=2), encoding="utf-8")

    # Final cache save
    cache_path.write_text(json.dumps(cached_preds, indent=2), encoding="utf-8")
    print(f"\n  Total tokens: {total_tokens}")

    y_pred = np.array(predictions)
    results = evaluate(y_true, y_pred)
    print_report(results, f"LLM {mode}-shot ({model})")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_true, y_pred))
    save_results(results, method_name, extra={
        "model": model, "mode": mode, "k_per_class": k_per_class,
        "n_evaluated": len(eval_idx), "total_tokens": total_tokens,
        "random_seed": RANDOM_SEED,
    })

    return results
