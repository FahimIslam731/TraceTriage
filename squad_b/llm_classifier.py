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
    get_train_test_indices,
)
from .evaluator import confusion_matrix_str, evaluate, print_report, save_results

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
MODEL_PRESETS = {
    "default": DEFAULT_MODEL,
    "gpt5-mini": "openai/gpt-5-mini",
    "gemini-flash-lite": "google/gemini-2.0-flash-lite-001",
}

CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ACTION_DEFINITIONS = """Recovery Action Definitions:
- LOCAL_REPAIR: CausalFlow found a localized reasoning, code, or tool step whose counterfactual repair resolves the failure.
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
    for action in sorted(TARGET_CLASSES, key=len, reverse=True):
        if action in response_upper:
            return action
    # Regex fallback
    match = re.search(r"(LOCAL_REPAIR|RETRIEVE_MORE|REPLAN|TOOL_FIX|RETRY|ESCALATE)", response_upper)
    if match:
        return match.group(1)
    return "UNKNOWN"


def _parse_prediction(response: str) -> dict:
    """Parse action/confidence from either JSON or plain action responses."""
    text = response.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"action": _parse_action(text), "confidence": None}

    action = _parse_action(str(data.get("action", "")))
    confidence = data.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    return {"action": action, "confidence": confidence}


def _message_content_to_text(content) -> str:
    """Normalize OpenAI/OpenRouter message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def resolve_model(model: str | None = None, model_preset: str = "default") -> str:
    """Resolve a CLI model preset or explicit OpenRouter model ID."""
    if model:
        return model
    return MODEL_PRESETS.get(model_preset, DEFAULT_MODEL)


def _safe_cache_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _build_zero_shot_prompt(trace_text: str, json_output: bool = False) -> str:
    output_instruction = (
        'Respond with JSON: {"action": "<one action>", "confidence": <0.0 to 1.0>}.'
        if json_output
        else f"Respond with ONLY the action name (one of: {', '.join(TARGET_CLASSES)})."
    )
    return f"""You are an expert at classifying why large language models fail based on analysing their traces.
    You are extremely skilled at this task.   
    
    Here are the action labels you may assign.
    {ACTION_DEFINITIONS}

    Given the following failed agent trace, classify the best recovery action label from the action labels above.
    {output_instruction}

    Trace:
    {_truncate_trace_text(trace_text)}

    Best recovery action:"""


def _build_few_shot_prompt(trace_text: str, examples: list[dict], json_output: bool = False) -> str:
    example_strs = []
    for ex in examples:
        ex_text = _truncate_trace_text(ex["text"], max_chars=1500)
        if json_output:
            answer = json.dumps({"action": ex["label"], "confidence": 1.0})
        else:
            answer = ex["label"]
        example_strs.append(f"Trace:\n{ex_text}\nBest recovery action: {answer}")

    examples_block = "\n\n---\n\n".join(example_strs)
    return f""" You are an expert at classifying why large language models fail based on analysing their traces.
    You are extremely skilled at this task.
    
    Here are the action labels you may assign.
    {ACTION_DEFINITIONS}

    Here are some labeled examples:

    {examples_block}

    ---

    Now classify this trace. {"Respond with JSON including action and confidence." if json_output else "Respond with ONLY the action name."}

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


def select_limited_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    limit: int | None,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Select a deterministic, roughly class-balanced subset of indices."""
    if limit is None or limit >= len(indices):
        return indices

    rng = np.random.RandomState(seed)
    selected = []
    per_class = max(1, limit // len(TARGET_CLASSES))
    for cls in TARGET_CLASSES:
        cls_indices = [i for i in indices if labels[i] == cls]
        if cls_indices:
            chosen = rng.choice(
                cls_indices,
                size=min(per_class, len(cls_indices)),
                replace=False,
            )
            selected.extend(chosen)

    if len(selected) < limit:
        remaining = [i for i in indices if i not in selected]
        if remaining:
            top_up = rng.choice(
                remaining,
                size=min(limit - len(selected), len(remaining)),
                replace=False,
            )
            selected.extend(top_up)

    selected_arr = np.array(selected)
    rng.shuffle(selected_arr)
    return selected_arr


def run_llm_on_indices(
    ds: dict,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    mode: str = "zero",
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    model_preset: str = "default",
    k_per_class: int = 1,
    json_output: bool = False,
    input_variant: str = "full_trace",
    cache_scope: str = "",
    result_name: str | None = None,
    print_prompts: bool = False,
    max_tokens: int = 512,
    save_result: bool = True,
    extra: dict | None = None,
) -> dict:
    """Classify a caller-provided evaluation split with an LLM."""
    model = resolve_model(model if model != DEFAULT_MODEL else None, model_preset)
    client = _get_client()
    if client is None:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    texts, labels = ds["texts"], ds["labels"]
    trace_ids = ds["trace_ids"]

    examples = []
    if mode == "few":
        examples = _select_few_shot_examples(texts, labels, train_idx, k_per_class)
        print(f"  Selected {len(examples)} few-shot examples "
              f"({k_per_class} per class)")

    eval_idx = select_limited_indices(eval_idx, labels, limit)
    y_true = labels[eval_idx]

    method_name = f"llm_{mode}shot"
    cache_parts = [
        method_name,
        model,
        input_variant,
        f"k{k_per_class}",
        "json" if json_output else "plain",
    ]
    if cache_scope:
        cache_parts.append(cache_scope)
    cache_name = _safe_cache_name("_".join(cache_parts))
    cache_path = CACHE_DIR / f"{cache_name}_predictions.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cached_preds = {}
    if cache_path.exists():
        cached_preds = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(cached_preds)} cached predictions from {cache_path}")

    predictions = []
    total_tokens = 0

    for i, idx in enumerate(eval_idx):
        tid = trace_ids[idx]
        if tid in cached_preds:
            predictions.append(cached_preds[tid]["action"])
            continue

        if mode == "few":
            prompt = _build_few_shot_prompt(texts[idx], examples, json_output=json_output)
        else:
            prompt = _build_zero_shot_prompt(texts[idx], json_output=json_output)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            content = _message_content_to_text(choice.message.content).strip()
            parsed = _parse_prediction(content)
            action = parsed["action"]
            confidence = parsed["confidence"]
            usage = resp.usage
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
            total_tokens += tokens

            finish_reason = getattr(choice, "finish_reason", None)
            cache_row = {
                "action": action,
                "raw": content,
                "tokens": tokens,
                "confidence": confidence,
                "finish_reason": finish_reason,
            }
            if content:
                cached_preds[tid] = cache_row
            else:
                print(
                    "  Empty model response; not caching this prediction. "
                    f"Try --max-tokens {max_tokens * 2} or a non-reasoning model."
                )
            predictions.append(action)
            status = "OK" if action != "UNKNOWN" else "PARSE_FAIL"
            print(f"  [{i+1}/{len(eval_idx)}] {tid[:40]}... -> {action} ({status})")

        except Exception as exc:
            predictions.append("UNKNOWN")
            cached_preds[tid] = {"action": "UNKNOWN", "raw": "", "error": repr(exc)}
            print(f"  [{i+1}/{len(eval_idx)}] {tid[:40]}... -> ERROR: {exc}")

        time.sleep(0.5)

        if (i + 1) % 10 == 0:
            cache_path.write_text(json.dumps(cached_preds, indent=2), encoding="utf-8")

    cache_path.write_text(json.dumps(cached_preds, indent=2), encoding="utf-8")
    print(f"\n  Total tokens: {total_tokens}")

    y_pred = np.array(predictions)
    results = evaluate(y_true, y_pred)
    print_report(results, f"LLM {mode}-shot ({model})")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_true, y_pred))

    if save_result:
        result_payload = {
            "model": model,
            "mode": mode,
            "k_per_class": k_per_class,
            "n_evaluated": len(eval_idx),
            "total_tokens": total_tokens,
            "random_seed": RANDOM_SEED,
            "json_output": json_output,
            "input_variant": input_variant,
            "max_tokens": max_tokens,
            "cache_path": str(cache_path),
        }
        if extra:
            result_payload.update(extra)
        save_results(results, result_name or method_name, extra=result_payload)

    return {
        "metrics": results,
        "predictions": y_pred,
        "gold": y_true,
        "eval_idx": eval_idx,
        "cache_path": str(cache_path),
        "total_tokens": total_tokens,
    }


def run_llm_classification(
    mode: str = "zero",
    test_size: float = 0.2,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    model_preset: str = "default",
    k_per_class: int = 1,
    json_output: bool = False,
    input_variant: str = "full_trace",
    print_prompts: bool = False,
    max_tokens: int = 512,
) -> dict:
    """Run LLM-based classification (zero-shot or few-shot).

    Args:
        mode: "zero" for zero-shot, "few" for few-shot.
        limit: Max test samples to classify (for cost control).
        model: OpenRouter model ID.
        k_per_class: Examples per class for few-shot mode.
    """
    print("Loading dataset...")
    ds = build_dataset(use_sqlite_features=False, input_variant=input_variant)
    labels = ds["labels"]

    train_idx, test_idx, dev_idx = get_train_test_indices(
        ds, labels, test_size, RANDOM_SEED
    )
    output = run_llm_on_indices(
        ds=ds,
        train_idx=train_idx,
        eval_idx=test_idx,
        mode=mode,
        limit=limit,
        model=model,
        model_preset=model_preset,
        k_per_class=k_per_class,
        json_output=json_output,
        input_variant=input_variant,
        print_prompts=print_prompts,
        max_tokens=max_tokens,
        extra={"split_source": "squad_a_frozen" if dev_idx is not None else "random"},
    )

    return output["metrics"]
