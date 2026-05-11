import json
import time
import os
import re
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
from tqdm.asyncio import tqdm

# Configuration
model_type = os.getenv("MODEL_TYPE", "gpt") # 'gpt' or 'llama'
models = {'llama': 'meta-llama/llama-3.3-70b-instruct', 'gpt': 'openai/gpt-oss-120b'}
MODEL = models[model_type]
INPUT_JSONL = Path("data/labeling_exports/failed_traces.jsonl")
OUTPUT_JSONL = Path(f"data/labeling_exports/{model_type}_auto_labels_parallel.jsonl")
ERROR_JSONL = Path(f"data/labeling_exports/{model_type}_auto_label_errors_parallel.jsonl")

# Global config
OPENROUTER_API_KEY = "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"
CONCURRENCY = 5 if model_type == "gpt" else 3

client = None
SEMAPHORE = None

def read_jsonl(path):
    if not path.exists(): return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Configuration from Notebook
MAX_STEPS = 30
MAX_STEP_CHARS = 900
MAX_PROBLEM_CHARS = 1800
MAX_ANSWER_CHARS = 600

def truncate(value, max_chars):
    if value is None: return ""
    text = str(value).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text

def compact_trace_text(record):
    lines = []
    steps = record.get("steps", [])
    if MAX_STEPS is not None:
        steps = steps[:MAX_STEPS]

    for step in steps:
        header = f"Step {step.get('step_index')} | id={step.get('step_id')} | type={step.get('step_type')}"
        if step.get("tool_name"):
            header += f" | tool={step.get('tool_name')}"
        lines.append(header)

        if step.get("tool_args_json"):
            lines.append("Tool args: " + truncate(step.get("tool_args_json"), 400))

        if step.get("text"):
            lines.append("Text: " + truncate(step.get("text"), MAX_STEP_CHARS))

        if step.get("tool_output_json"):
            lines.append("Tool output: " + truncate(step.get("tool_output_json"), MAX_STEP_CHARS))

        lines.append("")

    if MAX_STEPS is not None and len(record.get("steps", [])) > MAX_STEPS:
        lines.append(f"...[{len(record['steps']) - MAX_STEPS} additional steps omitted]")

    return "\n".join(lines)

def build_prompt(record):
    actions = [a for a in json.loads(record.get("applicable_actions_json") or "[]") if a != "LOCAL_REPAIR"]
    
    return f"""
You are labeling a failed AI agent trace for the Trace Triage recovery-action task.

Choose the recovery action most likely to fix the failed trace.

Task domain: {record.get('domain')}
Problem ID: {record.get('problem_id')}
Applicable actions for this domain: {actions}

Action definitions:
- RETRY: The failure looks like a stochastic mistake, formatting issue, or minor instability. Resampling the same strategy may fix it.
- REPLAN: The agent's strategy or plan was wrong from the start. It needs a different approach, not a local tweak.
- RETRIEVE_MORE: The failure is due to missing, weak, or insufficient external evidence. More/better retrieval is likely needed.
- TOOL_FIX: The failure is due to wrong tool choice, bad tool arguments, tool execution errors, or mishandled tool use.
- ESCALATE: The failure is ambiguous, unsafe, unverifiable, or unlikely to be fixed automatically.

Important:
- Do not choose LOCAL_REPAIR here; CausalFlow local repairs were already auto-labeled separately.
- Choose only from the applicable actions listed above.
- Focus on what recovery action should be tried next, not merely which step was wrong.

Problem:
{truncate(record.get('problem_statement'), MAX_PROBLEM_CHARS)}

Correct answer:
{truncate(record.get('gold_answer'), MAX_ANSWER_CHARS)}

Agent's wrong final answer:
{truncate(record.get('final_answer'), MAX_ANSWER_CHARS)}

Agent execution trace:
{compact_trace_text(record)}
""".strip()

async def call_model_label(record):
    async with SEMAPHORE:
        prompt = build_prompt(record)
        user_message = prompt + """

Return strictly valid JSON with this shape:
{
  "action": "RETRY | REPLAN | RETRIEVE_MORE | TOOL_FIX | ESCALATE",
  "rationale": "one concise sentence",
  "confidence": 0.0
}
"""
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert annotator for failed LLM-agent traces. Return only valid JSON matching the requested schema. Be conservative and use ESCALATE when the trace is too ambiguous to diagnose."},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        return {
            "trace_id": record["trace_id"],
            "problem_id": record.get("problem_id"),
            "domain": record.get("domain"),
            "provider": "openrouter",
            "model": MODEL,
            "action": data.get("action"),
            "rationale": data.get("rationale"),
            "confidence": data.get("confidence", 0.0),
            "allowed_actions": [a for a in json.loads(record.get("applicable_actions_json") or "[]") if a != "LOCAL_REPAIR"]
        }

async def process_record(record, pbar):
    try:
        label = await call_model_label(record)
        append_jsonl(OUTPUT_JSONL, label)
    except Exception as e:
        error_str = str(e)
        # print(f"Error for {record['trace_id']}: {error_str}")
        append_jsonl(ERROR_JSONL, {"trace_id": record["trace_id"], "error": error_str})
    finally:
        pbar.update(1)

async def main():
    global client, SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(CONCURRENCY)
    client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    
    existing_labels = {}
    if OUTPUT_JSONL.exists():
        for r in read_jsonl(OUTPUT_JSONL):
            if r.get("trace_id") and r.get("action"):
                existing_labels[r["trace_id"]] = r

    records = list(read_jsonl(INPUT_JSONL))
    to_label = []
    
    for record in records:
        if record.get("needs_labeling") == 0 or record.get("is_local_repairable") == 1:
            if record["trace_id"] not in existing_labels:
                label = {
                    "trace_id": record["trace_id"],
                    "problem_id": record.get("problem_id"),
                    "domain": record.get("domain"),
                    "provider": "causalflow",
                    "model": None,
                    "action": "LOCAL_REPAIR",
                    "rationale": "CausalFlow found at least one successful local counterfactual repair for this failed trace.",
                    "confidence": 1.0,
                    "allowed_actions": json.loads(record.get("applicable_actions_json") or "[]"),
                    "label_source": "auto_causalflow",
                }
                append_jsonl(OUTPUT_JSONL, label)
                existing_labels[record["trace_id"]] = label
        elif record["trace_id"] not in existing_labels:
            to_label.append(record)

    print(f"Loaded {len(existing_labels)} existing labels.")
    print(f"Starting {model_type} run for {len(to_label)} remaining records...")

    with tqdm(total=len(to_label)) as pbar:
        tasks = [process_record(r, pbar) for r in to_label]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
