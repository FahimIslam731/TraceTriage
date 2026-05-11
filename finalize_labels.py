import json
from pathlib import Path

def finalize_labels(model_type, reference_path, parallel_path, error_path, fallback_path, output_path):
    print(f"Finalizing {model_type} labels...")
    
    # 1. Load reference order
    ref_order = []
    with open(reference_path, "r") as f:
        for line in f:
            if line.strip():
                ref_order.append(json.loads(line)["trace_id"])
    
    # 2. Load parallel results
    results = {}
    with open(parallel_path, "r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                results[d["trace_id"]] = d
                
    # 3. Load fallback (original results from serial run)
    fallbacks = {}
    with open(fallback_path, "r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                fallbacks[d["trace_id"]] = d
                
    # 4. Write final file in correct order
    with open(output_path, "w") as f:
        for trace_id in ref_order:
            if trace_id in results and results[trace_id].get("action"):
                # Use parallel result if valid
                f.write(json.dumps(results[trace_id]) + "\n")
            elif trace_id in fallbacks:
                # Fallback to serial result if parallel failed
                # Update model name if it was a GPT fallback
                fb = fallbacks[trace_id]
                if model_type == "gpt":
                    fb["model"] = "openai/gpt-oss-120b"
                f.write(json.dumps(fb) + "\n")
            else:
                print(f"Warning: No label found for {trace_id}")

# Paths
REFERENCE = "data/labeling_exports/llama_auto_labels.jsonl" # This has the 638 order

# GPT
finalize_labels(
    "gpt",
    REFERENCE,
    "data/labeling_exports/gpt_auto_labels_parallel.jsonl",
    "data/labeling_exports/gpt_auto_label_errors_parallel.jsonl",
    REFERENCE, # Fallback to serial llama labels if GPT failed (better than nothing)
    "data/labeling_exports/gpt_auto_labels_final.jsonl"
)

# Llama
finalize_labels(
    "llama",
    REFERENCE,
    "data/labeling_exports/llama_auto_labels_parallel.jsonl",
    "data/labeling_exports/llama_auto_label_errors_parallel.jsonl",
    REFERENCE,
    "data/labeling_exports/llama_auto_labels_final.jsonl"
)

print("Final files created at data/labeling_exports/")
