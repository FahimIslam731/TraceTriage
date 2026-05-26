#!/usr/bin/env python3
import csv
import json
import sqlite3
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATED_CSV = PROJECT_ROOT / "squad_a" / "audit_results" / "consolidated_labels.csv"
FAILED_TRACES_JSONL = PROJECT_ROOT / "data" / "labeling_exports" / "failed_traces.jsonl"
DB_PATH = PROJECT_ROOT / "data" / "causal_runs.sqlite"
OUTPUT_CSV = PROJECT_ROOT / "squad_a" / "audit_results" / "all_1212_labels.csv"

def main():
    # 1. Map trace_num to trace_id using failed_traces.jsonl
    print("Loading failed_traces.jsonl to map trace_id...")
    with open(FAILED_TRACES_JSONL, "r", encoding="utf-8") as f:
        jsonl_traces = [json.loads(line) for line in f if line.strip()]
    
    # 2. Read consolidated_labels.csv (638 human audited traces)
    print("Reading 638 manually labeled traces from consolidated_labels.csv...")
    labeled_data = []
    with open(CONSOLIDATED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            trace_id = jsonl_traces[i]["trace_id"]
            labeled_data.append({
                "trace_id": trace_id,
                "human_majority": row["human_majority"]
            })
    
    print(f"Loaded {len(labeled_data)} manually labeled traces.")

    # 3. Query the 574 auto-labeled LOCAL_REPAIR traces from SQLite
    print("Querying 574 LOCAL_REPAIR traces from SQLite...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT t.trace_id 
        FROM traces t 
        JOIN triage_labels l ON t.trace_id = l.trace_id 
        WHERE t.is_failing_trace = 1 
          AND t.is_ablation = 0 
          AND l.is_local_repairable = 1
    """).fetchall()
    
    local_repair_data = []
    for row in rows:
        local_repair_data.append({
            "trace_id": row["trace_id"],
            "human_majority": "LOCAL_REPAIR"
        })
    conn.close()
    
    print(f"Loaded {len(local_repair_data)} LOCAL_REPAIR traces from SQLite.")
    
    # 4. Combine and write out to all_1212_labels.csv
    combined_data = labeled_data + local_repair_data
    print(f"Total combined traces: {len(combined_data)}")
    
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trace_id", "human_majority"])
        writer.writeheader()
        writer.writerows(combined_data)
        
    print(f"Successfully wrote 1212 traces to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
