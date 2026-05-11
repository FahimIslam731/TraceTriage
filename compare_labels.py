import json

def compare_files(old_path, new_path, label):
    print(f"Comparing {label}...")
    with open(old_path, "r") as f:
        old_data = {json.loads(line)["trace_id"]: json.loads(line)["action"] for line in f if line.strip()}
    with open(new_path, "r") as f:
        new_data = {json.loads(line)["trace_id"]: json.loads(line)["action"] for line in f if line.strip()}
    
    total = 0
    match = 0
    mismatch = []
    
    for tid, old_action in old_data.items():
        if tid in new_data:
            total += 1
            if old_action == new_data[tid]:
                match += 1
            else:
                mismatch.append((tid, old_action, new_data[tid]))
    
    print(f"  Total compared: {total}")
    print(f"  Matches: {match} ({match/total:.1%})")
    print(f"  Mismatches: {total - match}")
    if mismatch:
        print(f"  Sample mismatch (ID, Old, New): {mismatch[0]}")

compare_files("data/labeling_exports/gpt_auto_labels.jsonl", "data/labeling_exports/gpt_auto_labels_final.jsonl", "GPT")
compare_files("data/labeling_exports/llama_auto_labels.jsonl", "data/labeling_exports/llama_auto_labels_final.jsonl", "Llama")
