import csv
from pathlib import Path

def main():
    input_file = Path("squad_a/audit_results/consolidated_labels.csv")
    output_file = Path("squad_a/audit_results/full_disagreements.csv")

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    disagreements = []
    for row in rows:
        human_majority = row["human_majority"]
        # The 4 LLM columns
        llm_preds = [
            row["GPT (P1)"],
            row["GPT (P2)"],
            row["Llama (P1)"],
            row["Llama (P2)"]
        ]
        
        # If all 4 disagreed with the human majority
        if all(pred != human_majority for pred in llm_preds):
            disagreements.append(row)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreements)

    print(f"Found {len(disagreements)} full-disagreement traces.")
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
