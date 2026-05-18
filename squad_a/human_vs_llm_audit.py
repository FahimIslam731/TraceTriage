#!/usr/bin/env python3
"""
Squad A — Human vs. LLM Label Audit
====================================
Compare 6 human annotators' labels against 4 LLM auto-label runs on 638 traces.

Outputs:
  1. Human majority-vote ground truth (with agreement stats)
  2. Inter-annotator agreement (Fleiss' kappa, pairwise Cohen's kappa)
  3. Per-action precision / recall / F1 for each LLM system vs. human majority
  4. Confusion matrices
  5. Overall accuracy per LLM system
  6. CSV export of consolidated results

Usage:
    python squad_a/human_vs_llm_audit.py
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "squad_a" / "Trace Triage Failed Traces Human_LLM Audit  - 638 Traces (LABEL THESE!).csv"
OUTPUT_DIR = PROJECT_ROOT / "squad_a" / "audit_results"

# Column mappings from the CSV
HUMAN_COLS = ["Kavin", "Fahim", "Yurun", "Daivik", "Gerard", "Gezheng"]
LLM_COLS = [
    "AI output (Kavin's OpenAI)",
    "AI output (Fahim's OpenAI)",
    "AI output (Kavin's Llama)",
    "AI output (Fahim's Llama)",
]
LLM_SHORT_NAMES = {
    "AI output (Kavin's OpenAI)": "GPT (Kavin)",
    "AI output (Fahim's OpenAI)": "GPT (Fahim)",
    "AI output (Kavin's Llama)": "Llama (Kavin)",
    "AI output (Fahim's Llama)": "Llama (Fahim)",
}

VALID_LABELS = {"RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX", "ESCALATE", "LOCAL_REPAIR"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_label(raw: str) -> str:
    """Clean up a single label: strip, upper, fix typos, handle slash-separated."""
    raw = raw.strip().upper()
    # Fix known typos
    raw = raw.replace("ESCELATE", "ESCALATE").replace("REPPLAN", "REPLAN").replace("REPAN", "REPLAN")
    # If slash-separated (e.g. "RETRY/REPLAN"), take the first one
    if "/" in raw:
        raw = raw.split("/")[0].strip()
    return raw if raw in VALID_LABELS else None


def majority_vote(labels: list[str]) -> tuple[str, float]:
    """Return (majority_label, agreement_fraction) from a list of labels."""
    valid = [l for l in labels if l is not None]
    if not valid:
        return None, 0.0
    counter = Counter(valid)
    top_label, top_count = counter.most_common(1)[0]
    return top_label, top_count / len(valid)


def fleiss_kappa(matrix):
    """
    Compute Fleiss' kappa for a matrix of shape (N_subjects, N_categories).
    matrix[i][j] = number of raters who assigned category j to subject i.
    """
    N = len(matrix)       # number of subjects
    n = sum(matrix[0])    # number of raters per subject
    k = len(matrix[0])    # number of categories

    if N == 0 or n <= 1:
        return 0.0

    # p_j = proportion of all assignments to category j
    p = [sum(matrix[i][j] for i in range(N)) / (N * n) for j in range(k)]

    # P_i = extent of agreement for subject i
    P = [
        (sum(matrix[i][j] ** 2 for j in range(k)) - n) / (n * (n - 1))
        for i in range(N)
    ]

    P_bar = sum(P) / N
    P_e = sum(pj ** 2 for pj in p)

    if P_e == 1.0:
        return 1.0

    return (P_bar - P_e) / (1.0 - P_e)


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa between two raters."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    cats = sorted(set(labels_a) | set(labels_b))
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    # Confusion matrix
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        conf[cat_to_idx[a]][cat_to_idx[b]] += 1

    p_o = sum(conf[i][i] for i in range(k)) / n
    p_e = sum(
        sum(conf[i][j] for j in range(k)) * sum(conf[j][i] for j in range(k))
        for i in range(k)
    ) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def precision_recall_f1(y_true: list, y_pred: list, label: str):
    """Compute precision, recall, F1 for a single class."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def confusion_matrix(y_true: list, y_pred: list, labels: list) -> list[list[int]]:
    """Build a confusion matrix. Rows = true, Cols = pred."""
    idx = {l: i for i, l in enumerate(labels)}
    mat = [[0] * len(labels) for _ in range(len(labels))]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    return mat


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load CSV ────────────────────────────────────────────────────────────
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} traces from CSV\n")

    # ── 2. Parse & normalize labels ────────────────────────────────────────────
    data = []  # list of dicts with trace_num, human_labels, llm_labels, human_majority
    for row in rows:
        trace_num = int(row["Trace #"])
        human_labels = [normalize_label(row[col]) for col in HUMAN_COLS]
        llm_labels = {col: normalize_label(row[col]) for col in LLM_COLS}

        maj_label, agreement = majority_vote(human_labels)

        data.append({
            "trace_num": trace_num,
            "human_labels": human_labels,
            "human_majority": maj_label,
            "human_agreement": agreement,
            "llm_labels": llm_labels,
        })

    # ── 3. Human inter-annotator agreement ─────────────────────────────────────
    print("=" * 70)
    print("SECTION 1: HUMAN INTER-ANNOTATOR AGREEMENT")
    print("=" * 70)

    # 3a. Fleiss' kappa
    all_cats = sorted(VALID_LABELS)
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}

    fleiss_matrix = []
    for d in data:
        row = [0] * len(all_cats)
        for lbl in d["human_labels"]:
            if lbl and lbl in cat_to_idx:
                row[cat_to_idx[lbl]] += 1
        fleiss_matrix.append(row)

    fk = fleiss_kappa(fleiss_matrix)
    print(f"\n  Fleiss' kappa (all 6 raters): {fk:.4f}")
    if fk < 0.20:
        fk_interp = "Slight"
    elif fk < 0.40:
        fk_interp = "Fair"
    elif fk < 0.60:
        fk_interp = "Moderate"
    elif fk < 0.80:
        fk_interp = "Substantial"
    else:
        fk_interp = "Almost Perfect"
    print(f"  Interpretation: {fk_interp} agreement")

    # 3b. Pairwise Cohen's kappa
    print(f"\n  Pairwise Cohen's kappa:")
    kappas = []
    for (i, name_a), (j, name_b) in combinations(enumerate(HUMAN_COLS), 2):
        labels_a = [d["human_labels"][i] for d in data if d["human_labels"][i] and d["human_labels"][j]]
        labels_b = [d["human_labels"][j] for d in data if d["human_labels"][i] and d["human_labels"][j]]
        if labels_a:
            ck = cohen_kappa(labels_a, labels_b)
            kappas.append(ck)
            print(f"    {name_a:8s} vs {name_b:8s}: {ck:.4f}")
    print(f"\n  Mean pairwise Cohen's kappa: {sum(kappas)/len(kappas):.4f}")

    # 3c. Distribution of human majority agreement
    agreement_bins = Counter()
    for d in data:
        frac = d["human_agreement"]
        if frac == 1.0:
            agreement_bins["6/6 (unanimous)"] += 1
        elif frac >= 5/6:
            agreement_bins["5/6"] += 1
        elif frac >= 4/6:
            agreement_bins["4/6"] += 1
        elif frac >= 3/6:
            agreement_bins["3/6"] += 1
        else:
            agreement_bins["<3/6 (no majority)"] += 1

    print(f"\n  Human agreement distribution:")
    for k in ["6/6 (unanimous)", "5/6", "4/6", "3/6", "<3/6 (no majority)"]:
        cnt = agreement_bins.get(k, 0)
        pct = cnt / len(data) * 100
        bar = "█" * int(pct / 2)
        print(f"    {k:20s}: {cnt:4d} ({pct:5.1f}%) {bar}")

    # 3d. Human majority label distribution
    print(f"\n  Human majority-vote label distribution:")
    maj_counter = Counter(d["human_majority"] for d in data if d["human_majority"])
    for label in sorted(VALID_LABELS):
        cnt = maj_counter.get(label, 0)
        pct = cnt / len(data) * 100
        print(f"    {label:15s}: {cnt:4d} ({pct:5.1f}%)")

    # ── 4. LLM vs Human comparison ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SECTION 2: LLM vs HUMAN MAJORITY-VOTE COMPARISON")
    print("=" * 70)

    # Only compare traces where human majority exists
    valid_data = [d for d in data if d["human_majority"] is not None]
    print(f"\n  Traces with clear human majority: {len(valid_data)} / {len(data)}")

    all_labels_sorted = sorted(VALID_LABELS)

    for llm_col in LLM_COLS:
        short = LLM_SHORT_NAMES[llm_col]
        print(f"\n  {'─' * 60}")
        print(f"  LLM System: {short}")
        print(f"  {'─' * 60}")

        y_true = []
        y_pred = []
        for d in valid_data:
            pred = d["llm_labels"].get(llm_col)
            if pred is not None and d["human_majority"] is not None:
                y_true.append(d["human_majority"])
                y_pred.append(pred)

        if not y_true:
            print("    No valid predictions to compare.")
            continue

        # Overall accuracy
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        print(f"\n    Overall accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"    Total compared:  {len(y_true)}")

        # Per-class metrics
        print(f"\n    {'Action':<16s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Support':>8s}")
        print(f"    {'─'*62}")

        weighted_f1_sum = 0.0
        weighted_prec_sum = 0.0
        weighted_rec_sum = 0.0
        total_support = 0

        for label in all_labels_sorted:
            p, r, f1, tp, fp, fn = precision_recall_f1(y_true, y_pred, label)
            support = tp + fn
            print(f"    {label:<16s} {p:6.3f} {r:6.3f} {f1:6.3f} {tp:5d} {fp:5d} {fn:5d} {support:8d}")
            weighted_f1_sum += f1 * support
            weighted_prec_sum += p * support
            weighted_rec_sum += r * support
            total_support += support

        if total_support > 0:
            print(f"    {'─'*62}")
            wp = weighted_prec_sum / total_support
            wr = weighted_rec_sum / total_support
            wf = weighted_f1_sum / total_support
            print(f"    {'Weighted Avg':<16s} {wp:6.3f} {wr:6.3f} {wf:6.3f} {'':>5s} {'':>5s} {'':>5s} {total_support:8d}")

        # Cohen's kappa: LLM vs human majority
        ck_llm = cohen_kappa(y_true, y_pred)
        print(f"\n    Cohen's kappa vs human majority: {ck_llm:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, all_labels_sorted)
        print(f"\n    Confusion Matrix (rows=human, cols=LLM):")
        header = "    " + f"{'':16s}" + "".join(f"{l[:8]:>9s}" for l in all_labels_sorted)
        print(header)
        for i, row_label in enumerate(all_labels_sorted):
            row_str = "".join(f"{cm[i][j]:9d}" for j in range(len(all_labels_sorted)))
            print(f"    {row_label:<16s}{row_str}")

    # ── 5. Merged LLM comparison (GPT consensus, Llama consensus) ──────────
    print(f"\n{'=' * 70}")
    print("SECTION 3: MERGED LLM SYSTEMS (majority across runs)")
    print("=" * 70)

    for group_name, group_cols in [
        ("GPT (merged)", ["AI output (Kavin's OpenAI)", "AI output (Fahim's OpenAI)"]),
        ("Llama (merged)", ["AI output (Kavin's Llama)", "AI output (Fahim's Llama)"]),
        ("All LLMs (merged)", LLM_COLS),
    ]:
        print(f"\n  {'─' * 60}")
        print(f"  {group_name}")
        print(f"  {'─' * 60}")

        y_true = []
        y_pred = []
        for d in valid_data:
            preds = [d["llm_labels"][c] for c in group_cols if d["llm_labels"].get(c)]
            if not preds:
                continue
            llm_maj, _ = majority_vote(preds)
            if llm_maj and d["human_majority"]:
                y_true.append(d["human_majority"])
                y_pred.append(llm_maj)

        if not y_true:
            print("    No valid predictions.")
            continue

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        print(f"\n    Overall accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"    Total compared:  {len(y_true)}")

        print(f"\n    {'Action':<16s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
        print(f"    {'─'*42}")

        weighted_f1_sum = 0.0
        total_support = 0
        for label in all_labels_sorted:
            p, r, f1, tp, fp, fn = precision_recall_f1(y_true, y_pred, label)
            support = tp + fn
            print(f"    {label:<16s} {p:6.3f} {r:6.3f} {f1:6.3f} {support:8d}")
            weighted_f1_sum += f1 * support
            total_support += support

        if total_support > 0:
            wf = weighted_f1_sum / total_support
            print(f"    {'─'*42}")
            print(f"    {'Weighted Avg':<16s} {'':>6s} {'':>6s} {wf:6.3f} {total_support:8d}")

        ck = cohen_kappa(y_true, y_pred)
        print(f"\n    Cohen's kappa vs human majority: {ck:.4f}")

    # ── 6. Export consolidated CSV ─────────────────────────────────────────────
    export_path = OUTPUT_DIR / "consolidated_labels.csv"
    with open(export_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "trace_num", "human_majority", "human_agreement",
            *[f"human_{name}" for name in HUMAN_COLS],
            *[LLM_SHORT_NAMES[c] for c in LLM_COLS],
            "gpt_majority", "llama_majority", "all_llm_majority",
        ]
        writer.writerow(header)

        for d in data:
            gpt_preds = [d["llm_labels"][c] for c in LLM_COLS[:2] if d["llm_labels"].get(c)]
            llama_preds = [d["llm_labels"][c] for c in LLM_COLS[2:] if d["llm_labels"].get(c)]
            all_preds = [d["llm_labels"][c] for c in LLM_COLS if d["llm_labels"].get(c)]

            gpt_maj = majority_vote(gpt_preds)[0] if gpt_preds else ""
            llama_maj = majority_vote(llama_preds)[0] if llama_preds else ""
            all_maj = majority_vote(all_preds)[0] if all_preds else ""

            writer.writerow([
                d["trace_num"],
                d["human_majority"] or "",
                f"{d['human_agreement']:.2f}",
                *[l or "" for l in d["human_labels"]],
                *[d["llm_labels"].get(c, "") or "" for c in LLM_COLS],
                gpt_maj, llama_maj, all_maj,
            ])

    print(f"\n{'=' * 70}")
    print(f"Consolidated CSV exported to: {export_path}")

    # ── 7. Disagreement analysis ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SECTION 4: NOTABLE DISAGREEMENTS")
    print("=" * 70)

    # Traces where ALL 4 LLMs disagree with human majority
    llm_disagree_all = []
    for d in valid_data:
        all_wrong = True
        for col in LLM_COLS:
            pred = d["llm_labels"].get(col)
            if pred == d["human_majority"]:
                all_wrong = False
                break
        if all_wrong:
            llm_disagree_all.append(d)

    print(f"\n  Traces where ALL 4 LLMs disagree with human majority: {len(llm_disagree_all)}")
    if llm_disagree_all:
        print(f"\n    {'Trace':>6s}  {'Human':>15s}  {'GPT-K':>15s}  {'GPT-F':>15s}  {'Llm-K':>15s}  {'Llm-F':>15s}")
        for d in llm_disagree_all[:20]:  # show first 20
            preds = [d["llm_labels"].get(c, "???") or "???" for c in LLM_COLS]
            print(f"    {d['trace_num']:6d}  {d['human_majority']:>15s}  {preds[0]:>15s}  {preds[1]:>15s}  {preds[2]:>15s}  {preds[3]:>15s}")
        if len(llm_disagree_all) > 20:
            print(f"    ... and {len(llm_disagree_all) - 20} more")

    # LOCAL_REPAIR analysis (LLMs don't have this label)
    lr_traces = [d for d in valid_data if d["human_majority"] == "LOCAL_REPAIR"]
    print(f"\n  LOCAL_REPAIR traces (human majority): {len(lr_traces)}")
    if lr_traces:
        print(f"  LLM predictions on these traces:")
        lr_pred_counter = Counter()
        for d in lr_traces:
            for col in LLM_COLS:
                pred = d["llm_labels"].get(col)
                if pred:
                    lr_pred_counter[pred] += 1
        for lbl, cnt in lr_pred_counter.most_common():
            print(f"    {lbl}: {cnt}")

    print(f"\n{'=' * 70}")
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
