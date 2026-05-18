# Squad A: Human vs. LLM Label Audit Report

**Date:** May 17, 2026  
**Team:** Kavin, Fahim, Yurun, Daivik, Gerard, Gezheng  
**Traces audited:** 638  
**Script:** [`human_vs_llm_audit.py`](./human_vs_llm_audit.py)

---

## 1. Overview

### What We Did
Six human annotators independently labeled 638 failed LLM agent traces with a recovery action. We then compared those human labels against 4 automated LLM labeling runs (2 GPT, 2 Llama) to determine whether LLM auto-labels can be trusted as ground truth.

### Why It Matters
Labeling traces by hand doesn't scale. If LLM auto-labels are reliable, we can label thousands more traces automatically. This audit quantifies exactly how reliable they are — and where they fail.

### Label Set

| Action | Description |
|--------|-------------|
| `RETRY` | Re-attempt the same operation without changes |
| `REPLAN` | Rethink the overall approach/strategy |
| `RETRIEVE_MORE` | Gather additional evidence or context |
| `TOOL_FIX` | Fix or swap the tool being used |
| `LOCAL_REPAIR` | Make a small, targeted patch to the current step |
| `ESCALATE` | Escalate to a higher-level system or human |

> **Note:** The LLM auto-labeling prompt only included 5 actions (no `LOCAL_REPAIR`). The 6th label emerged organically from human annotation, revealing a gap in the original taxonomy.

---

## 2. Human Inter-Annotator Agreement

Before trusting the human labels, we measured how well the 6 annotators agreed with each other.

### Fleiss' Kappa (all 6 raters)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Fleiss' κ** | **0.7628** | Substantial agreement |

Fleiss' kappa measures agreement across all raters simultaneously, corrected for chance. A value of 0.76 means humans substantially agree on the correct label — our majority-vote ground truth is reliable.

### Pairwise Cohen's Kappa

Cohen's kappa measures agreement between every pair of annotators:

| Pair | κ |
|------|---|
| Kavin vs Fahim | 0.7824 |
| Kavin vs Yurun | 0.8151 |
| Kavin vs Daivik | 0.8150 |
| Kavin vs Gerard | 0.6734 |
| Kavin vs Gezheng | 0.8490 |
| Fahim vs Yurun | 0.8062 |
| Fahim vs Daivik | 0.8063 |
| Fahim vs Gerard | 0.6322 |
| Fahim vs Gezheng | 0.8220 |
| Yurun vs Daivik | 0.7956 |
| Yurun vs Gerard | 0.6463 |
| Yurun vs Gezheng | 0.8273 |
| Daivik vs Gerard | 0.6500 |
| Daivik vs Gezheng | 0.8771 |
| Gerard vs Gezheng | 0.6622 |
| **Mean** | **0.7640** |

Most pairs show substantial-to-almost-perfect agreement (0.63–0.88).

### Agreement Distribution

How many of the 6 annotators agreed on each trace:

| Agreement Level | Count | % | |
|-----------------|-------|---|-|
| 6/6 (unanimous) | 409 | 64.1% | ████████████████████████████████ |
| 5/6 | 127 | 19.9% | █████████ |
| 4/6 | 55 | 8.6% | ████ |
| 3/6 | 43 | 6.7% | ███ |
| <3/6 (no majority) | 4 | 0.6% | |

**84% of traces had ≥5/6 agreement.** Only 4 traces had no clear majority.

### Human Majority-Vote Label Distribution

| Action | Count | % |
|--------|-------|---|
| REPLAN | 275 | 43.1% |
| RETRIEVE_MORE | 202 | 31.7% |
| TOOL_FIX | 120 | 18.8% |
| RETRY | 27 | 4.2% |
| LOCAL_REPAIR | 13 | 2.0% |
| ESCALATE | 1 | 0.2% |

> **Class imbalance warning:** REPLAN dominates at 43%. ESCALATE and LOCAL_REPAIR are extremely rare. This will affect downstream classifier training (Squad B should use stratified sampling and class weighting).

---

## 3. LLM vs. Human Comparison

### Methodology
- **Ground truth:** Human majority-vote label per trace
- **Predictions:** Each of the 4 LLM auto-label runs
- **Metrics:** Accuracy, Cohen's κ, per-class precision/recall/F1
- **Merged systems:** Majority vote across GPT runs, Llama runs, and all 4

### Overall Accuracy

| LLM System | Accuracy | Cohen's κ |
|------------|----------|-----------|
| GPT (Kavin) | 59.1% | 0.418 |
| GPT (Fahim) | 57.8% | 0.405 |
| Llama (Kavin) | 61.0% | 0.422 |
| Llama (Fahim) | 62.2% | 0.422 |
| **GPT merged** | **59.1%** | **0.418** |
| **Llama merged** | **62.2%** | **0.437** |
| **All 4 LLMs merged** | **63.8%** | **0.470** |

For context: random guessing with 6 labels would give ~17% accuracy. The LLMs are well above chance but far below the human agreement level (κ=0.76).

### Per-Action Precision / Recall / F1 (All 4 LLMs Merged)

| Action | Precision | Recall | F1 | Support | Interpretation |
|--------|-----------|--------|----|---------|----------------|
| RETRIEVE_MORE | 0.614 | 0.881 | 0.724 | 202 | Best class — LLMs catch most cases but over-predict |
| REPLAN | 0.713 | 0.560 | 0.627 | 275 | High precision, but misses 44% of actual REPLAN cases |
| TOOL_FIX | 0.702 | 0.492 | 0.578 | 120 | Misses half the real TOOL_FIX cases |
| RETRY | 0.364 | 0.593 | 0.451 | 27 | Low precision — many false positives |
| LOCAL_REPAIR | 0.000 | 0.000 | 0.000 | 13 | Never predicted (not in LLM action set) |
| ESCALATE | 0.000 | 0.000 | 0.000 | 1 | Only 1 example — not meaningful |
| **Weighted Avg** | — | — | **0.627** | **638** | |

**Key metric definitions:**
- **Precision:** Of all times the LLM predicted label X, what % was correct? (Low precision = many false alarms)
- **Recall:** Of all actual label X cases, what % did the LLM catch? (Low recall = missing real cases)
- **F1:** Harmonic mean of precision and recall (balances both)

### Confusion Matrix (All 4 LLMs Merged, rows=human, cols=LLM)

|  | ESCALATE | LOCAL_R | REPLAN | RETRIEVE | RETRY | TOOL_FIX |
|--|----------|---------|--------|----------|-------|----------|
| **ESCALATE** | 0 | 0 | 0 | 1 | 0 | 0 |
| **LOCAL_REPAIR** | 0 | 0 | 9 | 0 | 3 | 1 |
| **REPLAN** | 3 | 0 | 163 | 89 | 16 | 4 |
| **RETRIEVE_MORE** | 2 | 0 | 8 | 174 | 15 | 3 |
| **RETRY** | 2 | 0 | 9 | 4 | 12 | 0 |
| **TOOL_FIX** | 0 | 0 | 53 | 23 | 4 | 40 |

**Biggest error pattern:** TOOL_FIX → REPLAN (53 traces). LLMs frequently mislabel tool-level fixes as strategy-level replanning.

---

## 4. Key Findings

### Finding 1: Human labels are reliable ground truth
With Fleiss' κ = 0.76 and 84% of traces at ≥5/6 agreement, the majority-vote labels are trustworthy for training and evaluation.

### Finding 2: LLM auto-labels are moderately accurate but not a replacement
At 64% accuracy (κ=0.47), LLMs are better than random but substantially worse than human agreement. They cannot be used as sole ground truth.

### Finding 3: TOOL_FIX vs. REPLAN is the primary confusion
The LLMs struggle to distinguish "fix the tool" from "rethink the strategy." This is a semantic distinction that may require better prompting or fine-tuning.

### Finding 4: Human annotation discovered a missing taxonomy category
`LOCAL_REPAIR` emerged from human labeling despite not being in the original 5-action schema. This demonstrates the value of human-in-the-loop annotation for surfacing categories that pre-defined schemas miss. It affects 2% of traces (13/638).

### Finding 5: 133 traces are universally hard for LLMs
All 4 LLM systems disagreed with human majority on 133 traces (20.8%). These represent the hardest cases and are worth investigating for prompt improvement.

---

## 5. Implications for Next Steps

| Action Item | Owner |
|-------------|-------|
| Freeze train/dev/test splits using human majority labels | Squad A |
| Use consolidated CSV as training data | Squad B |
| Address class imbalance (stratified sampling, class weights) | Squad B |
| Investigate 133 full-disagreement traces for prompt improvement | Squad A/C |

---

## 6. Files Produced

| File | Description |
|------|-------------|
| [`human_vs_llm_audit.py`](./human_vs_llm_audit.py) | Analysis script (run to reproduce all results) |
| [`audit_results/consolidated_labels.csv`](./audit_results/consolidated_labels.csv) | All 638 traces with human labels, LLM labels, majority votes, agreement scores |

---

## Appendix: Kappa Interpretation Scale (Landis & Koch, 1977)

| κ Range | Interpretation |
|---------|----------------|
| < 0.00 | Poor |
| 0.00–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost Perfect |
