# TraceTriage

**TraceTriage** is a failure-recovery framework for LLM agent traces. Given a failed trace, it classifies the failure into a recovery action (RETRY, REPLAN, RETRIEVE_MORE, TOOL_FIX, LOCAL_REPAIR, ESCALATE) and executes that action to recover the correct answer.

---

## Repository Structure

```
TraceTriage/
├── squad_a/              # Human labeling, inter-annotator agreement, dataset splits
├── squad_b/              # Triage classifiers (TF-IDF, embedding, LLM few-shot)
├── squad_c/              # Recovery simulation & policy comparison (Experiments 4 & 5)
├── data/
│   └── labeling_exports/ # Raw LLM auto-label outputs (GPT & Llama JSONL)
└── scratch/              # One-off analysis scripts
```

Each squad has its own `README.md` with detailed instructions.

---

## Dataset & Human Audit Records

The full corpus and human audit records live in [`squad_a/`](squad_a/):

| File | Description |
|------|-------------|
| [`AUDIT_REPORT.md`](squad_a/AUDIT_REPORT.md) | Six-annotator agreement analysis, including the full pairwise Cohen's κ table (mean κ = 0.764) |
| [`human_vs_llm_audit.py`](squad_a/human_vs_llm_audit.py) | Validation and audit script — computes agreement metrics across human and LLM labelers |
| [`audit_results/all_1212_labels.csv`](squad_a/audit_results/all_1212_labels.csv) | Complete set of 1,212 majority-vote labels across all domains |

> **Note:** These files are in the `squad_a/` labeling module, not in `data/labeling_exports/`. The `labeling_exports/` directory contains only the raw LLM auto-label JSONL outputs used as inputs to the audit.

> **Note:** The 1,212 and 638 counts are not in conflict. The full labeled dataset contains **1,212 traces**. Of these, **638 were routed to manual human annotation** (exported in `data/labeling_exports/failed_traces.md`); the remainder were labeled via the automated pipeline described in Section 4.2 of the paper.

---

## Getting Started

Each squad directory contains its own `README.md` with prerequisites, setup, and run instructions:

- [`squad_a/README`](squad_a/) — Human labeling & inter-annotator agreement
- [`squad_b/README`](squad_b/README.md) — Triage classifiers (TF-IDF, embedding, LLM few-shot)
- [`squad_c/README`](squad_c/README.md) — Recovery simulation & policy comparison
