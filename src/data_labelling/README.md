# TraceTriage Data Labeling Artifacts

This directory is the reviewer-facing entry point for the human-label records used in the TraceTriage paper.

| File | Description |
|---|---|
| `AUDIT_REPORT.md` | Six-annotator agreement analysis and human-vs-LLM audit summary for the 638 human-audited traces. |
| `human_vs_llm_audit.py` | Convenience wrapper for the audit script that computes the human-vs-LLM agreement metrics. |
| `all_1212_labels.csv` | Full majority-vote label table for the 1,212 failed traces, with columns `trace_id` and `human_majority`. |

The canonical implementation files remain in `squad_a/`. This folder exists so the label provenance artifacts are easy to locate from the paper and rebuttal.
