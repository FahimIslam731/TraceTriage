# Squad A — Labeling, Human Audit, and Frozen Splits

Squad A owns label provenance for TraceTriage. This folder contains both the human-audit materials and the frozen train/dev/test splits used by downstream experiments.

## Label Counts

Two counts appear throughout the project:

- **1,212 total labeled traces**: the full label set used for classifier training/evaluation and recovery-policy analysis.
- **638 human-audited traces**: the subset independently labeled by six human annotators to validate the taxonomy and automated labeling pipeline.

The 638-trace audit is not a separate dataset for Squad B; it is the validation subset that supports using the larger 1,212-trace label set.

## Key Files

| File | Description |
|---|---|
| `AUDIT_REPORT.md` | Human-vs-LLM label audit summary, including pairwise Cohen's kappa and confusion patterns. |
| `human_vs_llm_audit.py` | Reproduces the human-vs-LLM audit analysis. |
| `audit_results/consolidated_labels.csv` | Human-audited subset with human labels, LLM labels, agreement scores, and majority votes. |
| `audit_results/all_1212_labels.csv` | Full 1,212-trace label table used by downstream policy analysis. |
| `audit_results/full_disagreements.csv` | Cases where all LLM labeling runs disagreed with the human majority. |
| `train.csv`, `dev.csv`, `test.csv` | Frozen splits consumed by Squad B. Each row has `trace_id` and `human_majority`. |
| `stratified_split.py` | Builds frozen splits stratified by domain/action where possible. |
| `build_1212_dataset.py` | Builds the full labeled dataset table. |

## Human Audit Summary

The human audit used six annotators and majority vote as the gold label. Mean pairwise Cohen's kappa is **0.764**, with most traces receiving high agreement. The audit also surfaced `LOCAL_REPAIR` as a required sixth action: the original LLM pre-label prompt did not include that class, but human annotation identified CausalFlow-validated local repair as a distinct recovery action.

## Two-Stage Labeling Pipeline

Labeling was done in two stages. First, LLM pre-labelers produced scalable recovery-action proposals and rationales for failed traces. Second, the 638-trace human audit checked those labels with six independent annotators per trace and majority vote. The audit validates the taxonomy and label process; the resulting frozen files expose `human_majority` labels to downstream code.

## Downstream Use

- Squad B loads `train.csv`, `dev.csv`, and `test.csv` through `squad_b/data_loader.py`.
- Squad C reads `audit_results/all_1212_labels.csv` when evaluating policies such as `trace_triage` and `domain_policy`.
- `LOCAL_REPAIR` labels are tied to CausalFlow-validated local repairs; the label means a local repair exists and is the routed recovery action for that trace.
