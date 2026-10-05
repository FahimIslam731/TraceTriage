# TraceTriage

**TraceTriage** is a research codebase for studying recovery-action routing for failed LLM-agent executions. Given a failed trace, it classifies the failure into a recovery action (`RETRY`, `REPLAN`, `RETRIEVE_MORE`, `TOOL_FIX`, `LOCAL_REPAIR`, or `ESCALATE`) and evaluates whether that action recovers the correct answer.

## Repository Map

| Path | Purpose |
|---|---|
| `squad_a/` | Human/LLM label audit, frozen train/dev/test splits, and label provenance. |
| `squad_b/` | Recovery-action classifiers: TF-IDF, embeddings, LLM baselines, input ablations. |
| `squad_c/` | Offline recovery-action simulation, policy comparison, and cost tracking. |
| `data/` | Generated data exports. The SQLite database is gitignored because it is large. |
| `build_sqlite.py` | Rebuilds the local SQLite trace database from source artifacts. |
| `auto_label.ipynb` | LLM-assisted labeling notebook used during dataset construction. |

Each squad directory has its own README with more specific setup and run instructions.

## Dataset and Label Provenance

There are two important counts:

- **1,212 total labeled traces**: the full released label set used for Squad B classification and Squad C policy analysis.
- **638 human-audited traces**: the subset labeled by six human annotators and used to validate the label taxonomy and automated label pipeline.

The 1,212 and 638 counts are not in conflict: the 638 traces are the manually audited subset of the larger labeled corpus. The human audit found strong agreement, with mean pairwise Cohen's kappa of **0.764**.

Key Squad A artifacts:

| File | Description |
|---|---|
| [`squad_a/AUDIT_REPORT.md`](squad_a/AUDIT_REPORT.md) | Six-annotator agreement analysis and LLM-vs-human audit summary. |
| [`squad_a/human_vs_llm_audit.py`](squad_a/human_vs_llm_audit.py) | Script that reproduces the human-vs-LLM audit tables. |
| [`squad_a/dataset_split/train.csv`](squad_a/dataset_split/train.csv) | Frozen training split with `trace_id` and `human_majority`. |
| [`squad_a/dataset_split/dev.csv`](squad_a/dataset_split/dev.csv) | Frozen development split with `trace_id` and `human_majority`. |
| [`squad_a/dataset_split/test.csv`](squad_a/dataset_split/test.csv) | Frozen test split with `trace_id` and `human_majority`. |
| [`data/labeling_exports/`](data/labeling_exports/) | Raw LLM auto-label JSONL inputs used during labeling. |

`LOCAL_REPAIR` labels are tied to CausalFlow-validated single-step repairs. CausalFlow supplies validated local repair evidence; TraceTriage uses that evidence as one routed recovery action, rather than replacing recovery mechanisms themselves.

## How to Run

Set up the Python environment and install the local dependencies used by the experiments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy scikit-learn openai
```

Optional dependencies:

```bash
python -m pip install xgboost datasets
brew install libomp  # macOS only, needed by xgboost wheels
```

API-backed tasks require keys in the environment:

```bash
export OPENROUTER_API_KEY="..."
export SERPER_API_KEY="..."  # only needed for RETRIEVE_MORE recovery simulation
```

Run the main experiment groups:

```bash
# Label/audit summaries
python squad_a/human_vs_llm_audit.py

# Classification baselines and input ablations
python -m squad_b.main --task tfidf --no-structured
python -m squad_b.main --task ablations
python -m squad_b.main --task embedding

# LLM classifier baselines
python -m squad_b.main --task llm-zero --model-preset gemini-flash-lite --json-output
python -m squad_b.main --task llm-few --model-preset gemini-flash-lite --k-per-class 5 --json-output

# Recovery simulation
python -m squad_c.run_recovery --pilot --domain GSM8K MBPP SealQA MedBrowseComp
python -m squad_c.analyze_policies --stage a
python -m squad_c.run_recovery --full --domain GSM8K MBPP SealQA MedBrowseComp
python -m squad_c.analyze_policies --stage b
```

## Reproducibility Notes

- Frozen split seed and classifier seed are `RANDOM_SEED = 42`.
- Squad B results are written to `squad_b/results/`, which is gitignored.
- Squad B LLM and embedding caches are written to `squad_b/cache/`, which is gitignored.
- Squad C recovery outputs are written to `squad_c/results/`, which is gitignored.
- Recovery simulation is offline: it applies one selected recovery action to each failed trace and re-runs task verifiers. It is not a live human-in-the-loop deployment evaluation.

## Terminology

- **Majority baseline**: always predicts the most common training action.
- **Domain-only baseline / domain policy**: predicts the modal action for the trace domain.
- **Trace triage**: routes a failed trace to a recovery action; it is a routing layer over recovery mechanisms such as retry, replan, retrieval, tool-fix, local repair, and escalation.
