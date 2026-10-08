# TraceTriage

**TraceTriage** is a research codebase for studying recovery-action routing for failed LLM-agent executions. Given a failed trace, it classifies the failure into a recovery action (`RETRY`, `REPLAN`, `RETRIEVE_MORE`, `TOOL_FIX`, `LOCAL_REPAIR`, or `ESCALATE`) and evaluates whether that action recovers the correct answer.

## Repository Map

| Path | Purpose |
|---|---|
| `squad_a/` | Human/LLM label audit, frozen train/dev/test splits, and label provenance. |
| `src/data_labelling/` | Reviewer-facing label provenance artifacts: audit report, audit script, and full 1,212-label CSV. |
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
| [`src/data_labelling/AUDIT_REPORT.md`](src/data_labelling/AUDIT_REPORT.md) | Reviewer-facing copy of the six-annotator agreement analysis and human-vs-LLM audit summary. |
| [`src/data_labelling/human_vs_llm_audit.py`](src/data_labelling/human_vs_llm_audit.py) | Reviewer-facing wrapper for the audit script that computes the human-vs-LLM agreement metrics. |
| [`src/data_labelling/all_1212_labels.csv`](src/data_labelling/all_1212_labels.csv) | Full 1,212-trace majority-vote label table. |
| [`squad_a/AUDIT_REPORT.md`](squad_a/AUDIT_REPORT.md) | Six-annotator agreement analysis and LLM-vs-human audit summary. |
| [`squad_a/human_vs_llm_audit.py`](squad_a/human_vs_llm_audit.py) | Script that reproduces the human-vs-LLM audit tables. |
| [`squad_a/audit_results/all_1212_labels.csv`](squad_a/audit_results/all_1212_labels.csv) | Full 1,212-trace majority-vote label table used to generate the frozen split files. |
| [`squad_a/dataset_split/train.csv`](squad_a/dataset_split/train.csv) | Frozen training split with `trace_id` and `human_majority`. |
| [`squad_a/dataset_split/dev.csv`](squad_a/dataset_split/dev.csv) | Frozen development split with `trace_id` and `human_majority`. |
| [`squad_a/dataset_split/test.csv`](squad_a/dataset_split/test.csv) | Frozen test split with `trace_id` and `human_majority`. |
| [`data/labeling_exports/`](data/labeling_exports/) | Raw LLM auto-label JSONL inputs used during labeling. |

`LOCAL_REPAIR` labels are tied to CausalFlow-validated single-step repairs. CausalFlow supplies validated local repair evidence; TraceTriage uses that evidence as one routed recovery action, rather than replacing recovery mechanisms themselves.

## Reproducibility Pointers

For reviewer navigation, the main released resources and code entry points are:

| Need | Path |
|---|---|
| Full majority-vote label table | [`src/data_labelling/all_1212_labels.csv`](src/data_labelling/all_1212_labels.csv) |
| Frozen train split | [`squad_a/dataset_split/train.csv`](squad_a/dataset_split/train.csv) |
| Frozen development split | [`squad_a/dataset_split/dev.csv`](squad_a/dataset_split/dev.csv) |
| Frozen test split | [`squad_a/dataset_split/test.csv`](squad_a/dataset_split/test.csv) |
| Human audit report | [`src/data_labelling/AUDIT_REPORT.md`](src/data_labelling/AUDIT_REPORT.md) |
| Human audit script | [`src/data_labelling/human_vs_llm_audit.py`](src/data_labelling/human_vs_llm_audit.py) |
| Raw automated label exports | [`data/labeling_exports/`](data/labeling_exports/) |
| Labeling notebook | [`auto_label.ipynb`](auto_label.ipynb) |
| Classification CLI | [`squad_b/main.py`](squad_b/main.py) |
| Classification code and settings | [`squad_b/README.md`](squad_b/README.md), [`squad_b/data_loader.py`](squad_b/data_loader.py), [`squad_b/tfidf_baseline.py`](squad_b/tfidf_baseline.py) |
| Recovery simulation CLI | [`squad_c/run_recovery.py`](squad_c/run_recovery.py) |
| Policy analysis | [`squad_c/analyze_policies.py`](squad_c/analyze_policies.py) |
| Recovery action implementations | [`squad_c/recovery_actions.py`](squad_c/recovery_actions.py) |
| Recovery cost assumptions | [`squad_c/costtable.md`](squad_c/costtable.md) |

`data/labeling_exports/` contains raw automated labeling/model export files. It is **not** the human-audit record location. Human-audit records, the audit script, and the full majority-vote label table are exposed under `src/data_labelling/`.

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
- Frozen splits are 968 train / 122 dev / 122 test examples, stratified by domain/action where possible.
- Squad B TF-IDF uses up to 10,000 unigram/bigram features; see `squad_b/README.md` and `squad_b/tfidf_baseline.py` for classifier settings.
- Squad B API-backed LLM runs use OpenRouter model presets from `squad_b/llm_classifier.py`; local TF-IDF, ablation, and dry-run commands do not require API keys.
- Squad C Stage A uses 100 traces per domain; Stage B uses the full 1,204-trace recovery simulation set.
- Squad C recovery model, search, and cost settings are documented in `squad_c/README.md`, `squad_c/costtable.md`, and `squad_c/cost_tracker.py`.
- Squad B results are written to `squad_b/results/`, which is gitignored.
- Squad B LLM and embedding caches are written to `squad_b/cache/`, which is gitignored.
- Squad C recovery outputs are written to `squad_c/results/`, which is gitignored.
- Recovery simulation is offline: it applies one selected recovery action to each failed trace and re-runs task verifiers. It is not a live human-in-the-loop deployment evaluation.

## Terminology

- **Majority baseline**: always predicts the most common training action.
- **Domain-only baseline / domain policy**: predicts the modal action for the trace domain.
- **Trace triage**: routes a failed trace to a recovery action; it is a routing layer over recovery mechanisms such as retry, replan, retrieval, tool-fix, local repair, and escalation.
