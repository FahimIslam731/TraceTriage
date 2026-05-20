# Squad B — Trace Triage Classification Pipeline

## What This Does

This folder contains the classification infrastructure for predicting **recovery actions** on failing LLM agent traces. Given a failed trace, the pipeline classifies it into one of six actions:

| Action | Meaning |
|--------|---------|
| `LOCAL_REPAIR` | CausalFlow found a localized counterfactual repair |
| `RETRIEVE_MORE` | Agent lacked information — needs more data/searches |
| `REPLAN` | Strategy was wrong — needs a different approach entirely |
| `TOOL_FIX` | A tool call failed — fix the arguments or handle the error |
| `RETRY` | Minor mistake (e.g., arithmetic) — re-running may work |
| `ESCALATE` | Beyond the agent's capability — flag for human review |

Ground truth labels and train/dev/test partitions come from Squad A's frozen `train.csv`, `dev.csv`, and `test.csv` files using the `human_majority` label. If those files are unavailable, the loader falls back to CausalFlow/LLM labels and a deterministic random split.

---

## File Overview

```
squad_b/
├── main.py               # CLI entry point — run experiments from here
├── data_loader.py         # Loads data, builds features, handles train/test splits
├── tfidf_baseline.py      # TF-IDF + Logistic Regression / Random Forest baselines
├── ablations.py           # Experiment 2 input-variant ablations
├── cross_domain.py        # Experiment 3 leave-one-domain-out evaluation
├── embedding_pipeline.py  # OpenAI embedding-based classifiers (SVM, LogReg, XGBoost)
├── llm_classifier.py      # Zero-shot and few-shot LLM classification
├── llm_calibration.py     # Calibration metrics from cached LLM confidence outputs
├── evaluator.py           # Shared metrics, confusion matrix, result export
├── cache/                 # Auto-generated: cached embeddings & LLM predictions
└── results/               # Auto-generated: JSON result files per experiment
```

### `data_loader.py`
- Loads traces and CausalFlow triage metadata from `data/causal_runs.sqlite`
- Falls back to `data/labeling_exports/failed_traces.jsonl` if SQLite is unavailable
- Applies Squad A's frozen splits from `squad_a/train.csv`, `squad_a/dev.csv`, and `squad_a/test.csv`
- Uses Squad A's `human_majority` column as the gold label for frozen-split experiments
- Falls back to assigning `LOCAL_REPAIR` from CausalFlow metadata and joining GPT/Llama labels for the remaining actions when frozen splits are unavailable
- **Text features**: flattens each trace into a structured text block (domain, problem, steps, tool outputs, final answer)
- **Input variants**: full trace, final answer only, verifier feedback only, trace stats only, and causal-step neighborhood
- **Structured features**: extracts numerical features from trace data (step counts, tool failure rates, domain one-hots) and optionally from `trace_metrics` in SQLite
- Provides `build_dataset()` which returns everything needed for any classifier
- Provides frozen split indices for 968 train, 122 dev, and 122 test examples

### `tfidf_baseline.py`
- Fits TF-IDF (unigrams + bigrams, max 10k features) on flattened trace text
- Optionally stacks structured features alongside TF-IDF
- Trains **Logistic Regression** and **Random Forest** (both class-weight balanced)
- Also reports **majority** and **domain-only** baselines for reference
- Saves per-domain metrics and per-domain confusion matrices in result JSON

### `ablations.py`
- Runs the required input-variant ablations over TF-IDF classifiers
- Saves one result file per variant plus `results/input_ablation_summary_results.json`

### `cross_domain.py`
- Runs Experiment 3 leave-one-domain-out evaluation
- Trains on all domains except one held-out domain, then evaluates on that held-out domain
- Reports global majority, oracle domain-mode reference, TF-IDF + Logistic Regression, and TF-IDF + Random Forest
- Optionally runs LLM few-shot on each held-out domain with `--llm-cross-domain`
- Saves per-action transfer success and transfer gap versus in-domain TF-IDF results
- Skips very small domains by default; control this with `--min-domain-samples`

### `embedding_pipeline.py`
- Embeds traces via OpenAI `text-embedding-3-small` through OpenRouter
- Caches embeddings to `cache/embeddings.npy` so you only pay once
- Trains **SVM (RBF)**, **Logistic Regression**, and **XGBoost** on the embedding vectors
- XGBoost requires the optional `xgboost` package

### `llm_classifier.py`
- **Zero-shot**: sends each test trace to an LLM with action definitions, asks it to pick one
- **Few-shot**: same but includes k labeled examples per class from the training set
- Supports model presets for GPT-5 Mini and Gemini Flash-Lite via `--model-preset`
- Supports JSON action/confidence output via `--json-output` for calibration analysis
- Caches predictions per trace to avoid re-running (saves cost)
- Rate-limited with 0.5s delay between calls
- Default model: `google/gemini-3-flash-preview` (configurable via `--model`)

### `llm_calibration.py`
- Reads cached LLM JSON-confidence predictions without calling any API
- Computes expected calibration error (ECE) bins and saves `results/llm_calibration_results.json`
- Saves a reliability diagram SVG in `results/`

### `evaluator.py`
- `evaluate()` → accuracy, macro F1, weighted F1, per-class precision/recall/F1/support
- `confusion_matrix_str()` → formatted confusion matrix for console output
- `save_results()` → writes JSON to `results/<method>_results.json`
- All methods use this, so results are directly comparable

---

## How to Run

```bash
# From the project root:

# TF-IDF baselines (no API key needed)
python -m squad_b.main --task tfidf

# TF-IDF without structured features
python -m squad_b.main --task tfidf --no-structured

# One specific input ablation
python -m squad_b.main --task tfidf --input-variant causal_neighborhood --no-structured

# All input ablations (no API key needed)
python -m squad_b.main --task ablations

# Experiment 3: leave-one-domain-out cross-domain evaluation
python -m squad_b.main --task cross-domain

# Cross-domain without structured features
python -m squad_b.main --task cross-domain --no-structured

# Embedding pipeline (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY="your-key-here"
python -m squad_b.main --task embedding

# Embedding + XGBoost needs xgboost installed; the embedding task runs it automatically if available
python -m pip install xgboost
python -m squad_b.main --task embedding

# LLM zero-shot/few-shot examples (requires OPENROUTER_API_KEY)
python -m squad_b.main --task llm-zero --model-preset gpt5-mini --json-output
python -m squad_b.main --task llm-few --model-preset gpt5-mini --k-per-class 5 --json-output
python -m squad_b.main --task llm-zero --model-preset gemini-flash-lite --json-output
python -m squad_b.main --task llm-few --model-preset gemini-flash-lite --k-per-class 5 --json-output

# Small LLM smoke tests to control token spend
python -m squad_b.main --task llm-zero --model-preset gpt5-mini --limit 15 --json-output --max-tokens 512
python -m squad_b.main --task llm-few --model-preset gpt5-mini --k-per-class 5 --limit 15 --json-output --max-tokens 512

# Experiment 3 with the required LLM few-shot model added to each held-out domain
python -m squad_b.main --task cross-domain --llm-cross-domain --model-preset gpt5-mini --k-per-class 5 --json-output
python -m squad_b.main --task cross-domain --llm-cross-domain --model-preset gemini-flash-lite --k-per-class 5 --json-output

# Limited cross-domain LLM smoke test before spending on the full held-out domains
python -m squad_b.main --task cross-domain --llm-cross-domain --model-preset gpt5-mini --k-per-class 5 --limit 15 --json-output

# Calibration from cached LLM JSON-confidence predictions (no API call)
python -m squad_b.main --task llm-calibration

# Run local/no-API tasks
python -m squad_b.main --task all-local

# Run everything, including API-dependent embedding/LLM tasks
python -m squad_b.main --task all
```

Results are saved automatically to `squad_b/results/`.

---

## What You Need to Know Before Changing Things

Several values are **hardcoded to match the current dataset**. If the data changes, you'll need to update these — all in `data_loader.py` unless noted:

### Must update if the label taxonomy changes
- `TARGET_CLASSES` (line 31) — the six action labels
- `ACTION_DEFINITIONS` in `llm_classifier.py` (line 35) — LLM prompt text must match

### Must update if the trace schema changes
- `flatten_trace_to_text()` — assumes steps have `step_index`, `step_type`, `tool_name`, `text`, `tool_output_json`
- `extract_structured_features()` — assumes fields like `is_local_repairable`, `num_successful_repair_steps`
- `load_labeled_traces()` — assumes labels have fields `trace_id`, `action`, `rationale`, `confidence`

### Must update if domains/tools/step-types change
- **Domains** (line 153): `["GSM8K", "MBPP", "MedBrowseComp", "SealQA", "BrowseComp"]` — hardcoded one-hot encoding
- **Tools** (line 32): `TOOL_NAMES` list — new tools won't get features
- **Step types** (line 162): `["reasoning", "tool_call", "tool_response", ...]` — new types are silently ignored

### Must update if data files move or get renamed
- `DB_PATH`, `GPT_LABELS_PATH`, `LLAMA_LABELS_PATH`, `FAILED_TRACES_PATH`, and `FROZEN_SPLIT_PATHS`

### Must update if SQLite schema changes
- The SQL query in `extract_structured_features_from_sqlite()` (line 177) assumes specific column names in `trace_metrics`

---

## Current Results (already generated)

| Method | File |
|--------|------|
| TF-IDF + Logistic Regression | `results/tfidf_logreg_results.json` |
| TF-IDF + Random Forest | `results/tfidf_rf_results.json` |
| Majority Baseline | `results/majority_baseline_results.json` |
| Domain-Only Baseline | `results/domain_only_baseline_results.json` |
| Input Ablation Summary | `results/input_ablation_summary_results.json` |
| Cross-Domain Experiment 3 | `results/cross_domain_results.json` |
| LLM Calibration | `results/llm_calibration_results.json` |

Embedding and LLM results will appear after running those tasks with an API key.

---

## Dependencies

```
numpy, scikit-learn, scipy    # for local ML baselines
openai                        # for embeddings & LLM calls (via OpenRouter)
xgboost                       # optional, for embedding + XGBoost
```

All imports are standard except `openai` and optional `xgboost`. TF-IDF, ablations, and cross-domain baselines run without any API keys.
