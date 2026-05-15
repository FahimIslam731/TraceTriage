# Squad B — Trace Triage Classification Pipeline

## What This Does

This folder contains the classification infrastructure for predicting **recovery actions** on failing LLM agent traces. Given a failed trace, the pipeline classifies it into one of five actions:

| Action | Meaning |
|--------|---------|
| `RETRIEVE_MORE` | Agent lacked information — needs more data/searches |
| `REPLAN` | Strategy was wrong — needs a different approach entirely |
| `TOOL_FIX` | A tool call failed — fix the arguments or handle the error |
| `RETRY` | Minor mistake (e.g., arithmetic) — re-running may work |
| `ESCALATE` | Beyond the agent's capability — flag for human review |

Ground truth labels come from GPT auto-labeling (638 traces). We compare against Llama labels for inter-annotator agreement.

---

## File Overview

```
squad_b/
├── main.py               # CLI entry point — run experiments from here
├── data_loader.py         # Loads data, builds features, handles train/test splits
├── tfidf_baseline.py      # TF-IDF + Logistic Regression / Random Forest baselines
├── embedding_pipeline.py  # OpenAI embedding-based classifiers (SVM, LogReg)
├── llm_classifier.py      # Zero-shot and few-shot LLM classification
├── evaluator.py           # Shared metrics, confusion matrix, result export
├── cache/                 # Auto-generated: cached embeddings & LLM predictions
└── results/               # Auto-generated: JSON result files per experiment
```

### `data_loader.py`
- Loads traces from `data/labeling_exports/failed_traces.jsonl`
- Joins GPT and Llama labels by `trace_id`
- **Text features**: flattens each trace into a structured text block (domain, problem, steps, tool outputs, final answer)
- **Structured features**: extracts numerical features from trace data (step counts, tool failure rates, domain one-hots) and optionally from `trace_metrics` in SQLite
- Provides `build_dataset()` which returns everything needed for any classifier
- Stratified train/test splits with `RANDOM_SEED = 42` for reproducibility

### `tfidf_baseline.py`
- Fits TF-IDF (unigrams + bigrams, max 10k features) on flattened trace text
- Optionally stacks structured features alongside TF-IDF
- Trains **Logistic Regression** and **Random Forest** (both class-weight balanced)
- Also reports a **majority baseline** for reference

### `embedding_pipeline.py`
- Embeds traces via OpenAI `text-embedding-3-small` through OpenRouter
- Caches embeddings to `cache/embeddings.npy` so you only pay once
- Trains **SVM (RBF)** and **Logistic Regression** on the embedding vectors

### `llm_classifier.py`
- **Zero-shot**: sends each test trace to an LLM with action definitions, asks it to pick one
- **Few-shot**: same but includes k labeled examples per class from the training set
- Caches predictions per trace to avoid re-running (saves cost)
- Rate-limited with 0.5s delay between calls
- Default model: `google/gemini-3-flash-preview` (configurable via `--model`)

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

# Embedding pipeline (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY="your-key-here"
python -m squad_b.main --task embedding

# LLM zero-shot (limit to 20 traces to save cost)
python -m squad_b.main --task llm-zero --limit 20

# LLM few-shot with 2 examples per class
python -m squad_b.main --task llm-few --limit 20 --k-per-class 2

# Run everything
python -m squad_b.main --task all
```

Results are saved automatically to `squad_b/results/`.

---

## What You Need to Know Before Changing Things

Several values are **hardcoded to match the current dataset**. If the data changes, you'll need to update these — all in `data_loader.py` unless noted:

### Must update if the label taxonomy changes
- `TARGET_CLASSES` (line 31) — the five action labels
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
- `DB_PATH`, `GPT_LABELS_PATH`, `LLAMA_LABELS_PATH`, `FAILED_TRACES_PATH` (lines 22–25)

### Must update if SQLite schema changes
- The SQL query in `extract_structured_features_from_sqlite()` (line 177) assumes specific column names in `trace_metrics`

---

## Current Results (already generated)

| Method | File |
|--------|------|
| TF-IDF + Logistic Regression | `results/tfidf_logreg_results.json` |
| TF-IDF + Random Forest | `results/tfidf_rf_results.json` |
| Majority Baseline | `results/majority_baseline_results.json` |

Embedding and LLM results will appear after running those tasks with an API key.

---

## Dependencies

```
numpy, scikit-learn, scipy    # for ML baselines
openai                        # for embeddings & LLM calls (via OpenRouter)
```

All imports are standard except `openai`, which is only needed for embedding and LLM tasks. TF-IDF baselines run without any API keys.
