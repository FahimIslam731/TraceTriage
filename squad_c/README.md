# Squad C — Recovery Simulation (Experiments 4 & 5)

Squad C runs the actual recovery actions on failed traces, compares 8 routing policies, and produces the cost-aware utility analysis for the paper.

---

## File Overview

```
squad_c/
├── README.md                  # this file
├── costtable.md               # full cost breakdown with actuals (Stage A + B)
├── all_1212_labels.csv        # Squad A human majority labels (source of truth for triage routing)
│
├── recovery_actions.py        # 6 recovery action implementations (RETRY, REPLAN, etc.)
├── run_recovery.py            # CLI runner — executes actions across traces
├── analyze_policies.py        # policy comparison + Stage A decision gate
├── cost_tracker.py            # thread-safe token/cost/latency logging (JSONL append)
├── verify.py                  # answer verification per domain (GSM8K/MBPP/SealQA/MedBrowseComp)
├── docker_code_executor.py    # MBPP code execution via Docker sandbox
│
├── test_recovery.py           # unit tests + small live tests (--small, --live flags)
├── test_openrouter.py         # quick API connectivity check
│
└── results/                   # gitignored — all outputs live here
    ├── pilot_results.txt              # Stage A human-readable summary
    ├── recovery_results.jsonl         # one RecoveryResult per (trace, action)
    ├── cost_log.jsonl                 # one CallRecord per model call (full audit trail)
    ├── summary.json                   # aggregate stats from run_recovery
    ├── policy_comparison.json         # 8-policy metrics table (primary Experiment 4 output)
    ├── decision_gate.json             # Stage A gate result (pass/fail + margin)
    └── squad_b_best_classifier/
        └── Gemini/                    # Squad B LLM few-shot predictions per domain
            └── llm_fewshot_*_predictions.json
```

---

## How to Run

### Prerequisites

Keys in `.env` at the project root:
```
OPENROUTER_API_KEY=...
SERPER_API_KEY=...
```

### Step 1 — Sanity check (no API key needed)
```bash
python -m squad_c.test_recovery
```

### Step 2 — Stage A pilot (100 traces/domain)
```bash
python -m squad_c.run_recovery --pilot --domain GSM8K MBPP SealQA MedBrowseComp
python -m squad_c.analyze_policies --stage a
```
Outputs `pilot_results.txt`, `decision_gate.json`.

### Step 3 — Stage B full simulation (all 1204 traces)
```bash
python -m squad_c.run_recovery --full --domain GSM8K MBPP SealQA MedBrowseComp
python -m squad_c.analyze_policies --stage b
```
Outputs `recovery_results.jsonl`, `policy_comparison.json`.

### Other useful commands
```bash
# Dry run — count traces, no model calls
python -m squad_c.run_recovery --pilot --dry-run

# Test one action on one trace
python -m squad_c.run_recovery --test-action RETRY --trace-id <trace_id>

# Quick API check
python -m squad_c.test_openrouter
```

Runs are **resumable** — re-running skips already-completed (trace, action) pairs.

---

## Stage A Results (Pilot — 400 traces)

| Domain | Action | Success Rate | Avg Cost |
|---|---|---|---|
| GSM8K | LOCAL_REPAIR | 62% | $0.000 |
| GSM8K | RETRY | 79% | $0.000 |
| GSM8K | REPLAN | 84% | $0.000 |
| MBPP | LOCAL_REPAIR | 36% | $0.000 |
| MBPP | RETRY | 10% | $0.001 |
| MBPP | REPLAN | 9% | $0.001 |
| MBPP | TOOL_FIX | 53% | $0.001 |
| SealQA | RETRIEVE_MORE | 52% | $0.000 |
| SealQA | REPLAN | 50% | $0.000 |
| SealQA | TOOL_FIX | 47% | $0.000 |
| MedBrowseComp | RETRIEVE_MORE | 16% | $0.000 |
| MedBrowseComp | REPLAN | 9% | $0.000 |

**Decision gate:** FAILED — trace_triage beat domain_policy by only **2.24 ppt** (threshold: 5.0 ppt). Stage B was run per PI decision.

---

## Stage B Results (Full — 1204 traces, 8 policies) — 
Stage B full run (6,252 rows, not 7,224 because RETRIEVE_MORE is SealQA/MedBrowseComp only and TOOL_FIX excludes GSM8K)

| Policy | Recovery Rate | Util (λ=1) | Util (λ=2) | Notes |
|---|---|---|---|---|
| `oracle` | **54.9%** | 0.5487 | 0.5483 | Upper bound — best action that actually worked |
| `trace_triage_human_label` | 42.1% | 0.4204 | 0.4196 | Human majority labels — theoretical ceiling |
| `always_retrieve_more` | 40.5% | 0.4045 | 0.4037 | Strong fixed baseline |
| `domain_policy` | 40.4% | 0.4030 | 0.4023 | Modal action per domain |
| `trace_triage_classifier` | **43.6%** | 0.4354 | 0.4347 | Squad B Gemini few-shot classifier |
| `always_local_repair` | 32.9% | 0.3287 | 0.3286 | CausalFlow only |
| `always_replan` | 27.2% | 0.2710 | 0.2704 | |
| `always_retry` | 24.9% | 0.2486 | 0.2481 | Weakest fixed policy |

**Key finding:** Human-label trace_triage (42.1%) beats domain_policy (40.4%) by +1.7 ppt. The Squad B classifier (43.6%) exceeds domain_policy (40.2%), suggesting the model successfully regularizes across noisy annotations to discover more robust recovery paths.

Human triage recovers 42.1% of failed traces — capturing 77% of the theoretical maximum (oracle: 54.9%).

> **LOCAL_REPAIR cost note:** Cost behaviour differs by domain. For **GSM8K**, cost is $0 — the final answer is the pre-computed `repaired_text` returned directly from SQLite, no API call needed. For **MBPP, MedBrowseComp, and SealQA**, LOCAL_REPAIR does make a model call (MBPP generates code from the repaired reasoning; MedBrowseComp/SealQA re-run a search with the repaired query), so cost is non-zero. In all cases, the token cost of producing the `repaired_text` in the first place is not accounted for — CausalFlow runs do not report token usage to Squad C, so that upstream compute cost is invisible here.


---

## recovery_results.jsonl Schema

One JSON record per `(trace, action)` pair — 6,252 rows total.

| Field | Type | Description |
|---|---|---|
| `trace_id` | str | Which failed trace was attempted |
| `action` | str | Recovery action run (RETRY, REPLAN, LOCAL_REPAIR, RETRIEVE_MORE, TOOL_FIX, ESCALATE) |
| `success` | bool | Whether `recovered_answer` matched the gold answer |
| `recovered_answer` | str \| null | The answer text produced — see nulls below |
| `input_tokens` | int | Tokens in the prompt sent to the model |
| `output_tokens` | int | Tokens generated by the model |
| `total_tokens` | int | input + output |
| `cost_usd` | float | Dollar cost of this call |
| `latency_seconds` | float | Wall-clock time for the API call |
| `model_used` | str | Model that handled this trace (domain-dependent) |
| `error` | str \| null | Error message if the call crashed, otherwise null |
| `metadata` | dict | Extra info (e.g. `repair_step_id` for LOCAL_REPAIR, search counts for RETRIEVE_MORE) |

**Why `recovered_answer` is sometimes null:**
- **ESCALATE** — always null (1,204/1,204). No model call is made; the trace is marked unrecoverable by definition.
- **LOCAL_REPAIR** — null for 768/1,204 traces. These are traces where no `repaired_text` was available in SQLite (either `is_local_repairable = 0`). Nothing to return, so `success = False` and `recovered_answer = null`.
- **RETRY / REPLAN / RETRIEVE_MORE / TOOL_FIX** — null only on API crashes (1–2 rows each).

---

## Cost Summary

See `costtable.md` for full breakdown. Top-level:

| Stage | Traces | Action Calls | Total Cost | Serper Queries |
|---|---|---|---|---|
| Stage A (pilot) | 400 | 2,100 | $1.19 | ~1,000 |
| Stage B (full) | 1,204 | 6,252 | $3.15 | 2,410 |
| **Total** | | **8,352** | **$4.34** | **~3,410** |

### Per-model (Stage B)

| Model | Domain | Calls | Cost | Rate |
|---|---|---|---|---|
| `gemini-2.0-flash-lite-001` | GSM8K | 928 | $0.04 | $0.075/$0.30 per M in/out |
| `gemini-3-flash-preview` | SealQA, MedBrowseComp | 2,963 | $1.73 | $0.50/$3.00 per M in/out |
| `gpt-5-chat` | MBPP | 2,360 | $1.38 | $1.25/$10.00 per M in/out |

Serper free tier (2,500/month): 90 queries remaining after Stage B.

---

## Squad B Classifier Predictions

`results/squad_b_best_classifier/Gemini/` contains per-domain prediction files from the best Squad B classifier (Gemini few-shot, k=5, full-trace input, cross-domain evaluation). These are loaded by `analyze_policies.py` for the `trace_triage_clf` policy. Please load this under results folder before running full experiment.

File format: `{trace_id: {action, confidence, tokens, finish_reason}}`

Coverage: 1,204 traces (250 GSM8K, 472 MBPP, 336 MedBrowseComp, 146 SealQA).

---

## 8 Policies Compared

| Policy | Description |
|---|---|
| `always_retry` | RETRY on every failure |
| `always_local_repair` | LOCAL_REPAIR (CausalFlow) on every failure |
| `always_replan` | REPLAN on every failure |
| `always_retrieve_more` | RETRIEVE_MORE where applicable, LOCAL_REPAIR fallback |
| `domain_policy` | Modal action per domain (from Squad A labels) |
| `trace_triage` | Human majority-vote label per trace (upper bound) |
| `trace_triage_clf` | Squad B best classifier prediction per trace (practical) |
| `oracle` | Whichever action actually succeeded at lowest cost |


