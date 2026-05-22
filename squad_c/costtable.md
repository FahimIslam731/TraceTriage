# Squad C — Cost Table

> Pricing last updated: 2026-05-21. Rates confirmed from Stage B actuals (back-calculated exact input/output split).

## Pricing by Model (OpenRouter, verified from actuals)

| Model | Domain | Input ($/M tokens) | Output ($/M tokens) | Notes |
|---|---|---|---|---|
| `google/gemini-2.0-flash-lite-001` | GSM8K | $0.075 | $0.30 | Cheapest model — 4× cheaper output than flash-preview |
| `google/gemini-3-flash-preview` | SealQA, MedBrowseComp | $0.50 | $3.00 | Most tokens used overall (RETRIEVE_MORE context) |
| `openai/gpt-5-chat` | MBPP | $1.25 | $10.00 | Most expensive output rate |

## Stage B: Per-Model Actuals (full run)

| Model | Calls | Input Tokens | Output Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| `gemini-2.0-flash-lite-001` | 928 | 59,111 | 113,315 | 172,426 | $0.0384 |
| `gemini-3-flash-preview` | 2,963 | 1,375,458 | 347,011 | 1,722,469 | $1.7288 |
| `gpt-5-chat` | 2,360 | 213,176 | 111,426 | 324,602 | $1.3807 |
| **TOTAL** | **6,252** | **1,647,745** | **571,752** | **2,219,497** | **$3.1479** |

> `gpt-5-chat` has an asymmetric output rate ($10/M) — expensive despite fewest total tokens because MBPP solutions are verbose.

## Stage A: Pilot (100 traces per domain, all 146 SealQA)

### Actuals by Action

| Action | Calls | Tokens | Cost (USD) |
|---|---|---|---|
| ESCALATE | 400 | 0 | $0.0000 |
| LOCAL_REPAIR | 400 | 42,184 | $0.0676 |
| RETRY | 400 | 109,917 | $0.2391 |
| RETRIEVE_MORE | 200 | 525,701 | $0.3501 |
| TOOL_FIX | 300 | 101,812 | $0.2466 |
| REPLAN | 400 | 136,390 | $0.2913 |
| **TOTAL** | **2,100** | **916,004** | **$1.1946** |

### Notes
- ESCALATE costs $0 (no model call); LOCAL_REPAIR for GSM8K costs $0 (pre-computed text repair returned directly)
- RETRIEVE_MORE is the most expensive action — large context from web search results
- RETRIEVE_MORE runs 5 Serper searches per trace (~1,000 Serper queries used)
- Domains covered: GSM8K (100), MBPP (100), SealQA (100 of 146), MedBrowseComp (100)
- Model used per domain: GSM8K → gemini-2.0-flash-lite-001, MBPP → gpt-5-chat, SealQA/MedBrowseComp → gemini-3-flash-preview

---

## Stage B: Full Simulation (all 1204 traces)

### Call Counts by Domain

| Domain | Traces | Actions | Total Calls |
|---|---|---|---|
| GSM8K | 250 | LOCAL_REPAIR, RETRY, REPLAN, ESCALATE (4) | 1,000 |
| MBPP | 472 | LOCAL_REPAIR, RETRY, REPLAN, TOOL_FIX, ESCALATE (5) | 2,360 |
| SealQA | 146 | All 6 | 876 |
| MedBrowseComp | 336 | All 6 | 2,016 |
| **TOTAL** | **1,204** | | **6,252** |

### Actuals by Action

| Action | Calls | Tokens | Cost (USD) |
|---|---|---|---|
| ESCALATE | 1,204 | 0 | $0.0000 |
| LOCAL_REPAIR | 1,204 | 76,536 | $0.2097 |
| RETRY | 1,204 | 272,162 | $0.6401 |
| RETRIEVE_MORE | 482 | 1,226,855 | $0.7810 |
| TOOL_FIX | 954 | 307,144 | $0.7708 |
| REPLAN | 1,204 | 336,800 | $0.7463 |
| **TOTAL** | **6,252** | **2,219,497** | **$3.1479** |

### Serper API (RETRIEVE_MORE)

| Item | Queries |
|---|---|
| Serper free tier (monthly) | 2,500 |
| Used in Stage A | ~1,000 |
| Used in Stage B (full) | 2,410 (482 calls × 5) |
| Remaining after Stage B | ~90 queries |

> **Note:** Serper budget was tight — only ~90 queries remaining after Stage B. Do not re-run RETRIEVE_MORE traces against the free-tier key without resetting the monthly quota.

---

## Summary

| Stage | Calls | Tokens | Cost | Serper Queries |
|---|---|---|---|---|
| Stage A (pilot) | 2,100 | 916,004 | $1.1946 (actual) | ~1,000 |
| Stage B (full run, all 1204 traces) | 6,252 | 2,219,497 | $3.1479 (actual) | 2,410 |
| **Grand Total** | **8,352** | **3,135,501** | **$4.3425** | **~3,410** |
