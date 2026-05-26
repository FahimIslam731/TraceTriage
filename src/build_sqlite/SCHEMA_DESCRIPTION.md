# Causal Runs SQLite Schema
This database flattens the Mongo/BSON agent-run data into SQLite tables that are easier to inspect, query, label, and use for the Trace Triage paper.

The main idea:

- `runs` describes each benchmark run.
- `traces` describes each problem attempt.
- `steps` breaks each trace into individual agent actions.
- `repair_attempts`, `judge_votes`, `consensus_steps`, and `trace_metrics` store CausalFlow analysis results.
- `triage_labels` is the main working table for the Trace Triage paper.

For the paper, the most important workflow is:

```text
failed trace -> determine recovery action label
```

The recovery action labels are:

```text
LOCAL_REPAIR
RETRY
REPLAN
RETRIEVE_MORE
TOOL_FIX
ESCALATE
```

## Important Ablation Note

The database includes normal runs and ablation runs.

An ablation run is an experimental variant where some part of the system was removed or changed to test its importance.

In this data, the main ablation types are:

```text
nogold
minimality
```

In tables with ablation columns:

```text
is_ablation = 0, ablation_type = NULL
```

means this is a normal non-ablation row.

```text
is_ablation = 1, ablation_type = nogold
```

means this came from a no-gold ablation run.

```text
is_ablation = 1, ablation_type = minimality
```

means this came from a minimality ablation run.

So if `ablation_type` is null, that usually just means the row is from a normal run.

## Table: `runs`

One row per benchmark run.

Use this table for run-level summaries, benchmark/model comparisons, and ablation filtering.

| Column | Meaning |
|---|---|
| `run_id` | Unique run ID |
| `experiment_name` | Original experiment/run name from Mongo |
| `benchmark` | Cleaned benchmark name |
| `domain` | Normalized task domain, such as `GSM8K`, `MBPP`, `SealQA`, `MedBrowseComp` |
| `is_ablation` | Whether this is an ablation run |
| `ablation_type` | Ablation kind, such as `minimality` or `nogold` |
| `ablation_target` | Domain targeted by the ablation |
| `timestamp` | Run timestamp |
| `model_used` | Agent model, if recorded |
| `num_problems_declared` | Original declared problem count |
| `num_traces_stored` | Actual number of traces stored |
| `num_passing_traces` | Count of passing traces |
| `num_failing_traces` | Count of failing traces |
| `stats_total` | Reported total count from run stats |
| `stats_passing` | Reported passing count |
| `stats_failing` | Reported failing count |
| `stats_fixed` | Reported repaired/fixed count |
| `stats_accuracy` | Reported run accuracy |
| `total_experiment_time_minutes` | Runtime in minutes |
| `source_file` | BSON file this run came from |

## Table: `traces`

One row per problem attempt.

A trace is the original agent's attempt at one problem. It can be passing or failing.

Use this table for the failed-trace dataset, problem text, gold/final answers, and classification examples.

| Column | Meaning |
|---|---|
| `trace_id` | Unique trace ID |
| `run_id` | Parent run |
| `problem_id` | Problem/task ID |
| `benchmark` | Benchmark name |
| `domain` | Normalized task domain |
| `model_used` | Model used by the original agent |
| `is_ablation` | Whether this trace came from an ablation run |
| `ablation_type` | Ablation kind, if applicable |
| `timestamp` | Trace timestamp |
| `success` | Whether the original agent got the problem right |
| `problem_statement` | Original problem/prompt |
| `gold_answer` | Correct answer |
| `final_answer` | Agent's final answer |
| `num_steps` | Number of trace steps |
| `causal_flow_analysis_minutes` | CausalFlow analysis time |
| `is_passing_trace` | Whether stored in `passing_traces` |
| `is_failing_trace` | Whether stored in `failing_traces` |
| `answer_exact_match` | Simple string equality check between gold and final answer |

Note: `success` and `is_passing_trace` should usually agree, but they come from slightly different meanings:

- `success` is the semantic outcome.
- `is_passing_trace` / `is_failing_trace` describes which raw Mongo list stored the trace.

## Table: `steps`

One row per individual step inside a trace.

This table breaks full traces into ordered agent actions. A step can be reasoning, a tool call, a tool response, an LLM response, or the final answer.

Use this table for trace summaries, feature extraction, and failure-mode analysis.

| Column | Meaning |
|---|---|
| `step_uid` | Unique step ID |
| `trace_id` | Parent trace |
| `run_id` | Parent run |
| `problem_id` | Problem/task ID |
| `step_id` | Step ID from the raw trace |
| `step_index` | Step order within the trace |
| `step_type` | Step kind, such as `reasoning`, `tool_call`, `tool_response`, `final_answer` |
| `dependencies_json` | Prior-step dependencies as JSON |
| `text` | Step text, reasoning, or response content |
| `tool_name` | Tool used, such as `calculator`, `web_search`, `web_fetch`, code execution |
| `tool_args_json` | Tool inputs as JSON |
| `tool_output_json` | Tool output as JSON/text |
| `tool_call_result` | Tool success/failure flag, when available |
| `state_snapshot_json` | Browser/search/code state snapshot |
| `trace_success` | Whether the parent trace succeeded |
| `has_tool` | Whether this step used a tool |
| `is_reasoning_step` | Reasoning-step flag |
| `is_tool_call` | Tool-call flag |
| `is_tool_response` | Tool-response flag |
| `is_final_answer` | Final-answer flag |
| `text_length` | Length of the step text |

Example analysis question:

```text
Do search failures cluster around web_search, web_fetch, or later reasoning steps?
```

This means: among failed traces, are the important/repairable steps often search calls, page fetches, or reasoning after retrieval?

## Table: `repair_attempts`

One row per successful repairable step recovered from CausalFlow outputs.

A repair is a post-hoc counterfactual intervention: the original agent failed, then CausalFlow tried changing one step to see whether the failure could be fixed.

This is not the original agent fixing itself during the run. It is external post-hoc analysis.

Use this table to identify `LOCAL_REPAIR` cases.

| Column | Meaning |
|---|---|
| `repair_id` | Unique repair row |
| `step_uid` | Repaired step |
| `trace_id` | Failed trace |
| `run_id` | Parent run |
| `problem_id` | Problem/task ID |
| `step_id` | Repaired step ID |
| `repair_idx` | Repair index/key from raw data |
| `success_predicted` | Whether repair was predicted successful |
| `repair_succeeded` | Whether repair succeeded |
| `minimality_score` | Overall minimality score |
| `minimality_lex` | Lexical minimality |
| `minimality_edit` | Edit-distance minimality |
| `minimality_sem` | Semantic minimality |
| `original_step_type` | Original step type |
| `original_text` | Original step content |
| `repaired_text` | Repaired step content |
| `original_tool_name` | Original tool, if any |
| `original_tool_args_json` | Original tool args |
| `repaired_tool_name` | Repaired tool, if any |
| `repaired_tool_args_json` | Repaired tool args |
| `raw_repair_json` | Full original repair object |

Important:

```text
If a failed trace has successful repair evidence, it is auto-labeled LOCAL_REPAIR in triage_labels.
```

## Table: `judge_votes`

One row per LLM judge vote on whether a step caused failure.

In the raw CausalFlow analysis, candidate causal steps could be proposed and then critiqued by judge agents.

Use this table for LLM-as-a-judge or LLMs-as-a-council analysis.

| Column | Meaning |
|---|---|
| `judge_vote_id` | Unique vote ID |
| `step_uid` | Judged step |
| `trace_id` | Failed trace |
| `run_id` | Parent run |
| `problem_id` | Problem/task ID |
| `step_id` | Judged step ID |
| `candidate_idx` | Candidate causal-step key |
| `proposed_by` | Proposer agent, often `Agent_A` |
| `judge_agent` | Judge agent name, often `Agent_B` or `Agent_C` |
| `judge_role` | Critic/meta-critic role |
| `agrees` | Whether the judge agreed with the proposal |
| `confidence` | Judge confidence |
| `reasoning` | Judge rationale |
| `agreement_text` | Agreement text, if present |
| `evidence_strength` | Evidence-strength text, if present |
| `judge_says_causal` | Derived causal verdict |
| `is_repairable_step` | Whether the judged step was repairable |
| `vote_matches_repairability` | Whether judge verdict matched repairability |

This table is useful for questions like:

```text
When judges say a step caused failure, was that step actually repairable?
```

## Table: `consensus_steps`

One row per step selected by the judge/council process.

Use this table to compare council-selected causal steps against repairable steps.

| Column | Meaning |
|---|---|
| `consensus_id` | Unique consensus row |
| `step_uid` | Selected step |
| `trace_id` | Failed trace |
| `run_id` | Parent run |
| `problem_id` | Problem/task ID |
| `step_id` | Selected step ID |
| `step_type` | Selected step type |
| `consensus_score` | Council score, if available |
| `final_verdict` | Final verdict, if available |
| `proposed_by` | Proposer |
| `num_critiques` | Number of critiques |
| `final_critic_summary` | Council summary |
| `text` | Step text |
| `tool_name` | Tool used, if any |
| `tool_args_json` | Tool args |
| `dependencies_json` | Dependencies |
| `is_repairable_step` | Whether selected step was repairable |
| `has_successful_repair` | Whether selected step had a successful repair |
| `raw_consensus_json` | Full original consensus object |

Ideally, a council-selected step would be repairable, but that is an empirical question. The mismatch is useful for analysis.

## Table: `trace_metrics`

One row per failed trace with CausalFlow summary metrics.

Use this table for trace-level repair and attribution summaries.

| Column | Meaning |
|---|---|
| `trace_id` | Failed trace |
| `run_id` | Parent run |
| `problem_id` | Problem/task ID |
| `minimality_average` | Average repair minimality |
| `minimality_min` | Minimum repair minimality |
| `minimality_max` | Maximum repair minimality |
| `num_identified_causal_steps` | Number of identified causal steps |
| `attribution_precision` | Causal attribution precision |
| `attribution_recall` | Causal attribution recall |
| `attribution_f1` | Causal attribution F1 |
| `repairs_attempted` | Number of repair attempts |
| `repairs_successful` | Number of successful repairs |
| `repairs_failed` | Number of failed repairs |
| `repair_success_rate` | Repair success rate |
| `num_successful_repair_steps` | Number of successful repaired steps |
| `num_consensus_steps` | Number of council-selected steps |
| `multi_agent_skipped` | Whether council critique was skipped |
| `multi_agent_skip_reason` | Skip reason |
| `causal_steps_json` | Raw causal step IDs |
| `identified_steps_json` | Raw identified causal step objects |

## Table: `triage_labels`

This is the main table for the Trace Triage paper.

One row per failed trace. This table stores the final recovery-action label and all intermediate labels from LLMs/humans.

Use this table for Squad A labeling, train/dev/test split creation, and classifier training.

| Column | Meaning |
|---|---|
| `trace_id` | Failed trace ID |
| `run_id` | Parent run |
| `problem_id` | Problem/task ID |
| `benchmark` | Benchmark name |
| `domain` | Task domain |
| `is_ablation` | Whether this row came from an ablation run |
| `ablation_type` | Ablation kind, if applicable |
| `action_label` | Final recovery label |
| `label_source` | Where final label came from |
| `is_auto_labeled` | Whether label was auto-assigned |
| `needs_labeling` | Whether this trace still needs labeling |
| `is_local_repairable` | Whether CausalFlow fixed this trace |
| `num_successful_repair_steps` | Number of successful repair steps |
| `applicable_actions_json` | Actions allowed for this domain |
| `llm_1_action` | First LLM label |
| `llm_1_rationale` | First LLM rationale |
| `llm_2_action` | Second LLM label |
| `llm_2_rationale` | Second LLM rationale |
| `human_action` | Human audit label |
| `human_rationale` | Human rationale |
| `split` | Train/dev/test split |

### Current Auto-Labeling Rule

Right now, `action_label` is only assigned automatically for local repair cases.

The logic is:

```text
If CausalFlow found at least one successful repair:
    action_label = LOCAL_REPAIR
    label_source = auto_causalflow
    is_auto_labeled = 1
    needs_labeling = 0

If CausalFlow did not find a successful repair:
    action_label = NULL
    label_source = NULL
    is_auto_labeled = 0
    needs_labeling = 1
```

So `NULL` in `action_label` does not mean unknown forever. It means the trace still needs to be labeled into one of:

```text
RETRY
REPLAN
RETRIEVE_MORE
TOOL_FIX
ESCALATE
```

### Columns We Need To Fill During Experiments

These columns are intentionally empty at first:

| Column | When it gets filled |
|---|---|
| `llm_1_action` | After first labeling model assigns a recovery action |
| `llm_1_rationale` | After first labeling model gives rationale |
| `llm_2_action` | After second labeling model assigns a recovery action |
| `llm_2_rationale` | After second labeling model gives rationale |
| `human_action` | During human audit/adjudication |
| `human_rationale` | During human audit/adjudication |
| `split` | After train/dev/test splits are frozen |
| `action_label` | For non-LOCAL_REPAIR rows, after LLM agreement or human adjudication |
| `label_source` | Updated to show whether the final label came from auto-labeling, LLM agreement, or human audit |

## Common Queries

Get labeled failed traces with problem text:

```sql
SELECT t.*, l.action_label
FROM triage_labels l
JOIN traces t ON l.trace_id = t.trace_id;
```

Get traces that still need labeling:

```sql
SELECT l.trace_id, l.domain, t.problem_statement, t.gold_answer, t.final_answer
FROM triage_labels l
JOIN traces t ON l.trace_id = t.trace_id
WHERE l.needs_labeling = 1;
```

View ablation counts:

```sql
SELECT is_ablation, ablation_type, COUNT(*)
FROM triage_labels
GROUP BY is_ablation, ablation_type;
```

Get full step trace for one example:

```sql
SELECT s.*
FROM steps s
WHERE s.trace_id = ?
ORDER BY s.step_index;
```

Compare judge/council decisions to repairability:

```sql
SELECT judge_says_causal, is_repairable_step, COUNT(*)
FROM judge_votes
WHERE is_repairable_step IS NOT NULL
GROUP BY judge_says_causal, is_repairable_step;
```
