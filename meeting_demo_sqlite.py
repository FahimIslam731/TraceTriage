#!/usr/bin/env python3
"""
=============================================================================
CausalFlow Analysis — Meeting Demo (SQLite Version)
=============================================================================
This script demonstrates how to answer the exact same key research questions 
using the clean SQLite database (`causal_runs.sqlite`) instead of CSV files.
This is much cleaner for the repository!
"""

import sqlite3
import pandas as pd

print("Connecting to SQLite database...")
conn = sqlite3.connect("causal_runs.sqlite")
print("Connected successfully!\n")

# ---------------------------------------------------------------------------
# QUESTION 1: The Summary View 
# "Are we identifying root causes equally well across different domains?"
# ---------------------------------------------------------------------------
print("=" * 70)
print("QUESTION 1: Are we identifying root causes equally well across domains?")
print("=" * 70)

# We use SQL to join the traces and metrics tables, and calculate the proportion
query1 = """
SELECT 
    t.benchmark as domain, 
    COUNT(t.trace_id) as total_failures, 
    ROUND(AVG(t.num_steps), 2) as avg_steps_per_trace, 
    ROUND(SUM(m.num_identified_causal_steps) * 100.0 / SUM(t.num_steps), 1) as avg_causal_proportion_pct
FROM traces t
JOIN trace_metrics m ON t.trace_id = m.trace_id
WHERE t.success = 0
GROUP BY t.benchmark
"""
domain_stats = pd.read_sql(query1, conn)

print("\nResults:")
print(domain_stats.to_string(index=False))
print("\nTakeaway: Math tasks have a much higher percentage of steps flagged as causal compared to search tasks.")
print("\n")


# ---------------------------------------------------------------------------
# QUESTION 2: The Microscope View 
# "When the pipeline flags a step as causal, do the judges actually agree?"
# ---------------------------------------------------------------------------
print("=" * 70)
print("QUESTION 2: When the pipeline flags a step, do the judges agree?")
print("=" * 70)

# 1. Total steps the pipeline flagged as causal
total_flagged = conn.execute("SELECT SUM(num_identified_causal_steps) FROM trace_metrics").fetchone()[0]

# 2. Look at what the judges voted (grouping by step since there are 2 judges per step)
reviewed = conn.execute("""
    SELECT 
        SUM(judge_says_causal = 1), 
        SUM(judge_says_causal = 0)
    FROM (
        SELECT step_uid, MAX(judge_says_causal) as judge_says_causal
        FROM judge_votes
        GROUP BY step_uid
    )
""").fetchone()

agreed, disagreed = reviewed
unreviewed = total_flagged - (agreed + disagreed)

print(f"\nOut of {total_flagged} steps flagged as causal by the pipeline:")
print(f"  - Judges AGREED (causal):     {agreed} ({(agreed/total_flagged)*100:.1f}%)")
print(f"  - Judges DISAGREED (not_causal): {disagreed} ({(disagreed/total_flagged)*100:.1f}%)")
if unreviewed > 0:
    print(f"  - Unreviewed by judges:       {unreviewed} ({(unreviewed/total_flagged)*100:.1f}%)")

print("\nTakeaway: There is significant disagreement between the pipeline and the judges.")
print("\n")


# ---------------------------------------------------------------------------
# QUESTION 3: The Repair Quality View
# "Are our repairs 'cheating' by just writing much longer essays?"
# ---------------------------------------------------------------------------
print("=" * 70)
print("QUESTION 3: Are repairs cheating by just adding lots of text?")
print("=" * 70)

# We grab the text lengths from SQL and pull into Pandas to calculate the Median easily
query3 = """
SELECT 
    t.benchmark as domain,
    LENGTH(r.repaired_text) * 1.0 / NULLIF(LENGTH(r.original_text), 0) as text_len_ratio
FROM repair_attempts r
JOIN traces t ON r.trace_id = t.trace_id
WHERE r.repair_succeeded = 1
"""
df3 = pd.read_sql(query3, conn)
# Fill NaN with 0.0 for search tasks where text is empty (only tool args change)
df3['text_len_ratio'] = df3['text_len_ratio'].fillna(0.0)

print("\nAverage length ratio of repaired text vs original text:")
avg_ratio = df3['text_len_ratio'].mean()
print(f"  Overall Average: {avg_ratio:.2f}x")

ratio_by_domain = df3.groupby('domain')['text_len_ratio'].agg(['count', 'mean', 'median']).round(2)
print("\nBreakdown by domain:")
print(ratio_by_domain)

print("\nTakeaway: Repairs generally don't drastically increase the length of the step,")
print("especially in search tasks where tool arguments are modified instead of text.")
print("=" * 70)
