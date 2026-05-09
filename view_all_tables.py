import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("causal_runs.sqlite")

# The 7 tables available in the database
table_names = [
    "runs",
    "traces",
    "steps",
    "repair_attempts",
    "judge_votes",
    "consensus_steps",
    "trace_metrics"
]

print("Loading all tables into Pandas...\n")

# A dictionary to hold all our DataFrames
dfs = {}

for table in table_names:
    # Load the table into a pandas DataFrame
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    dfs[table] = df
    
    # Print out a summary of the table
    print(f"✅ Loaded '{table}' table!")
    print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # If you want to see all the columns available in this table, uncomment the line below:
    # print(f"   Columns: {list(df.columns)}")
    print("-" * 50)

print("\nDone! You now have a dictionary `dfs` where every table is a Pandas DataFrame.")
print("For example, dfs['steps'] has all the step data, and dfs['judge_votes'] has all the judge data.")
