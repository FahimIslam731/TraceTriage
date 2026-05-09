import sqlite3
import pandas as pd

print("Opening the SQLite container...")
conn = sqlite3.connect("causal_runs.sqlite")

print("\nPulling the 'traces' table into a Pandas DataFrame...")
# We use pd.read_sql to run a standard SQL query and turn the result into a DataFrame
traces_df = pd.read_sql("SELECT * FROM traces", conn)

print(f"\nSuccess! We loaded {len(traces_df)} rows.")
print("\nHere are the first 3 rows of the DataFrame:")

# Print just a few columns so it fits nicely on the screen
columns_to_show = ['problem_id', 'benchmark', 'num_steps', 'success']
print(traces_df[columns_to_show].head(3))
