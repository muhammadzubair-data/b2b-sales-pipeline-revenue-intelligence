"""Runs every .sql file in sql/queries/ against the local SQLite DB and
saves the result as a CSV preview under sql/query_outputs/. Used to validate
the analytics layer and to produce sample outputs for the README/portfolio
write-up.
"""
import glob
import os
import sqlite3
import pandas as pd

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "crm.db")
OUT_DIR = os.path.join(HERE, "query_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

for path in sorted(glob.glob(os.path.join(HERE, "queries", "*.sql"))):
    name = os.path.splitext(os.path.basename(path))[0]
    sql = open(path).read()
    try:
        df = pd.read_sql_query(sql, conn)
        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        df.head(200).to_csv(out_path, index=False)
        print(f"OK   {name:45s} -> {len(df):>8,} rows (preview saved)")
    except Exception as e:
        print(f"FAIL {name:45s} -> {e}")

conn.close()
