"""Loads all raw tables into a local SQLite database (crm.db) so the .sql
analytics scripts in this folder can be run and validated with sqlite3 /
DB Browser / any SQL client. In production this would be Snowflake/Postgres/
SQL Server - the queries here use standard ANSI SQL (CTEs, window functions)
that port directly.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import load_table  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "crm.db")
TABLES = ["territories", "sales_reps", "products", "accounts", "contacts",
          "leads", "opportunities", "pipeline_history", "sales_activities",
          "quotes", "won_deals", "lost_deals"]

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
for t in TABLES:
    df = load_table(t)
    # SQLite has no native date type - store ISO strings for date columns
    for col in df.columns:
        if "date" in col.lower() and df[col].dtype != "O":
            df[col] = df[col].astype(str).replace("NaT", None)
    df.to_sql(t, conn, if_exists="replace", index=False)
    print(f"loaded {t}: {len(df):,} rows")

# Helpful indexes for the analytics queries
cur = conn.cursor()
cur.executescript("""
CREATE INDEX idx_opp_account ON opportunities(account_id);
CREATE INDEX idx_opp_rep ON opportunities(rep_id);
CREATE INDEX idx_opp_stage ON opportunities(stage);
CREATE INDEX idx_opp_status ON opportunities(close_status);
CREATE INDEX idx_opp_created ON opportunities(created_date);
CREATE INDEX idx_activities_opp ON sales_activities(opportunity_id);
CREATE INDEX idx_pipeline_opp ON pipeline_history(opportunity_id);
CREATE INDEX idx_accounts_territory ON accounts(territory_id);
""")
conn.commit()
conn.close()
print(f"\nSQLite DB built at {DB_PATH}")
