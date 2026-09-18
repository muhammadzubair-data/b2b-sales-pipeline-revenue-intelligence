"""Shared helpers for loading the raw CRM tables used across every script."""
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

DATE_COLS = {
    "accounts": ["created_date"],
    "contacts": ["created_date"],
    "leads": ["created_date"],
    "opportunities": ["created_date", "expected_close_date", "actual_close_date"],
    "pipeline_history": ["entered_date", "exited_date"],
    "sales_activities": ["activity_date"],
    "quotes": ["quote_date"],
    "won_deals": ["close_date"],
    "lost_deals": ["close_date"],
    "sales_reps": ["hire_date"],
}


def load_table(name: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{name}.csv.gz")
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    for col in DATE_COLS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


AS_OF_DATE = pd.Timestamp("2026-09-18")
