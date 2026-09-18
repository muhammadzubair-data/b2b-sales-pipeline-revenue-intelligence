"""
Revenue forecasting: traditional CRM method vs ML-driven method, backtested
against actual realized revenue.

Traditional method:  Sum(Pipeline Value x Stage Probability)
    Stage probabilities are the textbook CRM defaults most sales orgs hard-
    code into their CRM (Qualification 10%, Needs Analysis 25%, Proposal
    50%, Negotiation 75%).

ML method:            Sum(Deal Value x Calibrated P(Win))
    Uses the calibrated model trained in lead_scoring.py.

Both are backtested the same way: take a HISTORICAL snapshot date, forecast
what will close in the following month using only information known at that
snapshot date, then compare to what actually closed. This is repeated across
many historical snapshot months so we get a distribution of forecast errors,
not a single lucky/unlucky comparison.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import PROCESSED_DIR, ROOT, AS_OF_DATE  # noqa: E402

FIG_DIR = os.path.join(ROOT, "reports", "figures")
MODEL_DIR = os.path.join(ROOT, "models", "artifacts")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "scoring_dataset.csv.gz"), compression="gzip",
                  parse_dates=["created_date", "scoring_date", "expected_close_date", "actual_close_date"])

model = joblib.load(os.path.join(MODEL_DIR, "best_lead_scoring_model.joblib"))
NUM_FEATURES = [
    "log_deal_value", "discount_requested_pct", "num_stakeholders", "employees",
    "engagement_score", "days_since_created_at_scoring", "stage_number_at_scoring",
    "days_in_current_stage_at_scoring", "num_activities_to_date", "num_calls",
    "num_emails", "num_meetings", "num_demos", "pct_positive_outcome",
    "pct_no_response", "days_since_last_activity", "activity_span_days",
]
CAT_FEATURES = ["source", "industry", "company_size_band", "rep_seniority", "stage_at_scoring"]

TRADITIONAL_STAGE_PROB = {
    "Qualification": 0.10, "Needs Analysis": 0.25, "Proposal": 0.50, "Negotiation": 0.75,
}

closed = df[df["is_closed"]].copy()
closed["close_month"] = closed["actual_close_date"].dt.to_period("M")

# Raw event tables are required to reconstruct every historical snapshot using
# ONLY information that existed at that month start. Reusing the manufactured
# scoring_date features here would leak later activities whenever scoring_date
# fell after the historical forecast date.
from data_io import load_table  # noqa: E402
activities = load_table("sales_activities")
pipeline_history = load_table("pipeline_history")
accounts = load_table("accounts")
sales_reps = load_table("sales_reps")
opportunities = load_table("opportunities")

static = opportunities.merge(
    accounts[["account_id", "industry", "company_size_band", "employees", "engagement_score"]],
    on="account_id", how="left",
).merge(
    sales_reps[["rep_id", "seniority_level"]].rename(columns={"seniority_level": "rep_seniority"}),
    on="rep_id", how="left",
)
static["log_deal_value"] = np.log1p(static["deal_value"])

STAGE_ORDER = {"Qualification": 1, "Needs Analysis": 2, "Proposal": 3, "Negotiation": 4}

def build_snapshot(candidate_ids, snapshot_date):
    """Rebuild model features as-of snapshot_date with no look-ahead."""
    base = static[static["opportunity_id"].isin(candidate_ids)].copy()
    base["days_since_created_at_scoring"] = (snapshot_date - base["created_date"]).dt.days.clip(lower=0)

    a = activities[(activities["opportunity_id"].isin(candidate_ids)) &
                   (activities["activity_date"] < snapshot_date)].copy()
    if len(a):
        agg = a.groupby("opportunity_id").agg(
            num_activities_to_date=("activity_id", "count"),
            num_calls=("activity_type", lambda x: (x == "Call").sum()),
            num_emails=("activity_type", lambda x: (x == "Email").sum()),
            num_meetings=("activity_type", lambda x: (x == "Meeting").sum()),
            num_demos=("activity_type", lambda x: (x == "Demo").sum()),
            pct_positive_outcome=("outcome", lambda x: (x == "Positive - Advancing").mean()),
            pct_no_response=("outcome", lambda x: (x == "No Response").mean()),
            first_activity_date=("activity_date", "min"),
            last_activity_date=("activity_date", "max"),
        ).reset_index()
        agg["days_since_last_activity"] = (snapshot_date - agg["last_activity_date"]).dt.days
        agg["activity_span_days"] = (agg["last_activity_date"] - agg["first_activity_date"]).dt.days
        agg = agg.drop(columns=["first_activity_date", "last_activity_date"])
        base = base.merge(agg, on="opportunity_id", how="left")
    else:
        for c in ["num_activities_to_date","num_calls","num_emails","num_meetings","num_demos",
                  "pct_positive_outcome","pct_no_response","days_since_last_activity","activity_span_days"]:
            base[c] = np.nan

    count_cols = ["num_activities_to_date","num_calls","num_emails","num_meetings","num_demos",
                  "pct_positive_outcome","pct_no_response","activity_span_days"]
    base[count_cols] = base[count_cols].fillna(0)
    base["days_since_last_activity"] = base["days_since_last_activity"].fillna(9999)

    ph = pipeline_history[(pipeline_history["opportunity_id"].isin(candidate_ids)) &
                          (pipeline_history["entered_date"] < snapshot_date)].copy()
    if len(ph):
        latest = ph.sort_values(["opportunity_id", "entered_date"]).groupby("opportunity_id").tail(1)
        latest["stage_at_scoring"] = latest["stage"]
        latest["stage_number_at_scoring"] = latest["stage"].map(STAGE_ORDER).fillna(1)
        latest["days_in_current_stage_at_scoring"] = (snapshot_date - latest["entered_date"]).dt.days
        base = base.merge(latest[["opportunity_id","stage_at_scoring","stage_number_at_scoring",
                                  "days_in_current_stage_at_scoring"]], on="opportunity_id", how="left")
    else:
        base["stage_at_scoring"] = "Qualification"
        base["stage_number_at_scoring"] = 1
        base["days_in_current_stage_at_scoring"] = 0
    base["stage_at_scoring"] = base["stage_at_scoring"].fillna("Qualification")
    base["stage_number_at_scoring"] = base["stage_number_at_scoring"].fillna(1)
    base["days_in_current_stage_at_scoring"] = base["days_in_current_stage_at_scoring"].fillna(0)
    return base

# Strict rolling monthly backtest over the latest 15 complete/available close months.
months = sorted(closed["close_month"].unique())[-15:]
rows = []
for month in months:
    month_start, month_end = month.start_time, month.end_time
    actual_month = closed[closed["close_month"] == month]
    won = actual_month[actual_month["close_status"] == "Closed Won"]
    actual_won_revenue = (won["deal_value"] * (1 - won["discount_requested_pct"] / 100)).sum()

    # Forecast the deals scheduled to close this month that existed and were open at month start.
    candidates = static[(static["created_date"] < month_start) &
                        ((static["actual_close_date"].isna()) | (static["actual_close_date"] >= month_start)) &
                        (static["expected_close_date"] >= month_start) &
                        (static["expected_close_date"] <= month_end)].copy()
    if candidates.empty:
        continue
    snap = build_snapshot(candidates["opportunity_id"].values, month_start)
    trad_prob = snap["stage_at_scoring"].map(TRADITIONAL_STAGE_PROB).fillna(0.10)
    traditional_forecast = (snap["deal_value"] * trad_prob *
                            (1 - snap["discount_requested_pct"] / 100)).sum()
    ml_win_prob = model.predict_proba(snap[NUM_FEATURES + CAT_FEATURES])[:, 1]
    ml_forecast = (snap["deal_value"] * ml_win_prob *
                   (1 - snap["discount_requested_pct"] / 100)).sum()
    rows.append({
        "month": str(month), "n_candidates": len(snap),
        "traditional_forecast": traditional_forecast, "ml_forecast": ml_forecast,
        "actual_won_revenue": actual_won_revenue,
        "traditional_error": traditional_forecast - actual_won_revenue,
        "ml_error": ml_forecast - actual_won_revenue,
        "traditional_pct_error": 100 * (traditional_forecast - actual_won_revenue) / max(actual_won_revenue, 1),
        "ml_pct_error": 100 * (ml_forecast - actual_won_revenue) / max(actual_won_revenue, 1),
    })

backtest = pd.DataFrame(rows)
backtest.to_csv(os.path.join(ROOT, "reports", "revenue_forecast_backtest.csv"), index=False)
summary = {
    "traditional_mean_pct_error": backtest["traditional_pct_error"].mean(),
    "traditional_mae": backtest["traditional_error"].abs().mean(),
    "ml_mean_pct_error": backtest["ml_pct_error"].mean(),
    "ml_mae": backtest["ml_error"].abs().mean(),
}
pd.Series(summary).to_csv(os.path.join(ROOT, "reports", "revenue_forecast_summary.csv"))
print(backtest[["month","n_candidates","traditional_forecast","ml_forecast","actual_won_revenue"]].round(0).to_string(index=False))
print(f"Traditional mean bias: {summary['traditional_mean_pct_error']:+.1f}% | MAE ${summary['traditional_mae']:,.0f}")
print(f"ML mean bias: {summary['ml_mean_pct_error']:+.1f}% | MAE ${summary['ml_mae']:,.0f}")
# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
plt.figure(figsize=(11, 5))
x = range(len(backtest))
plt.plot(x, backtest["actual_won_revenue"], marker="o", label="Actual realized revenue", linewidth=2, color="black")
plt.plot(x, backtest["traditional_forecast"], marker="s", label="Traditional (Pipeline x Stage Prob)")
plt.plot(x, backtest["ml_forecast"], marker="^", label="ML-driven (Deal Value x Calibrated P-Win)")
plt.xticks(x, backtest["month"], rotation=45, ha="right")
plt.ylabel("Revenue ($)")
plt.title("Monthly Revenue Forecast Backtest: Traditional vs ML vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "revenue_forecast_backtest.png"), dpi=140)
plt.close()
print(f"\nSaved chart -> {FIG_DIR}/revenue_forecast_backtest.png")
