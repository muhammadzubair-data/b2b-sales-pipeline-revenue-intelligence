"""
Point-in-time feature engineering for opportunity/lead scoring.

The critical design constraint: P(Deal Won) must be estimated using ONLY
information available AS OF a given scoring date - never information from
after that date (that would be leakage: e.g. using "number of activities in
the deal's full lifetime" to predict a deal that, at scoring time, hadn't
happened yet).

For every CLOSED opportunity we manufacture a synthetic scoring_date sampled
uniformly between its creation and its close, and re-derive every feature
using only activity/pipeline_history rows that occurred on or before that
date. This gives the model a realistic mix of "how deals looked at various
points in their life", not just "how they looked right before closing".

For every OPEN opportunity, scoring_date = AS_OF_DATE (today), which is
exactly the real-world scoring scenario: score the live pipeline right now.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import load_table, PROCESSED_DIR, AS_OF_DATE  # noqa: E402

RNG = np.random.default_rng(7)

print("Loading tables...")
opportunities = load_table("opportunities")
activities = load_table("sales_activities")
pipeline_history = load_table("pipeline_history")
accounts = load_table("accounts")
contacts = load_table("contacts")
sales_reps = load_table("sales_reps")

opportunities = opportunities.merge(
    accounts[["account_id", "industry", "company_size_band", "employees",
              "annual_revenue_usd", "engagement_score", "territory_id"]],
    on="account_id", how="left",
).merge(
    sales_reps[["rep_id", "seniority_level"]].rename(columns={"seniority_level": "rep_seniority"}),
    on="rep_id", how="left",
)

is_closed = opportunities["close_status"] != "Open"
n = len(opportunities)

# ----------------------------------------------------------------------------
# 1. Assign a scoring_date per opportunity
# ----------------------------------------------------------------------------
created = opportunities["created_date"].values
closed_date = opportunities["actual_close_date"].values

life_days = np.where(
    is_closed,
    (pd.Series(closed_date) - pd.Series(created)).dt.days.values,
    (AS_OF_DATE - pd.Series(created)).dt.days.values,
)
life_days = np.clip(life_days, 1, None)

# for closed deals: a random point strictly before close (uniform in (0, life_days))
frac = RNG.uniform(0.15, 0.98, size=n)  # avoid the very first day and the exact close day
offset_days = np.where(is_closed, np.floor(frac * life_days), life_days).astype(int)

scoring_date = pd.Series(created) + pd.to_timedelta(offset_days, unit="D")
scoring_date = scoring_date.where(~pd.Series(is_closed.values), scoring_date)  # closed: created+offset
scoring_date = scoring_date.where(pd.Series(is_closed.values), pd.Series([AS_OF_DATE] * n))  # open: as-of

opportunities["scoring_date"] = scoring_date
opportunities["days_since_created_at_scoring"] = (
    opportunities["scoring_date"] - opportunities["created_date"]
).dt.days

print("Building activity features (as of scoring_date)...")
# ----------------------------------------------------------------------------
# 2. Activity features truncated to <= scoring_date
# ----------------------------------------------------------------------------
scoring_map = opportunities.set_index("opportunity_id")["scoring_date"]
activities = activities.merge(scoring_map.rename("scoring_date"), on="opportunity_id", how="left")
act_valid = activities[activities["activity_date"] <= activities["scoring_date"]].copy()

act_agg = act_valid.groupby("opportunity_id").agg(
    num_activities_to_date=("activity_id", "count"),
    num_calls=("activity_type", lambda s: (s == "Call").sum()),
    num_emails=("activity_type", lambda s: (s == "Email").sum()),
    num_meetings=("activity_type", lambda s: (s == "Meeting").sum()),
    num_demos=("activity_type", lambda s: (s == "Demo").sum()),
    pct_positive_outcome=("outcome", lambda s: (s == "Positive - Advancing").mean()),
    pct_no_response=("outcome", lambda s: (s == "No Response").mean()),
    last_activity_date=("activity_date", "max"),
    first_activity_date=("activity_date", "min"),
).reset_index()

act_agg = act_agg.merge(scoring_map.rename("scoring_date"), on="opportunity_id", how="right")
act_agg["days_since_last_activity"] = (act_agg["scoring_date"] - act_agg["last_activity_date"]).dt.days
act_agg["activity_span_days"] = (act_agg["last_activity_date"] - act_agg["first_activity_date"]).dt.days
fill_cols = ["num_activities_to_date", "num_calls", "num_emails", "num_meetings", "num_demos",
             "pct_positive_outcome", "pct_no_response"]
act_agg[fill_cols] = act_agg[fill_cols].fillna(0)
act_agg["days_since_last_activity"] = act_agg["days_since_last_activity"].fillna(9999)
act_agg["activity_span_days"] = act_agg["activity_span_days"].fillna(0)
act_agg = act_agg.drop(columns=["scoring_date", "last_activity_date", "first_activity_date"])

print("Building stage/pipeline features (as of scoring_date)...")
# ----------------------------------------------------------------------------
# 3. Current stage + days-in-stage as of scoring_date
# ----------------------------------------------------------------------------
ph = pipeline_history.merge(scoring_map.rename("scoring_date"), on="opportunity_id", how="left")
ph_valid = ph[ph["entered_date"] <= ph["scoring_date"]].copy()
ph_valid = ph_valid.sort_values(["opportunity_id", "entered_date"])
current_stage_rows = ph_valid.groupby("opportunity_id").tail(1)

stage_order = {"Qualification": 1, "Needs Analysis": 2, "Proposal": 3, "Negotiation": 4}
current_stage_rows = current_stage_rows.assign(
    stage_number_at_scoring=current_stage_rows["stage"].map(stage_order),
    days_in_current_stage_at_scoring=(
        current_stage_rows["scoring_date"] - current_stage_rows["entered_date"]
    ).dt.days,
)[["opportunity_id", "stage", "stage_number_at_scoring", "days_in_current_stage_at_scoring"]]
current_stage_rows = current_stage_rows.rename(columns={"stage": "stage_at_scoring"})

# ----------------------------------------------------------------------------
# 4. Assemble the full feature table
# ----------------------------------------------------------------------------
feat = opportunities.merge(act_agg, on="opportunity_id", how="left")
feat = feat.merge(current_stage_rows, on="opportunity_id", how="left")
feat["stage_at_scoring"] = feat["stage_at_scoring"].fillna("Qualification")
feat["stage_number_at_scoring"] = feat["stage_number_at_scoring"].fillna(1)
feat["days_in_current_stage_at_scoring"] = feat["days_in_current_stage_at_scoring"].fillna(0)

feat["label_won"] = (feat["close_status"] == "Closed Won").astype(int)
feat["is_closed"] = is_closed.values
feat["log_deal_value"] = np.log1p(feat["deal_value"])

feature_cols = [
    "opportunity_id", "account_id", "rep_id", "product_id", "source",
    "industry", "company_size_band", "employees", "annual_revenue_usd",
    "engagement_score", "rep_seniority", "territory_id",
    "created_date", "scoring_date", "expected_close_date", "actual_close_date", "close_status",
    "deal_value", "log_deal_value", "discount_requested_pct", "num_stakeholders",
    "days_since_created_at_scoring", "stage_at_scoring", "stage_number_at_scoring",
    "days_in_current_stage_at_scoring",
    "num_activities_to_date", "num_calls", "num_emails", "num_meetings", "num_demos",
    "pct_positive_outcome", "pct_no_response", "days_since_last_activity", "activity_span_days",
    "is_closed", "label_won",
]
scoring_dataset = feat[feature_cols].copy()

out_path = os.path.join(PROCESSED_DIR, "scoring_dataset.csv.gz")
scoring_dataset.to_csv(out_path, index=False, compression="gzip")
print(f"\nSaved {len(scoring_dataset):,} rows -> {out_path}")
print(f"  closed (labeled): {scoring_dataset['is_closed'].sum():,}")
print(f"  open (to be scored today): {(~scoring_dataset['is_closed']).sum():,}")
print(f"  base win rate among closed: {scoring_dataset.loc[scoring_dataset.is_closed, 'label_won'].mean():.3f}")
