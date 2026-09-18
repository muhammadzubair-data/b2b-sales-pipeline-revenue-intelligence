"""
Next Best Action generator.

Combines the lead-scoring model, the survival/time-to-close model, and the
capacity-optimization ranking into a single actionable row per open
opportunity - turning three separate models into one decision a sales
manager can actually read and act on, e.g.:

    Opportunity: Enterprise Software Renewal
    Win Probability: 72%
    Expected Revenue: $84,000
    Deal Risk: Medium
    Main Risk: 19 days without stakeholder response
    Recommended Action: Executive follow-up
    Priority: High
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import ROOT, PROCESSED_DIR  # noqa: E402
from data_io import load_table  # noqa: E402

worklist = pd.read_csv(os.path.join(PROCESSED_DIR, "capacity_optimized_worklist.csv.gz"), compression="gzip")
accounts = load_table("accounts")
products = load_table("products")

df = worklist.merge(accounts[["account_id", "account_name"]], on="account_id", how="left") \
             .merge(products[["product_id", "product_name"]], on="product_id", how="left")

df["opportunity_label"] = df["account_name"] + " — " + df["product_name"]

# ----------------------------------------------------------------------------
# Deal Risk tier
# ----------------------------------------------------------------------------
conditions_risk = [
    (df["is_stuck"]) & (df["days_since_last_activity"] > 21),
    (df["is_stuck"]) | (df["days_since_last_activity"] > 14),
    (df["days_since_last_activity"] > 7),
]
df["deal_risk"] = np.select(conditions_risk, ["High", "Medium", "Medium"], default="Low")

# ----------------------------------------------------------------------------
# Main risk driver (plain-language, single biggest factor)
# ----------------------------------------------------------------------------
def main_risk(row):
    if row["days_since_last_activity"] > 14:
        return f"{int(row['days_since_last_activity'])} days without stakeholder response"
    if row["is_stuck"]:
        return f"In-stage {int(row['days_open_so_far'])} days, longer than the {int(row['expected_median_days'])}-day norm for its segment"
    if row["num_activities_to_date"] <= 1:
        return "Minimal engagement recorded since creation"
    if row["calibrated_win_probability"] < 0.25:
        return "Low overall win probability given deal profile"
    return "On track — no material risk flagged"


df["main_risk"] = df.apply(main_risk, axis=1)

# ----------------------------------------------------------------------------
# Recommended action
# ----------------------------------------------------------------------------
def recommend_action(row):
    if row["days_since_last_activity"] > 21 and row["calibrated_win_probability"] > 0.35:
        return "Executive follow-up"
    if row["days_since_last_activity"] > 14:
        return "Re-engagement call / email"
    if row["stage_at_scoring"] in ("Proposal", "Negotiation") and row["calibrated_win_probability"] > 0.5:
        return "Schedule closing/negotiation call"
    if row["stage_at_scoring"] == "Qualification" and row["num_activities_to_date"] <= 1:
        return "Initial discovery outreach"
    if row["calibrated_win_probability"] < 0.15:
        return "De-prioritize / consider disqualifying"
    return "Continue standard cadence"


df["recommended_action"] = df.apply(recommend_action, axis=1)

# ----------------------------------------------------------------------------
# Priority (driven by the capacity-optimization ranking already computed)
# ----------------------------------------------------------------------------
n = len(df)
df = df.sort_values("capacity_rank")
df["priority"] = pd.cut(
    df["capacity_rank"], bins=[0, n * 0.05, n * 0.20, n],
    labels=["High", "Medium", "Low"],
)

output_cols = [
    "opportunity_id", "opportunity_label", "account_name", "rep_id",
    "company_size_band" if "company_size_band" in df.columns else "industry",
    "calibrated_win_probability", "expected_revenue", "expected_incremental_revenue",
    "deal_risk", "main_risk", "recommended_action", "priority", "capacity_rank",
    "in_recommended_worklist",
]
output_cols = [c for c in output_cols if c in df.columns]
next_best_action = df[output_cols].rename(columns={
    "calibrated_win_probability": "win_probability",
})

out_path = os.path.join(ROOT, "reports", "next_best_action.csv")
next_best_action.to_csv(out_path, index=False)
print(f"Saved {len(next_best_action):,} rows -> {out_path}\n")

print("Priority distribution:")
print(next_best_action["priority"].value_counts().to_string())

print("\nRecommended action distribution:")
print(next_best_action["recommended_action"].value_counts().to_string())

print("\nSample Next Best Action cards (top 5 by priority):")
for _, r in next_best_action.head(5).iterrows():
    print(f"""
Opportunity: {r['opportunity_label']}
Win Probability: {r['win_probability']*100:.0f}%
Expected Revenue: ${r['expected_revenue']:,.0f}
Deal Risk: {r['deal_risk']}
Main Risk: {r['main_risk']}
Recommended Action: {r['recommended_action']}
Priority: {r['priority']}""")
