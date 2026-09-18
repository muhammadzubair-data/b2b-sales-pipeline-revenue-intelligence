"""
Sales-cycle survival analysis.

Instead of only asking "will this deal close?", asks "WHEN is it likely to
close?" - modeling time-to-close as a survival/duration problem where OPEN
deals are right-censored (we know they haven't closed yet as of AS_OF_DATE,
but not when/if they will).

Produces:
  - Kaplan-Meier close-time curves overall and by company size band
  - A Cox Proportional Hazards model using deal/engagement covariates
  - Per-open-opportunity probability of closing within 7 / 30 / 60 / 90+ days
  - Flags "stuck" opportunities: open far longer than the model expects for
    deals like them
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import ROOT, PROCESSED_DIR, AS_OF_DATE  # noqa: E402
from data_io import load_table  # noqa: E402

FIG_DIR = os.path.join(ROOT, "reports", "figures")

opportunities = load_table("opportunities")
accounts = load_table("accounts")
sales_reps = load_table("sales_reps")

opp = opportunities.merge(
    accounts[["account_id", "industry", "company_size_band", "engagement_score"]],
    on="account_id", how="left",
).merge(
    sales_reps[["rep_id", "seniority_level"]].rename(columns={"seniority_level": "rep_seniority"}),
    on="rep_id", how="left",
)

# Duration = days from creation to event (close) or to AS_OF_DATE if still open (censored)
opp["duration_days"] = np.where(
    opp["close_status"] == "Open",
    (AS_OF_DATE - opp["created_date"]).dt.days,
    (opp["actual_close_date"] - opp["created_date"]).dt.days,
)
opp["event_observed"] = (opp["close_status"] != "Open").astype(int)  # 1 = closed (event), 0 = censored

print(f"n={len(opp):,} | events (closed)={opp['event_observed'].sum():,} | "
      f"censored (still open)={ (opp['event_observed']==0).sum():,}")

# ----------------------------------------------------------------------------
# Kaplan-Meier: overall + by company size band
# ----------------------------------------------------------------------------
kmf = KaplanMeierFitter()
plt.figure(figsize=(9, 6))
kmf.fit(opp["duration_days"], event_observed=opp["event_observed"], label="Overall")
kmf.plot_survival_function()

km_summary = {}
for band in ["SMB", "Mid-Market", "Enterprise"]:
    sub = opp[opp["company_size_band"] == band]
    kmf_b = KaplanMeierFitter()
    kmf_b.fit(sub["duration_days"], event_observed=sub["event_observed"], label=band)
    kmf_b.plot_survival_function()
    km_summary[band] = kmf_b.median_survival_time_

plt.title("Probability an opportunity is STILL OPEN, by days since creation")
plt.xlabel("Days since opportunity created")
plt.ylabel("Survival probability (still open)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "survival_curve_by_size.png"), dpi=140)
plt.close()

print("\nMedian time-to-close (days) by company size band:")
for band, med in km_summary.items():
    print(f"  {band:12s}: {med:.0f} days")

# ----------------------------------------------------------------------------
# Cox Proportional Hazards model
# ----------------------------------------------------------------------------
cox_df = opp[[
    "duration_days", "event_observed", "deal_value", "discount_requested_pct",
    "num_stakeholders", "engagement_score",
]].copy()
cox_df["log_deal_value"] = np.log1p(cox_df["deal_value"])
cox_df = cox_df.drop(columns=["deal_value"])

for col in ["company_size_band", "industry", "rep_seniority", "source"]:
    dummies = pd.get_dummies(opp[col], prefix=col, drop_first=True)
    cox_df = pd.concat([cox_df, dummies], axis=1)

cox_df = cox_df.astype({c: int for c in cox_df.columns if cox_df[c].dtype == bool})

cph = CoxPHFitter(penalizer=0.05)
cph.fit(cox_df, duration_col="duration_days", event_col="event_observed", show_progress=False)

coef_df = cph.summary[["coef", "exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)
coef_df.to_csv(os.path.join(ROOT, "reports", "survival_cox_coefficients.csv"))
print("\nTop hazard-ratio drivers (exp(coef) > 1 = closes FASTER; < 1 = closes SLOWER):")
print(coef_df.head(10).to_string())

# ----------------------------------------------------------------------------
# Score OPEN opportunities: P(close within 7 / 30 / 60 / 90 days from today)
# ----------------------------------------------------------------------------
open_opp = opp[opp["close_status"] == "Open"].copy().reset_index(drop=True)
open_cox_input = open_opp[["deal_value", "discount_requested_pct", "num_stakeholders", "engagement_score"]].copy()
open_cox_input["log_deal_value"] = np.log1p(open_cox_input["deal_value"])
open_cox_input = open_cox_input.drop(columns=["deal_value"])
for col in ["company_size_band", "industry", "rep_seniority", "source"]:
    dummies = pd.get_dummies(open_opp[col], prefix=col, drop_first=True)
    open_cox_input = pd.concat([open_cox_input, dummies], axis=1)
# align columns to the fitted model's covariates
model_covariates = [c for c in cox_df.columns if c not in ("duration_days", "event_observed")]
for c in model_covariates:
    if c not in open_cox_input.columns:
        open_cox_input[c] = 0
open_cox_input = open_cox_input[model_covariates].astype(float)

t0 = open_opp["duration_days"].values  # days already elapsed (conditioning point)
horizons = {"7_days": 7, "30_days": 30, "60_days": 60, "90_days": 90}

surv_func = cph.predict_survival_function(open_cox_input)  # index = time grid, columns = row idx
time_grid = surv_func.index.values


def survival_at(row_i, t):
    idx = np.searchsorted(time_grid, t)
    idx = min(idx, len(time_grid) - 1)
    return surv_func.iloc[idx, row_i]


for label, h in horizons.items():
    probs = []
    for i in range(len(open_opp)):
        s_now = survival_at(i, t0[i])
        s_future = survival_at(i, t0[i] + h)
        # P(close within h days | still open at t0) = 1 - S(t0+h)/S(t0)
        p = 1 - (s_future / s_now) if s_now > 0 else np.nan
        probs.append(np.clip(p, 0, 1))
    open_opp[f"prob_close_within_{label}"] = probs

# "stuck" flag: already open longer than the size band's median time-to-close
open_opp["expected_median_days"] = open_opp["company_size_band"].map(km_summary)
open_opp["days_open_so_far"] = t0
open_opp["is_stuck"] = open_opp["days_open_so_far"] > open_opp["expected_median_days"]

survival_out_cols = [
    "opportunity_id", "account_id", "rep_id", "company_size_band", "industry",
    "deal_value", "days_open_so_far", "expected_median_days", "is_stuck",
    "prob_close_within_7_days", "prob_close_within_30_days",
    "prob_close_within_60_days", "prob_close_within_90_days",
]
survival_scored = open_opp[survival_out_cols]
survival_scored.to_csv(os.path.join(PROCESSED_DIR, "open_pipeline_survival.csv.gz"),
                        index=False, compression="gzip")

print(f"\nScored time-to-close for {len(survival_scored):,} open opportunities.")
print(f"Flagged as 'stuck' (open longer than typical for their segment): {survival_scored['is_stuck'].sum():,}"
      f" ({100*survival_scored['is_stuck'].mean():.1f}%)")
print("\nSample:")
print(survival_scored.sort_values("prob_close_within_30_days", ascending=False).head(5).to_string(index=False))
