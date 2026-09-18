"""
Sales capacity optimization - the flagship analytical component.

The problem: reps can meaningfully follow up with only a fraction of the
open pipeline. Simply ranking by win probability, or even by expected
revenue (deal_value x P(win)), is not necessarily optimal - a deal that
will close anyway with no further effort gets the same priority as one
where a timely follow-up could be the deciding factor. What actually
matters for capacity allocation is the INCREMENTAL revenue a follow-up
would produce:

    Expected Incremental Revenue = Deal Value x Expected Follow-Up Uplift

Since we don't have a randomized follow-up experiment to estimate a true
causal uplift, we build a defensible, transparent uplift PROXY grounded in
two well-established sales patterns used here as modeling assumptions:
  1. Intervention leverage is highest for deals sitting in the "undecided"
     probability band (~mid-range P(win)) - a deal that's nearly certain to
     win or lose is far less movable by one more touch.
  2. Leverage is higher for deals that have gone quiet (long
     days-since-last-activity / flagged "stuck" by the survival model) -
     these are exactly the deals where a nudge is most likely to matter,
     versus a deal already being worked hard.

This uplift proxy is clearly labeled as an assumption-driven heuristic
throughout (not a causal estimate) - in a real deployment it would be
learned from an actual randomized/quasi-experimental follow-up rollout.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import ROOT, PROCESSED_DIR  # noqa: E402

FIG_DIR = os.path.join(ROOT, "reports", "figures")

scored = pd.read_csv(os.path.join(PROCESSED_DIR, "open_pipeline_scored.csv.gz"), compression="gzip")
survival = pd.read_csv(os.path.join(PROCESSED_DIR, "open_pipeline_survival.csv.gz"), compression="gzip")

pipeline = scored.merge(
    survival[["opportunity_id", "days_open_so_far", "expected_median_days", "is_stuck",
              "prob_close_within_30_days", "prob_close_within_90_days"]],
    on="opportunity_id", how="left",
)

p = pipeline["calibrated_win_probability"].values
days_stale = pipeline["days_since_last_activity"].clip(upper=180).values

# --- Uplift proxy -----------------------------------------------------------
BASE_UPLIFT = 0.15  # max plausible win-probability lift from an effective, well-timed follow-up
probability_leverage = 4 * p * (1 - p)                      # peaks at p=0.5, in [0, 1]
staleness_leverage = 1 / (1 + np.exp(-(days_stale - 14) / 10))  # sigmoid, higher when quiet
expected_uplift_prob = BASE_UPLIFT * probability_leverage * staleness_leverage

pipeline["expected_uplift_probability"] = expected_uplift_prob
pipeline["expected_revenue"] = (pipeline["deal_value"] * p
                                 * (1 - pipeline["discount_requested_pct"] / 100))
pipeline["expected_incremental_revenue"] = (pipeline["deal_value"] * expected_uplift_prob
                                             * (1 - pipeline["discount_requested_pct"] / 100))

# ----------------------------------------------------------------------------
# Compare 3 prioritization strategies under a fixed rep-capacity constraint
# ----------------------------------------------------------------------------
N_TOTAL = len(pipeline)
CAPACITY = max(int(N_TOTAL * 0.055), 500)  # ~5.5% of open pipeline can be meaningfully worked, mirrors the
                                            # spec's "10,000 active / 1,000 can be followed up" ratio

strategies = {
    "naive_win_probability": pipeline.sort_values("calibrated_win_probability", ascending=False),
    "expected_revenue": pipeline.sort_values("expected_revenue", ascending=False),
    "expected_incremental_revenue": pipeline.sort_values("expected_incremental_revenue", ascending=False),
}

summary_rows = []
for name, ranked in strategies.items():
    top = ranked.head(CAPACITY)
    summary_rows.append({
        "strategy": name,
        "total_expected_revenue_captured": top["expected_revenue"].sum(),
        "total_expected_incremental_revenue_captured": top["expected_incremental_revenue"].sum(),
        "avg_win_probability_of_selected": top["calibrated_win_probability"].mean(),
        "pct_selected_already_likely_to_win_anyway (p>0.75)": (top["calibrated_win_probability"] > 0.75).mean() * 100,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(ROOT, "reports", "capacity_strategy_comparison.csv"), index=False)
print(f"Open pipeline: {N_TOTAL:,} opportunities | rep capacity assumed: {CAPACITY:,} follow-ups\n")
display_summary = summary.copy()
for col in display_summary.columns:
    if "revenue" in col:
        display_summary[col] = display_summary[col].round(0).map(lambda v: f"${v:,.0f}")
    elif "probability" in col or "pct" in col:
        display_summary[col] = display_summary[col].round(3)
print(display_summary.to_string(index=False))

incr_naive = summary.loc[summary.strategy == "naive_win_probability",
                          "total_expected_incremental_revenue_captured"].iloc[0]
incr_optimal = summary.loc[summary.strategy == "expected_incremental_revenue",
                            "total_expected_incremental_revenue_captured"].iloc[0]
lift_pct = 100 * (incr_optimal / incr_naive - 1)
print(f"\n=> Prioritizing by expected INCREMENTAL revenue captures {lift_pct:+.1f}% more incremental "
      f"revenue with the SAME rep capacity than naive win-probability ranking.")

# ----------------------------------------------------------------------------
# Save the recommended (expected-incremental-revenue-ranked) worklist
# ----------------------------------------------------------------------------
ranked_worklist = strategies["expected_incremental_revenue"].reset_index(drop=True)
ranked_worklist["capacity_rank"] = np.arange(1, len(ranked_worklist) + 1)
ranked_worklist["in_recommended_worklist"] = ranked_worklist["capacity_rank"] <= CAPACITY
ranked_worklist.to_csv(os.path.join(PROCESSED_DIR, "capacity_optimized_worklist.csv.gz"),
                        index=False, compression="gzip")

plt.figure(figsize=(8, 5))
plt.bar(summary["strategy"], summary["total_expected_incremental_revenue_captured"])
plt.ylabel("Expected incremental revenue captured ($)")
plt.title(f"Incremental revenue captured under {CAPACITY:,}-deal rep capacity, by prioritization strategy")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "capacity_strategy_comparison.png"), dpi=140)
plt.close()
print(f"\nSaved worklist ({len(ranked_worklist):,} rows) and comparison chart.")
