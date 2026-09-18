"""
Builds a Power BI-ready star schema export layer under power_bi/data/.

Design mirrors the "Sales Revenue Command Center":
  Executive Overview -> Sales Funnel -> Pipeline Health -> Revenue Forecast
  -> Lead Scoring -> Deal Velocity -> Sales Rep Performance -> Next Best Actions

Dimension tables are small lookup tables; fact tables carry the metrics.
Everything here is plain, denormalized CSV so it imports cleanly with Power
BI's "Get Data > Text/CSV" with no transformation required, and relationships
can be wired up in Power BI's model view on the *_id keys.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import ROOT, PROCESSED_DIR  # noqa: E402
from data_io import load_table  # noqa: E402

OUT = os.path.join(ROOT, "power_bi", "data")
os.makedirs(OUT, exist_ok=True)

# ---- Dimensions -------------------------------------------------------
accounts = load_table("accounts")
reps = load_table("sales_reps")
territories = load_table("territories")
products = load_table("products")

accounts.to_csv(f"{OUT}/dim_accounts.csv", index=False)
reps.merge(territories, on="territory_id").to_csv(f"{OUT}/dim_sales_reps.csv", index=False)
territories.to_csv(f"{OUT}/dim_territories.csv", index=False)
products.to_csv(f"{OUT}/dim_products.csv", index=False)

# A standalone date dimension spanning the full CRM history + a bit of headroom,
# for Power BI's CALENDAR()-style time intelligence (DAX measures reference this).
dates = pd.DataFrame({"date": pd.date_range("2024-01-01", "2026-12-31", freq="D")})
dates["year"] = dates["date"].dt.year
dates["month"] = dates["date"].dt.month
dates["month_name"] = dates["date"].dt.strftime("%b %Y")
dates["quarter"] = "Q" + dates["date"].dt.quarter.astype(str) + " " + dates["date"].dt.year.astype(str)
dates["year_month"] = dates["date"].dt.strftime("%Y-%m")
dates.to_csv(f"{OUT}/dim_date.csv", index=False)

# ---- Fact: opportunities (the core fact table) -------------------------
opportunities = load_table("opportunities")
scored = pd.read_csv(os.path.join(PROCESSED_DIR, "open_pipeline_scored.csv.gz"), compression="gzip")
fact_opp = opportunities.merge(
    scored[["opportunity_id", "calibrated_win_probability"]], on="opportunity_id", how="left"
)
# for closed deals, win probability is simply the realized outcome (1/0) - no model needed
fact_opp["calibrated_win_probability"] = fact_opp["calibrated_win_probability"].fillna(
    (fact_opp["close_status"] == "Closed Won").astype(float)
)
fact_opp["net_deal_value"] = fact_opp["deal_value"] * (1 - fact_opp["discount_requested_pct"] / 100)
fact_opp["expected_value"] = fact_opp["net_deal_value"] * fact_opp["calibrated_win_probability"]
fact_opp.to_csv(f"{OUT}/fact_opportunities.csv", index=False)

# ---- Fact: activities summary (one row per opportunity, pre-aggregated for BI) ----
activities = load_table("sales_activities")
act_summary = activities.groupby("opportunity_id").agg(
    total_activities=("activity_id", "count"),
    last_activity_date=("activity_date", "max"),
    pct_positive=("outcome", lambda s: (s == "Positive - Advancing").mean()),
).reset_index()
act_summary.to_csv(f"{OUT}/fact_activity_summary.csv", index=False)

# ---- Fact: revenue forecast (traditional vs ML backtest) ---------------
forecast_path = os.path.join(ROOT, "reports", "revenue_forecast_backtest.csv")
if os.path.exists(forecast_path):
    pd.read_csv(forecast_path).to_csv(f"{OUT}/fact_revenue_forecast.csv", index=False)

# ---- Fact: next best action worklist ------------------------------------
nba_path = os.path.join(ROOT, "reports", "next_best_action.csv")
if os.path.exists(nba_path):
    pd.read_csv(nba_path).to_csv(f"{OUT}/fact_next_best_action.csv", index=False)

# ---- Fact: won/lost deal marts (used directly by the Sales Funnel & Lost
#      Reasons pages) ------------------------------------------------------
load_table("won_deals").to_csv(f"{OUT}/fact_won_deals.csv", index=False)
load_table("lost_deals").to_csv(f"{OUT}/fact_lost_deals.csv", index=False)
load_table("pipeline_history").to_csv(f"{OUT}/fact_pipeline_history.csv", index=False)

print("Power BI export layer written to power_bi/data/:")
for f in sorted(os.listdir(OUT)):
    size_mb = os.path.getsize(os.path.join(OUT, f)) / 1e6
    print(f"  {f:35s} {size_mb:7.1f} MB")
