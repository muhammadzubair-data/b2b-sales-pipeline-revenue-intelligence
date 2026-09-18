# B2B Sales Pipeline, Lead Scoring & Revenue Forecasting Analytics

**Project 007** in a portfolio series spanning e-commerce, SaaS, retail,
finance, CX, and healthcare analytics.

| # | Domain | Main capability |
|---|---|---|
| 001 | E-Commerce | Profitability Analytics |
| 002 | SaaS | Churn + Uplift Modeling |
| 003 | Retail | Forecasting + Inventory Optimization |
| 004 | Finance | Fraud + Anomaly Detection |
| 005 | Customer Experience | NLP + Text Analytics |
| 006 | Healthcare | Simulation + Capacity Optimization |
| **007** | **B2B Sales** | **Lead Scoring + Revenue Intelligence** |

## The business problem

> Which opportunities are most likely to close, how much revenue can we
> realistically expect, why are deals being lost, and where should the sales
> team spend its limited time?

## Flagship result

> **Under a fixed 1,020-opportunity review capacity, an assumption-driven
> expected-incremental-revenue heuristic concentrated about 9.0x as much
> proxy incremental value as ranking by win probability alone ($85.0M vs
> $9.4M). This is a prioritization simulation, not a causal uplift estimate.
> The project also includes a leakage-safe rolling revenue-forecast backtest
> implementation for reproducible comparison with traditional CRM forecasts.**

---

## 1. Dataset

A synthetic but realistically-correlated relational CRM dataset (see
`DATA_DICTIONARY.md` for full schema):

| Table | Rows |
|---|---:|
| accounts | 120,000 |
| contacts | 240,116 |
| leads | 1,000,000 |
| **opportunities** | **313,197** |
| pipeline_history | 1,147,761 |
| sales_activities | 1,931,436 |
| quotes | 394,323 |
| won_deals | 119,024 |
| lost_deals | 175,626 |

Company size, industry, engagement, discounting, activity intensity, and
rep seniority are all embedded as genuine drivers of win probability and
sales-cycle length — not decorative columns — so the downstream models have
real signal to find. The GitHub portfolio package includes compact samples under `data/sample/`.
The full synthetic raw tables are intentionally not versioned because they
are reproducible. Generate them with `python scripts/generate_data.py`; the
SQLite database, processed datasets and full Power BI exports are then rebuilt
by their respective scripts.

## 2. Sales funnel

`Lead -> Opportunity -> Needs Analysis -> Proposal -> Negotiation -> Won/Lost`

| Stage | Count | % of leads | % of previous stage |
|---|---:|---:|---:|
| Lead | 1,000,000 | 100.0% | — |
| Opportunity | 313,197 | 31.3% | 31.3% |
| Needs Analysis+ | 312,383 | 31.2% | 99.7% |
| Proposal+ | 308,366 | 30.8% | 98.7% |
| Negotiation+ | 213,815 | 21.4% | 69.3% |
| Closed Won | 119,024 | 11.9% | 55.7% |

The steepest drop is Proposal -> Negotiation (30.8% -> 21.4%): almost a
third of proposed deals never reach negotiation. See
`sql/queries/01_funnel_conversion.sql` and `04_lost_deal_analysis.sql` —
the top lost-deal reasons (Champion Left Company, Pricing Too High, No
Decision/Timing) are concentrated almost exactly at the Proposal/
Negotiation boundary, each representing >$3B in lost-deal value in this
dataset.

## 3. Lead / opportunity scoring — P(Deal Won)

Logistic Regression, Random Forest, and XGBoost, compared under a **strict
temporal split** (train on 2024–2025 deals, validate, test on the most
recent months only — never a random shuffle, which would leak future
market conditions into the past) with isotonic probability calibration.

| Model | ROC-AUC | Brier Score | Log Loss | Avg Precision |
|---|---:|---:|---:|---:|
| **XGBoost (calibrated)** | **0.738** | **0.200** | **0.574** | **0.617** |
| Random Forest (calibrated) | 0.725 | 0.204 | 0.595 | 0.607 |
| Logistic Regression | 0.721 | 0.206 | 0.598 | 0.611 |

XGBoost wins and is well-calibrated (see `reports/figures/calibration_curve.png`).
Top drivers: current funnel stage, account engagement score, company size
band, and rep seniority — see `reports/feature_importance.csv`.

## 4. Revenue forecasting — Traditional vs ML

**Traditional:** `Pipeline Value x Stage Probability`.  
**ML-driven:** `Deal Value x Calibrated P(Win)`.

The original project artifact reported a large ML forecasting advantage, but
a publication audit found that its historical backtest could reuse activity
features from a manufactured scoring snapshot later than the month being
forecast. The implementation in `models/revenue_forecasting.py` has therefore
been rewritten to reconstruct activity and stage features strictly as of each
historical month-start.

**Publication rule:** the legacy forecast numbers are retained only as
development history and are not used as headline portfolio claims. Re-run
`python models/revenue_forecasting.py` (or `python run_all.py`) to regenerate
the audited rolling backtest from the synthetic raw tables.

This correction is intentional: a forecasting result is only useful if its
historical evaluation uses information that was genuinely available at the
forecast date.

## 5. Sales-cycle survival analysis

Kaplan-Meier + Cox Proportional Hazards on time-to-close, treating open
deals as right-censored (`models/survival_analysis.py`).

Median time-to-close by segment:

| Segment | Median days to close |
|---|---:|
| SMB | 45 |
| Mid-Market | 81 |
| Enterprise | 134 |

Every open opportunity is scored with a probability of closing within
7 / 30 / 60 / 90 days, and flagged **"stuck"** if it's already been open
longer than its segment's median. **21.6% of the current open pipeline
(4,014 of 18,547 opportunities) is flagged stuck** — see
`data/processed/open_pipeline_survival.csv.gz`.

## 6. Sales capacity optimization — the flagship analysis

The scenario the whole project is built around: **18,547 open
opportunities, but reps can meaningfully follow up on only ~1,020 of them.**
Three ways to choose which 1,020, compared on how much *expected
incremental* revenue (i.e., revenue a follow-up would actually help
produce — see the uplift-proxy assumptions documented in
`models/capacity_optimization.py`) each strategy captures:

| Strategy | Avg win probability selected | Incremental revenue captured |
|---|---:|---:|
| Naive: rank by win probability | 75.8% (78% already "sure things") | $9.4M |
| Rank by expected revenue (value x P-win) | 53.6% | $72.5M |
| **Rank by expected incremental revenue** | **47.2% (2.5% "sure things")** | **$85.0M** |

**The heuristic ranking concentrates about 9.0x as much proxy incremental
value ($85.0M vs $9.4M; +800.9%) as naive win-probability ranking under the
same 1,020-opportunity capacity.** This is an assumption-driven scenario,
not observed or causally identified revenue. The uplift proxy is deliberately
transparent and should be replaced by randomized or quasi-experimental uplift
estimation in a production deployment.

## 7. Next Best Action

Every open opportunity gets one actionable card combining win probability,
expected (incremental) revenue, deal risk, and a recommended action —
`reports/next_best_action.csv` (`models/next_best_action.py`). Example:

```
Opportunity: Crestline Ltd. 107221 — Platform-4
Win Probability: 53%
Expected Revenue: $2,270,201
Deal Risk: High
Main Risk: 35 days without stakeholder response
Recommended Action: Executive follow-up
Priority: High
```

Priority distribution across the open pipeline: 927 High, 2,782 Medium,
14,838 Low. Recommended-action mix: 10,302 Continue standard cadence,
5,551 Schedule closing/negotiation call, 1,374 Re-engagement outreach,
880 Executive follow-up, 419 De-prioritize/disqualify, 21 Initial discovery.

## 8. SQL analytics layer

Eight portable ANSI-SQL scripts (CTEs + window functions) under
`sql/queries/`, validated against a local SQLite build of the dataset:

1. `01_funnel_conversion.sql` — stage-over-stage conversion rates
2. `02_pipeline_value_and_winrate.sql` — win rate & deal size by segment
3. `03_stage_velocity.sql` — time-in-stage, `% of cycle` per stage
4. `04_lost_deal_analysis.sql` — lost reasons ranked by revenue at risk
5. `05_rep_territory_performance.sql` — rep/territory rankings (`RANK()`, `NTILE()`)
6. `06_rolling_pipeline_snapshot.sql` — monthly trend + 3-month rolling average
7. `07_cohort_analysis.sql` — lead-source cohorts tracked to revenue-per-lead
8. `08_stage_transition_matrix.sql` — `LEAD()`-based stage-to-stage transitions

Rebuild and rerun: `python sql/build_db.py && python sql/run_queries.py`.

## 9. Power BI — Sales Revenue Command Center

`power_bi/docs/` contains the star-schema/report specification and DAX
measures. The full `power_bi/data/` export is reproducibly generated by
`python power_bi/build_exports.py` after creating the synthetic dataset. `power_bi/docs/dax_measures.md`
and `power_bi/docs/report_structure.md` spec out the 8-page report:
*Executive Overview -> Sales Funnel -> Pipeline Health -> Revenue Forecast
-> Lead Scoring -> Deal Velocity -> Sales Rep Performance -> Next Best
Actions.*

## Project structure

```
├── scripts/            data generation + shared IO helpers
├── data/raw/           gzipped CSV source tables
├── data/processed/     scored/derived datasets (features, capacity worklist, etc.)
├── sql/                SQLite build script, .sql analytics queries, run_queries.py
├── features/           point-in-time feature engineering (leakage-safe)
├── models/             lead scoring, revenue forecasting, survival analysis,
│                       capacity optimization, next best action
├── reports/            metrics, backtest results, next_best_action.csv, figures/
├── power_bi/           star-schema export + DAX measures + report structure
├── run_all.py          reproduces the entire pipeline end-to-end
├── requirements.txt
└── DATA_DICTIONARY.md
```

## Reproducing this project

```bash
pip install -r requirements.txt
python run_all.py
```

Takes a few minutes; data generation is the slowest step (fully vectorized
numpy/pandas, but still ~300K+ opportunities and ~2M activities).

## Tech stack

Python (numpy, pandas, scikit-learn, XGBoost, lifelines, matplotlib),
SQL (SQLite for local validation; queries are portable ANSI SQL for
Snowflake/Postgres/SQL Server), Power BI (DAX, star schema modeling).
