# DAX Measures — Sales Revenue Command Center

Paste these into a Power BI measures table (`_Measures`) after importing the
CSVs from `power_bi/data/` (unzip the `.csv.gz` files first — Power BI's
Text/CSV connector expects plain CSV). Relationships to wire up in Model
view: `fact_opportunities[account_id] -> dim_accounts[account_id]`,
`fact_opportunities[rep_id] -> dim_sales_reps[rep_id]`,
`fact_opportunities[product_id] -> dim_products[product_id]`,
`fact_opportunities[created_date] -> dim_date[date]` (mark `dim_date` as a
Date Table).

## Pipeline & Funnel

```
Total Opportunities = COUNTROWS(fact_opportunities)

Open Pipeline Value =
CALCULATE(
    SUM(fact_opportunities[deal_value]),
    fact_opportunities[close_status] = "Open"
)

Weighted Pipeline (Traditional) =
SUMX(
    FILTER(fact_opportunities, fact_opportunities[close_status] = "Open"),
    fact_opportunities[deal_value] *
    SWITCH(
        fact_opportunities[stage],
        "Qualification", 0.10,
        "Needs Analysis", 0.25,
        "Proposal", 0.50,
        "Negotiation", 0.75,
        0.10
    )
)

Weighted Pipeline (ML) =
SUMX(
    FILTER(fact_opportunities, fact_opportunities[close_status] = "Open"),
    fact_opportunities[deal_value] * fact_opportunities[calibrated_win_probability]
        * (1 - fact_opportunities[discount_requested_pct] / 100)
)

Win Rate % =
DIVIDE(
    CALCULATE(COUNTROWS(fact_opportunities), fact_opportunities[close_status] = "Closed Won"),
    CALCULATE(COUNTROWS(fact_opportunities), fact_opportunities[close_status] IN {"Closed Won","Closed Lost"})
)

Avg Deal Size (Won) =
CALCULATE(AVERAGE(fact_opportunities[net_deal_value]), fact_opportunities[close_status] = "Closed Won")

Avg Sales Cycle (Days) =
CALCULATE(AVERAGE(fact_opportunities[sales_cycle_days_planned]), fact_opportunities[close_status] = "Closed Won")
```

## Revenue & Forecasting

```
Revenue Closed Won =
CALCULATE(SUM(fact_opportunities[net_deal_value]), fact_opportunities[close_status] = "Closed Won")

Revenue Closed Won (MTD) =
TOTALMTD([Revenue Closed Won], dim_date[date])

Revenue Closed Won (YTD) =
TOTALYTD([Revenue Closed Won], dim_date[date])

Forecast Error % (Traditional) =
DIVIDE(
    SUM(fact_revenue_forecast[traditional_error]),
    SUM(fact_revenue_forecast[actual_won_revenue])
)

Forecast Error % (ML) =
DIVIDE(
    SUM(fact_revenue_forecast[ml_error]),
    SUM(fact_revenue_forecast[actual_won_revenue])
)
```

## Lead Scoring & Next Best Action

```
Avg Win Probability (Open Pipeline) =
CALCULATE(AVERAGE(fact_opportunities[calibrated_win_probability]), fact_opportunities[close_status] = "Open")

High Priority Deal Count =
CALCULATE(COUNTROWS(fact_next_best_action), fact_next_best_action[priority] = "High")

Expected Incremental Revenue (High Priority) =
CALCULATE(
    SUM(fact_next_best_action[expected_incremental_revenue]),
    fact_next_best_action[priority] = "High"
)

Stale Deals (14+ Days No Activity) =
CALCULATE(
    COUNTROWS(fact_activity_summary),
    DATEDIFF(fact_activity_summary[last_activity_date], TODAY(), DAY) >= 14
)
```

## Sales Rep Performance

```
Revenue per Rep =
DIVIDE([Revenue Closed Won], DISTINCTCOUNT(fact_opportunities[rep_id]))

Rep Rank (Revenue) =
RANKX(ALL(dim_sales_reps[rep_id]), CALCULATE([Revenue Closed Won]), , DESC)

Quota Attainment % =
DIVIDE(
    [Revenue Closed Won] / 4,   -- quarterly quota approximated over a full year view
    SUM(dim_sales_reps[quota_target_quarterly])
)
```
