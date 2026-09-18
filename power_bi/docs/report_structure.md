# Power BI — Sales Revenue Command Center — Report Structure

Import order: unzip everything in `power_bi/data/*.csv.gz` to plain `.csv`,
then Get Data > Text/CSV for each file. Build the relationships described in
`dax_measures.md`, mark `dim_date` as a Date Table, then create the measures
in that file inside a dedicated `_Measures` table.

## Page 1 — Executive Overview
KPI cards: Open Pipeline Value, Weighted Pipeline (ML), Win Rate %, Revenue
Closed Won (YTD), Avg Deal Size (Won). A trend line of monthly Revenue Closed
Won next to Weighted Pipeline (ML) so leadership sees "what we've booked" vs
"what's coming." A small multiples bar of pipeline value by `company_size_band`.

## Page 2 — Sales Funnel
Funnel visual: Lead -> Opportunity -> Needs Analysis+ -> Proposal+ ->
Negotiation+ -> Closed Won (drives from `fact_opportunities[stage]` /
`stage_reached_before_close`). Conversion % callouts between each stage.
Slicer: lead `source`.

## Page 3 — Pipeline Health
Stacked bar of open pipeline by stage and company size band. A table of
"stale" deals (`fact_activity_summary[last_activity_date]` > 14 days old)
joined to `fact_opportunities`. Matrix of win rate % by industry x company
size band (from `fact_opportunities`).

## Page 4 — Revenue Forecast
Line chart from `fact_revenue_forecast`: Actual Won Revenue vs Traditional
Forecast vs ML Forecast, by month. Card visuals for [Forecast Error %
(Traditional)] and [Forecast Error % (ML)] side by side — this is the
"is CRM forecasting systematically optimistic?" page.

## Page 5 — Lead Scoring
Histogram of `calibrated_win_probability` across the open pipeline.
Scatter: win probability (x) vs deal_value (y), sized by expected_value,
colored by company_size_band. A table of the top 20 open deals by
expected_value.

## Page 6 — Deal Velocity
Bar chart of average days-in-stage (mirrors `sql/queries/03_stage_velocity.sql`).
Box-plot-style visual (or a ribbon chart) of sales-cycle-days distribution by
company size band. A KPI showing the % of deals flagged `is_stuck` from the
survival model output.

## Page 7 — Sales Rep Performance
Table from `dim_sales_reps` + `fact_opportunities`: rep name, territory,
win rate, revenue closed, [Quota Attainment %], [Rep Rank (Revenue)].
Slicer by territory/region. Conditional formatting on Quota Attainment %.

## Page 8 — Next Best Actions
Table straight from `fact_next_best_action`: opportunity, win probability,
expected revenue, expected incremental revenue, deal risk, main risk,
recommended action, priority. Default sort by `capacity_rank`. Slicer on
`priority` and `recommended_action` so a rep can filter to just their
worklist for the day.
