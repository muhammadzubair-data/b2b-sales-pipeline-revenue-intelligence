# Project 007 Publication Audit

## Status
GitHub-ready after publication-safety corrections.

## Verified strengths
- Synthetic CRM design: 120,000 accounts, 1,000,000 leads, 313,197 opportunities, 1.93M sales activities.
- Point-in-time opportunity scoring features and chronological model evaluation.
- XGBoost calibrated model: ROC-AUC 0.7384, average precision 0.6173, Brier 0.1995.
- Sales-cycle survival analysis includes right-censoring for open opportunities.
- SQL layer contains eight analytical queries with CTE/window-function patterns.
- Power BI-ready star-schema documentation and DAX specification are included.

## Corrections made
1. **Revenue forecast backtest:** the original implementation could reuse non-stage features from a manufactured scoring snapshot later than the historical month-start. The code now reconstructs activities and pipeline stage strictly as of each forecast date. Legacy forecast numbers are no longer headline claims until the corrected backtest is rerun.
2. **Capacity optimization wording:** `$85.0M` versus `$9.4M` is about **9.0x as much** proxy incremental value, equivalent to **+800.9%**. The old wording called this “8x.”
3. **Causal language:** the follow-up uplift is an assumption-driven heuristic (`BASE_UPLIFT`, probability leverage, staleness leverage), not an experimentally estimated treatment effect. README language now states this explicitly.
4. **Publication tests:** six automated checks were added, plus GitHub Actions CI.

## Safe headline claims
- Calibrated XGBoost: ROC-AUC ~0.738 and average precision ~0.617 on the saved temporal evaluation.
- 18,547 open opportunities; fixed capacity of 1,020 follow-ups in the saved prioritization scenario.
- Heuristic expected-incremental-revenue ranking concentrates ~$85.0M proxy incremental value vs ~$9.4M for win-probability ranking, about 9.0x as much, under stated assumptions.
- 4,014 of 18,547 open opportunities (21.6%) are flagged “stuck” using segment median time-to-close.

## Claims to regenerate before using publicly
The old `reports/revenue_forecast_backtest.csv` and `reports/revenue_forecast_summary.csv` were produced by the pre-audit backtest. They are retained only for traceability in the full master, but are excluded from the GitHub-ready package. Run the corrected pipeline to create new forecast metrics before quoting a forecast-error improvement percentage.
