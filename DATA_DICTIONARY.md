# Data Dictionary

All raw tables live in `data/raw/*.csv.gz` (gzip-compressed CSV — read with
`pandas.read_csv(path, compression="gzip")`, or via the `load_table()` helper
in `scripts/data_io.py`, which also parses date columns automatically).

## territories
| column | type | description |
|---|---|---|
| territory_id | int | primary key |
| territory_name | str | e.g. "Territory-07" |
| region | str | North America / EMEA / APAC / LATAM |
| country | str | a representative country within the region |

## sales_reps
| column | type | description |
|---|---|---|
| rep_id | int | primary key |
| rep_name | str | |
| territory_id | int | FK -> territories |
| seniority_level | str | Junior / Mid / Senior / Principal |
| hire_date | date | |
| quota_target_quarterly | float | quarterly quota target, USD |

## products
| column | type | description |
|---|---|---|
| product_id | int | primary key |
| product_name | str | |
| category | str | Platform License / Add-on Module / Professional Services / Support & Success / Data & Analytics |
| list_price | float | USD |

## accounts
| column | type | description |
|---|---|---|
| account_id | int | primary key |
| account_name | str | |
| industry | str | one of 10 industries |
| company_size_band | str | SMB / Mid-Market / Enterprise |
| employees | int | |
| annual_revenue_usd | float | |
| territory_id | int | FK -> territories |
| created_date | date | account creation date |
| engagement_score | float | 0–100, synthetic proxy for how responsive the account tends to be |

## contacts
| column | type | description |
|---|---|---|
| contact_id | int | primary key |
| account_id | int | FK -> accounts |
| contact_name | str | |
| title | str | Individual Contributor / Buyer-Manager / Director / VP / C-Level |
| seniority_score | int | 0–4, derived from title |
| email | str | synthetic |
| created_date | date | |

## leads
| column | type | description |
|---|---|---|
| lead_id | int | primary key |
| account_id | int | FK -> accounts |
| contact_id | int | FK -> contacts |
| source | str | Inbound - Web / Outbound - Cold Call / Outbound - Email / Referral / Partner Channel / Trade Show-Event |
| created_date | date | |
| lead_score | float | 1–99, synthetic pre-qualification score |
| status | str | Converted / Disqualified / Nurturing |
| converted_to_opportunity | bool | whether this lead became an opportunity |

## opportunities
| column | type | description |
|---|---|---|
| opportunity_id | int | primary key |
| account_id, contact_id, lead_id, rep_id, product_id | int | FKs |
| source | str | inherited from the originating lead |
| created_date | date | |
| deal_value | float | list-price-based deal value before discount, USD |
| discount_requested_pct | float | 0–45 |
| num_stakeholders | int | number of people involved in the buying decision |
| expected_close_date | date | originally scheduled close date |
| actual_close_date | date | actual close date (null if still open) |
| stage | str | current/final funnel stage, or Closed Won/Lost |
| stage_reached_before_close | str | last funnel stage reached (useful for lost-deal analysis) |
| close_status | str | Open / Closed Won / Closed Lost |
| lost_reason | str | populated only for Closed Lost |
| sales_cycle_days_planned | int | the deal's total cycle length (realized or, if open, the length it was generated to eventually take) |

## pipeline_history
One row per funnel stage an opportunity passed through.
| column | type | description |
|---|---|---|
| history_id | int | primary key |
| opportunity_id | int | FK |
| stage | str | Qualification / Needs Analysis / Proposal / Negotiation |
| entered_date, exited_date | date | exited_date is null if the opportunity is still in that stage |
| duration_days | float | null if still in progress |
| still_in_stage | bool | true for the current, unfinished stage of an open opportunity |
| days_in_stage_so_far | float | populated only when still_in_stage is true |

## sales_activities
| column | type | description |
|---|---|---|
| activity_id | int | primary key |
| opportunity_id, rep_id | int | FKs |
| activity_type | str | Call / Email / Meeting / Demo / Proposal Discussion |
| activity_date | date | |
| duration_minutes | int | 0 for non-timed activity types (email) |
| outcome | str | Positive - Advancing / Neutral / Negative - Objection / No Response |

## quotes
| column | type | description |
|---|---|---|
| quote_id | int | primary key |
| opportunity_id, product_id | int | FKs |
| quote_date | date | |
| quoted_price | float | after discount |
| discount_pct | float | |
| revision_number | int | 1 = original quote, 2 = a subsequent revision |
| status | str | Accepted / Rejected / Pending / Superseded |

## won_deals / lost_deals
Denormalized convenience marts, one row per closed opportunity of that
outcome, joining in the final value/lost reason/stage-at-loss/sales cycle
length for quick BI consumption without re-deriving from `opportunities`.

---

## Derived / processed files (`data/processed/`)
- **scoring_dataset.csv.gz** — the point-in-time feature table used to train
  and evaluate the lead-scoring model (see `features/build_features.py`).
  Includes both closed (labeled) and open (unlabeled) opportunities.
- **open_pipeline_scored.csv.gz** — every open opportunity with its
  calibrated win probability from the best model.
- **open_pipeline_survival.csv.gz** — every open opportunity with
  probability of closing within 7/30/60/90 days and a "stuck" flag.
- **capacity_optimized_worklist.csv.gz** — the open pipeline ranked by
  expected incremental revenue, with a capacity cutoff flag.

## Reports (`reports/`)
- **lead_scoring_metrics.csv** — AUC/Brier/LogLoss/AvgPrecision per model.
- **feature_importance.csv** — top drivers of the winning model.
- **revenue_forecast_backtest.csv** / **revenue_forecast_summary.csv** —
  monthly traditional-vs-ML forecast backtest and its summary bias/MAE.
- **survival_cox_coefficients.csv** — Cox PH hazard ratios.
- **capacity_strategy_comparison.csv** — the 3-strategy capacity allocation
  comparison.
- **next_best_action.csv** — the full actionable worklist for every open
  opportunity.
- **figures/** — every chart referenced in the README.
