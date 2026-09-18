-- ============================================================================
-- 06. ROLLING PIPELINE SNAPSHOT
-- Monthly new pipeline created, revenue closed-won, and a 3-month rolling
-- average of both, using a window frame (ROWS BETWEEN 2 PRECEDING).
-- ============================================================================

WITH monthly_created AS (
    SELECT
        strftime('%Y-%m', created_date) AS month,
        COUNT(*)                        AS opps_created,
        SUM(deal_value)                 AS pipeline_value_created
    FROM opportunities
    GROUP BY 1
),
monthly_won AS (
    SELECT
        strftime('%Y-%m', close_date) AS month,
        COUNT(*)                      AS deals_won,
        SUM(final_value)              AS revenue_won
    FROM won_deals
    GROUP BY 1
),
combined AS (
    SELECT
        COALESCE(c.month, w.month)                        AS month,
        COALESCE(c.opps_created, 0)                        AS opps_created,
        COALESCE(c.pipeline_value_created, 0)              AS pipeline_value_created,
        COALESCE(w.deals_won, 0)                           AS deals_won,
        COALESCE(w.revenue_won, 0)                         AS revenue_won
    FROM monthly_created c
    FULL OUTER JOIN monthly_won w ON w.month = c.month
    -- requires SQLite >= 3.39 (2022) / any modern warehouse; both have it
)
SELECT
    month,
    opps_created,
    pipeline_value_created,
    deals_won,
    revenue_won,
    ROUND(AVG(revenue_won) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 0)
                                                            AS revenue_won_3mo_rolling_avg,
    ROUND(AVG(pipeline_value_created) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 0)
                                                            AS pipeline_created_3mo_rolling_avg
FROM combined
ORDER BY month;
