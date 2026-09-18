-- ============================================================================
-- 05. SALES REP & TERRITORY PERFORMANCE RANKING
-- Ranks reps within their own territory (window PARTITION) and overall, on
-- win rate, revenue closed, and average sales cycle.
-- ============================================================================

WITH rep_stats AS (
    SELECT
        r.rep_id,
        r.rep_name,
        r.seniority_level,
        r.territory_id,
        t.territory_name,
        t.region,
        COUNT(o.opportunity_id)                                              AS total_opportunities,
        SUM(CASE WHEN o.close_status = 'Closed Won' THEN 1 ELSE 0 END)       AS deals_won,
        SUM(CASE WHEN o.close_status = 'Closed Lost' THEN 1 ELSE 0 END)      AS deals_lost,
        ROUND(100.0 * SUM(CASE WHEN o.close_status = 'Closed Won' THEN 1 ELSE 0 END)
              / NULLIF(SUM(CASE WHEN o.close_status IN ('Closed Won','Closed Lost') THEN 1 ELSE 0 END), 0), 2)
                                                                              AS win_rate_pct,
        ROUND(SUM(CASE WHEN o.close_status = 'Closed Won'
                  THEN o.deal_value * (1 - o.discount_requested_pct/100.0) ELSE 0 END), 0)
                                                                              AS revenue_closed,
        ROUND(AVG(CASE WHEN o.close_status = 'Closed Won' THEN o.sales_cycle_days_planned END), 1)
                                                                              AS avg_sales_cycle_days
    FROM sales_reps r
    JOIN territories t ON t.territory_id = r.territory_id
    LEFT JOIN opportunities o ON o.rep_id = r.rep_id
    GROUP BY r.rep_id, r.rep_name, r.seniority_level, r.territory_id, t.territory_name, t.region
)
SELECT
    *,
    RANK() OVER (PARTITION BY territory_id ORDER BY revenue_closed DESC) AS rank_within_territory,
    RANK() OVER (ORDER BY revenue_closed DESC)                           AS rank_company_wide,
    NTILE(4) OVER (ORDER BY revenue_closed DESC)                         AS performance_quartile
FROM rep_stats
ORDER BY revenue_closed DESC;
