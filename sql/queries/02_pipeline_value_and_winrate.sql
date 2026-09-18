-- ============================================================================
-- 02. PIPELINE VALUE, WIN RATE & AVERAGE DEAL SIZE BY SEGMENT
-- ============================================================================

WITH opp_enriched AS (
    SELECT
        o.*,
        a.industry,
        a.company_size_band,
        a.territory_id
    FROM opportunities o
    JOIN accounts a ON a.account_id = o.account_id
)
SELECT
    company_size_band,
    industry,
    COUNT(*)                                                            AS total_opportunities,
    SUM(CASE WHEN close_status = 'Open' THEN deal_value ELSE 0 END)     AS open_pipeline_value,
    SUM(CASE WHEN close_status = 'Closed Won' THEN 1 ELSE 0 END)        AS won_count,
    SUM(CASE WHEN close_status = 'Closed Lost' THEN 1 ELSE 0 END)       AS lost_count,
    ROUND(100.0 * SUM(CASE WHEN close_status = 'Closed Won' THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN close_status IN ('Closed Won','Closed Lost') THEN 1 ELSE 0 END), 0), 2)
                                                                         AS win_rate_pct,
    ROUND(AVG(CASE WHEN close_status = 'Closed Won' THEN deal_value END), 0) AS avg_won_deal_size,
    ROUND(AVG(deal_value), 0)                                           AS avg_deal_size_all
FROM opp_enriched
GROUP BY company_size_band, industry
ORDER BY open_pipeline_value DESC;
