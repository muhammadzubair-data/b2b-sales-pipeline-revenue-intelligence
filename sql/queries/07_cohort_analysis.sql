-- ============================================================================
-- 07. COHORT ANALYSIS
-- Groups leads by creation month and tracks the cohort all the way through:
-- % converted to opportunity, % eventually won, and revenue per 100 leads -
-- so marketing/source investment can be judged by downstream revenue, not
-- just top-of-funnel volume.
-- ============================================================================

WITH lead_cohort AS (
    SELECT
        l.lead_id,
        l.source,
        strftime('%Y-%m', l.created_date) AS cohort_month,
        l.converted_to_opportunity,
        o.opportunity_id,
        o.close_status,
        o.deal_value,
        o.discount_requested_pct
    FROM leads l
    LEFT JOIN opportunities o ON o.lead_id = l.lead_id
)
SELECT
    cohort_month,
    source,
    COUNT(DISTINCT lead_id)                                                        AS leads_in_cohort,
    SUM(converted_to_opportunity)                                                  AS converted_to_opp,
    ROUND(100.0 * SUM(converted_to_opportunity) / COUNT(DISTINCT lead_id), 2)      AS pct_converted,
    SUM(CASE WHEN close_status = 'Closed Won' THEN 1 ELSE 0 END)                   AS eventually_won,
    ROUND(100.0 * SUM(CASE WHEN close_status = 'Closed Won' THEN 1 ELSE 0 END)
          / COUNT(DISTINCT lead_id), 2)                                            AS pct_lead_to_win,
    ROUND(SUM(CASE WHEN close_status = 'Closed Won'
              THEN deal_value * (1 - discount_requested_pct/100.0) ELSE 0 END), 0) AS cohort_revenue,
    ROUND(SUM(CASE WHEN close_status = 'Closed Won'
              THEN deal_value * (1 - discount_requested_pct/100.0) ELSE 0 END)
          / COUNT(DISTINCT lead_id), 2)                                            AS revenue_per_lead
FROM lead_cohort
GROUP BY cohort_month, source
ORDER BY cohort_month, revenue_per_lead DESC;
