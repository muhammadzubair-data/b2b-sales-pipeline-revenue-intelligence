-- ============================================================================
-- 03. STAGE VELOCITY
-- Average / median-proxy / p90 time spent in each funnel stage, and each
-- stage's share of the total sales cycle. Uses window functions to compute
-- per-opportunity total cycle time alongside each stage row.
-- ============================================================================

WITH completed_stages AS (
    SELECT
        ph.opportunity_id,
        ph.stage,
        ph.duration_days,
        SUM(ph.duration_days) OVER (PARTITION BY ph.opportunity_id) AS opp_total_tracked_days
    FROM pipeline_history ph
    WHERE ph.duration_days IS NOT NULL
)
SELECT
    stage,
    COUNT(*)                                        AS stage_instances,
    ROUND(AVG(duration_days), 1)                     AS avg_days_in_stage,
    ROUND(AVG(100.0 * duration_days / NULLIF(opp_total_tracked_days, 0)), 1) AS avg_pct_of_cycle,
    MIN(duration_days)                               AS min_days,
    MAX(duration_days)                               AS max_days
FROM completed_stages
GROUP BY stage
ORDER BY
    CASE stage
        WHEN 'Qualification' THEN 1
        WHEN 'Needs Analysis' THEN 2
        WHEN 'Proposal' THEN 3
        WHEN 'Negotiation' THEN 4
        ELSE 5
    END;
