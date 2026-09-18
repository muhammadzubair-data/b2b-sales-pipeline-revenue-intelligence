-- ============================================================================
-- 08. STAGE TRANSITION MATRIX
-- For every opportunity, orders its stage history and uses LEAD() to see
-- which stage it moved to next (or "Closed Won" / "Closed Lost" if it was
-- the final stage record). Aggregates into a from->to transition matrix -
-- useful for spotting where deals stall or bounce backward.
-- ============================================================================

WITH ordered_stages AS (
    SELECT
        ph.opportunity_id,
        ph.stage,
        ph.entered_date,
        o.close_status,
        LEAD(ph.stage) OVER (PARTITION BY ph.opportunity_id ORDER BY ph.entered_date) AS next_stage,
        ROW_NUMBER() OVER (PARTITION BY ph.opportunity_id ORDER BY ph.entered_date DESC) AS rn_from_end
    FROM pipeline_history ph
    JOIN opportunities o ON o.opportunity_id = ph.opportunity_id
),
transitions AS (
    SELECT
        stage AS from_stage,
        COALESCE(
            next_stage,
            CASE WHEN rn_from_end = 1 AND close_status = 'Closed Won' THEN 'Closed Won'
                 WHEN rn_from_end = 1 AND close_status = 'Closed Lost' THEN 'Closed Lost'
                 ELSE 'Still Open (no further movement yet)'
            END
        ) AS to_stage
    FROM ordered_stages
)
SELECT
    from_stage,
    to_stage,
    COUNT(*) AS num_opportunities,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY from_stage), 2) AS pct_of_from_stage
FROM transitions
GROUP BY from_stage, to_stage
ORDER BY from_stage, num_opportunities DESC;
