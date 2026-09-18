-- ============================================================================
-- 04. LOST DEAL ANALYSIS
-- Why deals are lost, at which stage, and how much revenue that represents -
-- ranked so the biggest revenue-at-risk reasons surface first.
-- ============================================================================

WITH lost AS (
    SELECT
        ld.*,
        o.stage_reached_before_close,
        o.discount_requested_pct,
        a.industry,
        a.company_size_band
    FROM lost_deals ld
    JOIN opportunities o ON o.opportunity_id = ld.opportunity_id
    JOIN accounts a ON a.account_id = ld.account_id
)
SELECT
    lost_reason,
    stage_at_loss,
    COUNT(*)                                   AS num_lost_deals,
    ROUND(SUM(lost_value), 0)                  AS total_lost_value,
    ROUND(AVG(lost_value), 0)                  AS avg_lost_deal_size,
    ROUND(AVG(sales_cycle_days), 1)            AS avg_days_before_loss,
    RANK() OVER (ORDER BY SUM(lost_value) DESC) AS revenue_at_risk_rank
FROM lost
GROUP BY lost_reason, stage_at_loss
ORDER BY total_lost_value DESC;
