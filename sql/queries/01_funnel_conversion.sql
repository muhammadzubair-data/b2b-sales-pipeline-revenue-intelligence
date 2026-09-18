-- ============================================================================
-- 01. FUNNEL CONVERSION
-- Lead -> Qualified/Converted Lead -> Opportunity -> Proposal -> Negotiation
--       -> Won/Lost
-- Reports absolute counts and stage-over-stage conversion rates.
-- Portable ANSI SQL (CTEs + window functions). Validated on SQLite; ports to
-- Snowflake / Postgres / SQL Server unchanged.
-- ============================================================================

WITH lead_funnel AS (
    SELECT
        COUNT(*)                                                   AS total_leads,
        SUM(CASE WHEN converted_to_opportunity = 1 THEN 1 ELSE 0 END) AS converted_leads
    FROM leads
),
opp_funnel AS (
    SELECT
        COUNT(*)                                                              AS total_opportunities,
        SUM(CASE WHEN stage_reached_before_close IN
                 ('Needs Analysis','Proposal','Negotiation')
                 OR close_status = 'Closed Won' THEN 1 ELSE 0 END)            AS reached_needs_analysis_plus,
        SUM(CASE WHEN stage_reached_before_close IN ('Proposal','Negotiation')
                 OR close_status = 'Closed Won' THEN 1 ELSE 0 END)            AS reached_proposal_plus,
        SUM(CASE WHEN stage_reached_before_close = 'Negotiation'
                 OR close_status = 'Closed Won' THEN 1 ELSE 0 END)            AS reached_negotiation_plus,
        SUM(CASE WHEN close_status = 'Closed Won' THEN 1 ELSE 0 END)          AS won,
        SUM(CASE WHEN close_status = 'Closed Lost' THEN 1 ELSE 0 END)         AS lost,
        SUM(CASE WHEN close_status = 'Open' THEN 1 ELSE 0 END)                AS open_opps
    FROM opportunities
),
stages AS (
    SELECT 0 AS ord, 'Lead'                    AS stage_name, total_leads       AS stage_count FROM lead_funnel
    UNION ALL
    SELECT 1, 'Converted Lead / Opportunity',  total_opportunities              FROM opp_funnel
    UNION ALL
    SELECT 2, 'Needs Analysis+',               reached_needs_analysis_plus      FROM opp_funnel
    UNION ALL
    SELECT 3, 'Proposal+',                     reached_proposal_plus            FROM opp_funnel
    UNION ALL
    SELECT 4, 'Negotiation+',                  reached_negotiation_plus         FROM opp_funnel
    UNION ALL
    SELECT 5, 'Closed Won',                    won                              FROM opp_funnel
)
SELECT
    ord,
    stage_name,
    stage_count,
    ROUND(100.0 * stage_count / FIRST_VALUE(stage_count) OVER (ORDER BY ord), 2)              AS pct_of_top_of_funnel,
    ROUND(100.0 * stage_count / NULLIF(LAG(stage_count) OVER (ORDER BY ord), 0), 2)            AS pct_of_previous_stage
FROM stages
ORDER BY ord;
