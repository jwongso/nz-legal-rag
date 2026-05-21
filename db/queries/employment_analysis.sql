-- Employment / personal grievance analytics
-- Works on the employment_view which joins documents + courts + employment_cases.

-- 1. Cases by grievance type and outcome
SELECT
    grievance_type,
    outcome,
    COUNT(*)                        AS cases
FROM employment_view
GROUP BY grievance_type, outcome
ORDER BY grievance_type, cases DESC;


-- 2. Compensation statistics by grievance type
SELECT
    grievance_type,
    COUNT(*)                                                        AS cases,
    COUNT(*) FILTER (WHERE compensation IS NOT NULL)                AS cases_with_compensation,
    ROUND(AVG(compensation) FILTER (WHERE compensation IS NOT NULL), 0) AS avg_compensation,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY compensation)       AS median_compensation,
    MAX(compensation)                                               AS max_compensation
FROM employment_view
WHERE grievance_type IS NOT NULL
GROUP BY grievance_type
HAVING COUNT(*) >= 5
ORDER BY avg_compensation DESC NULLS LAST;


-- 3. Reinstatement rates
SELECT
    grievance_type,
    COUNT(*) FILTER (WHERE reinstatement = TRUE)    AS reinstated,
    COUNT(*)                                        AS total,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE reinstatement = TRUE) / COUNT(*), 1
    )                                               AS reinstatement_rate_pct
FROM employment_view
WHERE grievance_type IS NOT NULL
GROUP BY grievance_type
HAVING COUNT(*) >= 5
ORDER BY reinstatement_rate_pct DESC;


-- 4. Contributory conduct distribution (how often and how much)
SELECT
    grievance_type,
    COUNT(*) FILTER (WHERE contributory_conduct_pct IS NOT NULL)    AS cases_with_contrib,
    ROUND(AVG(contributory_conduct_pct) FILTER (WHERE contributory_conduct_pct IS NOT NULL), 1) AS avg_contrib_pct,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY contributory_conduct_pct) AS median_contrib_pct
FROM employment_view
WHERE grievance_type IS NOT NULL
GROUP BY grievance_type
ORDER BY avg_contrib_pct DESC NULLS LAST;


-- 5. Outcomes by year (trend analysis)
SELECT
    EXTRACT(YEAR FROM decision_date)::INTEGER   AS year,
    COUNT(*)                                    AS cases,
    COUNT(*) FILTER (WHERE outcome = 'upheld')  AS upheld,
    COUNT(*) FILTER (WHERE reinstatement)       AS reinstated,
    ROUND(AVG(compensation) FILTER (WHERE compensation IS NOT NULL), 0) AS avg_compensation
FROM employment_view
WHERE decision_date IS NOT NULL
GROUP BY year
ORDER BY year;


-- 6. Top cases by remedy amount
SELECT
    citation, title, court, decision_date,
    grievance_type, outcome, remedy_amount, reinstatement
FROM employment_view
WHERE remedy_amount IS NOT NULL
ORDER BY remedy_amount DESC
LIMIT 20;
