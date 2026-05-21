-- Sentencing analytics queries
-- Works on the sentencing_view which joins documents + courts + sentencing_cases.

-- 1. Cases by offence and court (with date range)
SELECT citation, title, court, decision_date,
       starting_point, final_sentence, guilty_plea_discount
FROM sentencing_view
WHERE court = 'NZCA'
  AND offence ILIKE '%robbery%'
  AND decision_date BETWEEN '2020-01-01' AND '2025-12-31'
ORDER BY decision_date DESC;


-- 2. Median starting point and final sentence by offence
SELECT
    offence,
    COUNT(*)                                                    AS cases,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY starting_point) AS median_start_months,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY final_sentence)  AS median_final_months,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY guilty_plea_discount) AS median_gpd_pct
FROM sentencing_view
WHERE offence IS NOT NULL
  AND starting_point IS NOT NULL
GROUP BY offence
HAVING COUNT(*) >= 5
ORDER BY cases DESC;


-- 3. Guilty plea discount distribution
SELECT
    offence,
    ROUND(AVG(guilty_plea_discount), 1)                         AS avg_discount,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY guilty_plea_discount) AS p25,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY guilty_plea_discount) AS p75,
    COUNT(*) FILTER (WHERE flag_youth)                          AS youth_cases,
    COUNT(*) FILTER (WHERE flag_mental_health)                  AS mental_health_cases,
    COUNT(*) FILTER (WHERE flag_tikanga_maori)                  AS tikanga_cases,
    COUNT(*)                                                    AS total
FROM sentencing_view
WHERE guilty_plea_discount IS NOT NULL
  AND offence IS NOT NULL
GROUP BY offence
HAVING COUNT(*) >= 10
ORDER BY total DESC;


-- 4. Sentencing trend over time (annual averages)
SELECT
    EXTRACT(YEAR FROM decision_date)::INTEGER   AS year,
    court,
    COUNT(*)                                    AS cases,
    ROUND(AVG(starting_point), 1)               AS avg_start_months,
    ROUND(AVG(final_sentence), 1)               AS avg_final_months,
    ROUND(AVG(guilty_plea_discount), 1)         AS avg_gpd_pct
FROM sentencing_view
WHERE decision_date IS NOT NULL
  AND starting_point IS NOT NULL
GROUP BY year, court
ORDER BY court, year;


-- 5. Home detention usage
SELECT
    offence,
    COUNT(*) FILTER (WHERE home_detention_months IS NOT NULL)   AS home_detention_cases,
    COUNT(*)                                                    AS total_cases,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE home_detention_months IS NOT NULL) / COUNT(*), 1
    )                                                           AS home_detention_pct,
    ROUND(AVG(home_detention_months) FILTER (WHERE home_detention_months IS NOT NULL), 1) AS avg_hd_months
FROM sentencing_view
WHERE offence IS NOT NULL
GROUP BY offence
HAVING COUNT(*) >= 5
ORDER BY home_detention_pct DESC;


-- 6. Factor prevalence
SELECT
    COUNT(*) FILTER (WHERE flag_self_defence)       AS self_defence,
    COUNT(*) FILTER (WHERE flag_provocation)        AS provocation,
    COUNT(*) FILTER (WHERE flag_mental_health)      AS mental_health,
    COUNT(*) FILTER (WHERE flag_intoxication)       AS intoxication,
    COUNT(*) FILTER (WHERE flag_youth)              AS youth,
    COUNT(*) FILTER (WHERE flag_tikanga_maori)      AS tikanga_maori,
    COUNT(*) FILTER (WHERE flag_cultural_factors)   AS cultural_factors,
    COUNT(*)                                        AS total
FROM sentencing_view;
