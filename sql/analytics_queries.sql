-- KPI 1: overall readmission distribution
SELECT
    readmitted,
    COUNT(*) AS encounter_count
FROM raw_hospital_records
GROUP BY readmitted
ORDER BY encounter_count DESC;

-- KPI 2: readmission rate by age bucket
SELECT
    age,
    ROUND(100.0 * AVG(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END), 2) AS readmission_rate_pct,
    COUNT(*) AS encounter_count
FROM raw_hospital_records
GROUP BY age
ORDER BY readmission_rate_pct DESC, encounter_count DESC;

-- KPI 3: readmission rate by admission type
SELECT
    admission_type_id,
    ROUND(100.0 * AVG(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END), 2) AS readmission_rate_pct,
    COUNT(*) AS encounter_count
FROM raw_hospital_records
GROUP BY admission_type_id
ORDER BY readmission_rate_pct DESC, encounter_count DESC;

-- KPI 4: readmission rate by length of stay
SELECT
    time_in_hospital,
    ROUND(100.0 * AVG(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END), 2) AS readmission_rate_pct,
    COUNT(*) AS encounter_count
FROM raw_hospital_records
GROUP BY time_in_hospital
HAVING COUNT(*) >= 100
ORDER BY time_in_hospital;

-- KPI 5: top primary diagnoses associated with readmission
SELECT
    diag_1,
    COUNT(*) AS readmitted_encounters
FROM raw_hospital_records
WHERE readmitted = '<30' AND diag_1 IS NOT NULL AND diag_1 <> '?'
GROUP BY diag_1
ORDER BY readmitted_encounters DESC
LIMIT 10;
