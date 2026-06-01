# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:59:04 2026

@author: jenny
"""

CREATE OR REPLACE TABLE clean_storm_data.ml_features_joined AS
WITH parsed_damage AS (
  SELECT 
    EVENT_ID,
    STATE,
    YEAR,
    EVENT_TYPE,
    MAGNITUDE,
    INJURIES_DIRECT,
    DEATHS_DIRECT,
    BEGIN_DATE_TIME,
    
    -- Extract the month from the timestamp for seasonal features
    EXTRACT(MONTH FROM SAFE.PARSE_TIMESTAMP('%d-%b-%y %H:%M:%S', BEGIN_DATE_TIME)) AS storm_month,

    -- Step 1: Clean and scale the messy DAMAGE_PROPERTY text strings into pure numbers
    CASE 
      WHEN UPPER(DAMAGE_PROPERTY) LIKE '%K' 
        THEN SAFE_CAST(REGEXP_REPLACE(DAMAGE_PROPERTY, r'(?i)K', '') AS FLOAT64) * 1000
      WHEN UPPER(DAMAGE_PROPERTY) LIKE '%M' 
        THEN SAFE_CAST(REGEXP_REPLACE(DAMAGE_PROPERTY, r'(?i)M', '') AS FLOAT64) * 1000000
      WHEN UPPER(DAMAGE_PROPERTY) LIKE '%B' 
        THEN SAFE_CAST(REGEXP_REPLACE(DAMAGE_PROPERTY, r'(?i)B', '') AS FLOAT64) * 1000000000
      ELSE SAFE_CAST(DAMAGE_PROPERTY AS FLOAT64)
    END AS damage_property_numeric

  FROM 
    `clean_storm_data.staging_storm_details`
),

consolidated_groups AS (
  SELECT 
    *,
    -- Step 2: Consolidate messy/rare storm types into cleaner parent categories for the ML model
    CASE 
      WHEN UPPER(EVENT_TYPE) LIKE '%FLOOD%' OR UPPER(EVENT_TYPE) LIKE '%SURGE%' THEN 'FLOOD_EVENT'
      WHEN UPPER(EVENT_TYPE) LIKE '%WIND%' THEN 'WIND_EVENT'
      WHEN UPPER(EVENT_TYPE) LIKE '%SNOW%' OR UPPER(EVENT_TYPE) LIKE '%BLIZZARD%' OR UPPER(EVENT_TYPE) LIKE '%ICE%' THEN 'WINTER_STORM'
      WHEN UPPER(EVENT_TYPE) LIKE '%TORNADO%' THEN 'TORNADO'
      WHEN UPPER(EVENT_TYPE) LIKE '%HAIL%' THEN 'HAIL'
      WHEN UPPER(EVENT_TYPE) LIKE '%HEAT%' OR UPPER(EVENT_TYPE) LIKE '%DROUGHT%' THEN 'EXTREME_HEAT'
      ELSE 'OTHER_WEATHER'
    END AS simplified_event_type
  FROM 
    parsed_damage
)

-- Step 3: Package the clean features and dynamically build the ML target variable
SELECT
  EVENT_ID,
  STATE,
  YEAR,
  storm_month,
  simplified_event_type,
  COALESCE(MAGNITUDE, 0.0) AS magnitude_clean,
  COALESCE(INJURIES_DIRECT, 0) AS injuries_clean,
  COALESCE(DEATHS_DIRECT, 0) AS deaths_clean,
  COALESCE(damage_property_numeric, 0.0) AS total_damage_usd,
  
  -- Create the ML Target Variable: 1 if damage >= $50,000 (Severe), else 0 (Minor)
  CASE 
    WHEN COALESCE(damage_property_numeric, 0.0) >= 50000 THEN 1 
    ELSE 0 
  END AS is_severe_damage

FROM 
  consolidated_groups;