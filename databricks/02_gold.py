# Databricks notebook source
# MAGIC %md
# MAGIC # 02: Gold table, delay rate by origin airport and hour with 95% confidence intervals
# MAGIC The notebook's confidence-interval analysis in Spark. A bucket is `valid` only with at least 100 flights and a
# MAGIC CI half-width of at most 7 percentage points (small buckets looked scarier than they were).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {T('flights_gold_origin_hour')}")
spark.sql(f"""
CREATE TABLE {T('flights_gold_origin_hour')} AS
WITH grp AS (
  SELECT origin, hour, COUNT(*) AS n, SUM(dep_del15) AS k
  FROM {T('flights_silver')}
  GROUP BY origin, hour
), rates AS (
  SELECT *, k / n AS p, 100 * k / n AS delay_rate FROM grp
), ci AS (
  SELECT *, 100 * 1.96 * SQRT(p * (1 - p) / n) AS margin FROM rates
)
SELECT origin, hour, n, delay_rate, margin,
       GREATEST(delay_rate - margin, 0)   AS ci_lower,
       LEAST(delay_rate + margin, 100)    AS ci_upper,
       (n >= 100 AND margin <= 7.0)       AS valid
FROM ci
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Late-evening check: the README found DFW, PIA, SFB and SGF with high late-evening delay rates. Only buckets with
# MAGIC `valid = true` should be trusted.

# COMMAND ----------

display(spark.sql(f"""
SELECT origin, hour, n, ROUND(delay_rate, 1) AS delay_rate, ROUND(margin, 1) AS margin
FROM {T('flights_gold_origin_hour')}
WHERE valid AND hour >= 20 AND origin IN ('DFW', 'PIA', 'SFB', 'SGF')
ORDER BY origin, hour
"""))
