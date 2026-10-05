# Databricks notebook source
# MAGIC %md
# MAGIC # 01: Bronze and silver tables
# MAGIC Raw DOT CSVs (uploaded to a Unity Catalog volume) -> **bronze** Delta table (as received, lowercase column
# MAGIC names) -> **silver** (operational flights with a label, plus the modelling and cause columns).
# MAGIC Mirrors the cleaning in `flight_data_set.ipynb`. Built for Databricks Free Edition (serverless): no `.cache()`,
# MAGIC DataFrames only, intermediate results go to Delta tables.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("volume", "raw")
dbutils.widgets.text("raw_path", "")             # blank = the Unity Catalog volume below; set to read CSVs from elsewhere
dbutils.widgets.text("sample_fraction", "1.0")   # use 0.1 while developing: the free tier has a daily compute quota

CATALOG, SCHEMA, VOLUME = (dbutils.widgets.get(n) for n in ("catalog", "schema", "volume"))
SAMPLE = float(dbutils.widgets.get("sample_fraction"))
RAW = dbutils.widgets.get("raw_path") or f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from pyspark.sql import functions as F

raw = (spark.read.option("header", True).option("inferSchema", True)
       .csv([f"{RAW}/flight_jan.csv", f"{RAW}/flight_feb.csv", f"{RAW}/flight_mar.csv"]))

# same renames and lowercasing as the notebook; the trailing empty column the DOT export adds is dropped
bronze = raw.toDF(*[c.lower() for c in raw.columns]).withColumnRenamed("day_of_month", "day") \
            .withColumnRenamed("op_unique_carrier", "op_carrier")
bronze = bronze.drop(*[c for c in bronze.columns if c.startswith("_c")])
if SAMPLE < 1.0:
    bronze = bronze.sample(fraction=SAMPLE, seed=42)
bronze.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(T("flights_bronze"))

# COMMAND ----------

MODEL_COLS = ["year", "month", "day", "hour", "op_carrier", "origin", "origin_state_nm", "dest", "dest_state_nm",
              "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance", "distance_group",
              "dep_time_blk", "arr_time_blk"]
CAUSE_COLS = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]

silver = (spark.table(T("flights_bronze"))
          .where((F.col("cancelled") != 1) & (F.col("diverted") != 1))      # cancelled/diverted have no real departure
          .where(F.col("op_carrier").isNotNull())
          .withColumn("hour", (F.col("crs_dep_time") / 100).cast("int"))    # scheduled departure hour, as in the notebook
          .where(F.col("dep_del15").isNotNull())                            # no label, no training row
          .withColumn("dep_del15", F.col("dep_del15").cast("int"))
          .select("fl_date", "origin", *[c for c in MODEL_COLS if c != "origin"], "dep_del15", *CAUSE_COLS))
silver.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(T("flights_silver"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reconciliation against the pandas notebook
# MAGIC On the full data (`sample_fraction = 1.0`) the pandas notebook gives: raw 1,645,503 rows, silver 1,611,046 rows,
# MAGIC overall delay rate 0.18966. If these do not match, find out why before modelling.

# COMMAND ----------

n_raw = spark.table(T("flights_bronze")).count()
stats = spark.table(T("flights_silver")).agg(F.count("*").alias("n"), F.avg("dep_del15").alias("delay_rate")).first()
print(f"bronze rows: {n_raw:,}   silver rows: {stats['n']:,}   delay rate: {stats['delay_rate']:.5f}")
if SAMPLE == 1.0:
    assert n_raw == 1_645_503, n_raw
    assert stats["n"] == 1_611_046, stats["n"]
    assert abs(stats["delay_rate"] - 0.18966) < 5e-5, stats["delay_rate"]
