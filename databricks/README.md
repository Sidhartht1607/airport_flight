# Flight-delay pipeline in PySpark (Databricks)

A Spark rebuild of the data prep and baseline models from `../flight_data_set.ipynb`: raw CSVs -> **bronze** -> **silver**
-> **gold** Delta tables, then Spark ML models logged to MLflow. The PyTorch embedding model (0.730 AUC) is **not**
ported; that number stays from the local notebook.

| Notebook | Does | Output table / run |
|---|---|---|
| `01_bronze_silver.py` | reads the three CSVs from a Unity Catalog volume; bronze = as received with lowercase columns; silver = operational flights (not cancelled or diverted, carrier present, label present) with the modelling and delay-cause columns; reconciles against the pandas notebook | `flights_bronze`, `flights_silver` |
| `02_gold.py` | delay rate per origin airport and scheduled hour with 95% CIs; `valid` = at least 100 flights and CI half-width at most 7 points | `flights_gold_origin_hour` |
| `03_train.py` | Spark ML logistic regression (one-hot) and gradient-boosted trees (depth 5, 50 trees), AUC and accuracy to MLflow | two MLflow runs |

## What has and has not been run

**Run locally** on the real CSVs with open-source Spark 4.0 (local mode), all three notebooks end to end:
- Silver reconciles with pandas: **1,645,503** raw rows, **1,611,046** silver rows, delay rate **0.18966** (the notebook
  asserts these).
- Gold DFW at hour 22: 2,372 flights, 28.5% delayed, same as pandas.
- Spark ML test results (`randomSplit` 80/20, seed 42):

| Model | Test AUC | Test accuracy |
|---|---|---|
| Spark logistic regression | 0.646 | 81.0% |
| Spark GBT (depth 5, 50 trees) | 0.708 | 81.6% |

**Not run on Databricks.** The notebooks have not been run on Databricks Free Edition (serverless), no Job exists yet,
and there is no run screenshot. Local Spark has no Delta, so `saveAsTable` wrote plain tables there. The notebooks
use only portable SQL for that reason. Expect small differences on serverless and fix them in the notebooks.

**Not comparable to the PyTorch results.** The sklearn split was stratified (`random_state=42`); Spark's
`randomSplit` is not, so 0.646 vs 0.647 (logistic regression) and 0.708 vs 0.730 (embedding MLP) are separate
experiments. The GBT here uses ordinal-indexed categories, not learned embeddings, which is a different model.

## Run it on Databricks Free Edition

1. Sign up for Databricks Free Edition. In Catalog, open your catalog (usually `workspace`) -> schema `default` ->
   Create -> Volume named `raw`.
2. Upload `flight_jan.csv`, `flight_feb.csv`, `flight_mar.csv` to the volume (gzip them first if the dialog refuses
   the size; Spark reads `.csv.gz`). The data is not in this repo; see the main README, "Getting the data".
3. Import the three `.py` files as notebooks (they use the `# Databricks notebook source` format), attach Serverless.
4. Develop with widget `sample_fraction = 0.1`: the free tier has a daily compute quota and stops the workspace when
   it is exceeded. The row-count assertions only run at `1.0`.
5. Workflows -> Create job with three tasks in order (`01` -> `02` -> `03`), all on Serverless. Run it once, leave any
   schedule paused (or weekly at most), and add the screenshot of the run graph here.

## Free Edition constraints the code is built around
- Serverless only, Python and SQL: no `.cache()` or `.persist()` (they error), no RDD code, so everything is
  DataFrames and intermediate results are Delta tables.
- Spark ML models are capped at 100 MB each: the GBT is deliberately small. `maxBins=400` because `origin` and `dest`
  have 334 categories each and a tree needs at least as many bins as its largest categorical.
- Outbound internet is restricted, so the CSVs are uploaded rather than downloaded in the notebook.
- If `mlflow.spark.log_model` is not permitted on serverless, `03_train.py` prints that and still logs the metrics.
