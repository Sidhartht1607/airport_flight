# Databricks notebook source
# MAGIC %md
# MAGIC # 03: Spark ML baselines, logged to MLflow
# MAGIC Logistic regression and gradient-boosted trees on the silver table, same features as the notebook.
# MAGIC Notes:
# MAGIC * **Not comparable one-to-one with the PyTorch numbers** (0.647 logreg / 0.730 embedding MLP): `randomSplit` is not the
# MAGIC   stratified sklearn split. Report these as their own results. The PyTorch embedding model is not ported.
# MAGIC * Free Edition caps a Spark ML model at 100 MB, so the GBT stays small. No `.cache()` on serverless.
# MAGIC * `maxBins=400` because `origin` and `dest` each have 334 categories and a tree needs at least as many bins.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("experiment", "")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import functions as F

experiment = dbutils.widgets.get("experiment") or \
    f"/Users/{spark.sql('SELECT current_user()').first()[0]}/flight-delay-spark"
mlflow.set_experiment(experiment)

NUMERIC = ["year", "month", "day", "hour", "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance", "distance_group"]
CATEGORICAL = ["op_carrier", "origin", "origin_state_nm", "dest", "dest_state_nm", "dep_time_blk", "arr_time_blk"]

df = (spark.table(T("flights_silver"))
      .select(*NUMERIC, *CATEGORICAL, F.col("dep_del15").cast("double").alias("label"))
      .fillna("UNK", subset=CATEGORICAL))
train, test = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

def features(one_hot: bool):
    """Median-impute numerics; index categoricals (unseen -> extra bucket); one-hot only for the linear model."""
    imputed = [f"{c}_imp" for c in NUMERIC]
    stages = [Imputer(strategy="median", inputCols=NUMERIC, outputCols=imputed)]
    stages += [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    if one_hot:
        stages.append(OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL],
                                    outputCols=[f"{c}_oh" for c in CATEGORICAL], handleInvalid="keep"))
        cats = [f"{c}_oh" for c in CATEGORICAL]
    else:
        cats = [f"{c}_idx" for c in CATEGORICAL]
    stages.append(VectorAssembler(inputCols=imputed + cats, outputCol="features"))
    return stages


def run(name, estimator, one_hot, params):
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**params, "n_train": train.count(), "n_test": test.count(), "split": "randomSplit 0.8/0.2 seed 42"})
        model = Pipeline(stages=features(one_hot) + [estimator]).fit(train)
        pred = model.transform(test)
        auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(pred)
        acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred)
        mlflow.log_metrics({"test_auc": auc, "test_accuracy": acc})
        try:                                      # saving a Spark ML model can be restricted on serverless
            mlflow.spark.log_model(model, "model")
        except Exception as exc:
            print("model not logged:", type(exc).__name__, str(exc)[:200])
        print(f"{name}: AUC {auc:.4f}  accuracy {acc:.4f}")
        return auc, acc


# COMMAND ----------

run("spark_logreg", LogisticRegression(maxIter=50, regParam=0.0), True, {"model": "LogisticRegression", "maxIter": 50})

# COMMAND ----------

run("spark_gbt", GBTClassifier(maxDepth=5, maxIter=50, maxBins=400, seed=42), False,
    {"model": "GBTClassifier", "maxDepth": 5, "maxIter": 50, "maxBins": 400})
