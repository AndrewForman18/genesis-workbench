# Databricks notebook source
# MAGIC %md
# MAGIC # Update MLflow Run Status
# MAGIC Utility notebook called by the Parabricks alignment job to update
# MAGIC MLflow run tags on success or failure.

# COMMAND ----------

dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("job_status", "", "Job Status")

mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
job_status = dbutils.widgets.get("job_status")

# COMMAND ----------

import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.set_tag("job_status", job_status)
    print(f"Updated run {mlflow_run_id} with status: {job_status}")
