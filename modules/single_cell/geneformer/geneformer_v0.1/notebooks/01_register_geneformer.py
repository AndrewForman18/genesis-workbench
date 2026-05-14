# Databricks notebook source
# MAGIC %md
# MAGIC ### Geneformer Registration
# MAGIC
# MAGIC Registers [Geneformer](https://huggingface.co/ctheodoris/Geneformer) (30M parameter model)
# MAGIC into the Genesis Workbench module registry for single-cell gene expression analysis.
# MAGIC
# MAGIC Geneformer is a foundation transformer model pretrained on ~30M single-cell
# MAGIC transcriptomes. It can be fine-tuned for cell classification, gene network analysis,
# MAGIC and in-silico perturbation experiments.

# COMMAND ----------

dbutils.widgets.text("catalog", "dhbl_discovery_us_dev", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "geneformer_30m", "Model Name")
dbutils.widgets.text("model_version", "v0.1", "Model Version")
dbutils.widgets.text("user_email", "andrew_forman@eisai.com", "User Email")
dbutils.widgets.text("sql_warehouse_id", "d3fdeafd104a6c26", "SQL Warehouse ID")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Model: {model_name} ({model_version})")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(f"Genesis Workbench library: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

import os
os.environ["CORE_CATALOG_NAME"] = catalog
os.environ["CORE_SCHEMA_NAME"] = schema

from genesis_workbench.workbench import initialize
initialize(catalog, schema, sql_warehouse_id)

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      register_batch_model,
                                      set_mlflow_experiment)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Geneformer as a batch model
# MAGIC
# MAGIC Geneformer is registered as a batch model (not a real-time endpoint) because:
# MAGIC - It requires GPU for inference
# MAGIC - Typical usage involves processing entire datasets of cells
# MAGIC - Fine-tuning workflows run as batch jobs

# COMMAND ----------

register_batch_model(
    model_name=model_name,
    model_display_name="Geneformer 30M",
    model_description="Foundation transformer model pretrained on ~30M single-cell transcriptomes. "
                      "Supports cell classification, gene network analysis, and in-silico perturbation. "
                      "Reference: Theodoris et al., Nature 2023.",
    model_category=ModelCategory.SINGLE_CELL.value,
    module="geneformer",
    job_id=str(spark.conf.get("spark.databricks.job.id", "")),
    job_name="register_geneformer_model",
    cluster_type="GPU_SMALL",
    added_by=user_email
)

print(f"✓ Registered {model_name} as batch model in {catalog}.{schema}.batch_models")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registration Complete
# MAGIC
# MAGIC The geneformer model is now registered in the Genesis Workbench module registry.
# MAGIC To use it, fine-tune from the pretrained weights at:
# MAGIC `/Workspace/Users/andrew_forman@eisai.com/geneformer/`

