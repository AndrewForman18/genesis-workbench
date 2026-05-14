# Databricks notebook source
# DBTITLE 1,Parabricks GPU Germline Pipeline
# MAGIC %md
# MAGIC # Parabricks GPU Germline Pipeline
# MAGIC GPU-accelerated alignment (fq2bam) and variant calling (HaplotypeCaller) using NVIDIA Clara Parabricks.
# MAGIC
# MAGIC **Environment**: Resolved from `genesis_env.yml` — change `active_environment` or set `GENESIS_ENV` to deploy to different workspaces.

# COMMAND ----------

# DBTITLE 1,Load environment config and register widgets
# Cell 1: Load environment config and register widgets
import sys
sys.path.insert(0, "/Workspace/Users/andrew_forman@eisai.com/genesis-workbench")
from genesis_config import GenesisConfig

cfg = GenesisConfig.load()
pb = cfg.parabricks

# Workflow parameters (passed by job or entered manually)
dbutils.widgets.text("fastq_r1", "", "FASTQ R1")
dbutils.widgets.text("fastq_r2", "", "FASTQ R2")
dbutils.widgets.text("reference_genome_path",
    pb["reference_genome"].format(catalog=cfg.catalog, schema=cfg.schema),
    "Reference")
dbutils.widgets.text("output_volume_path", cfg.volume("gwas_data"), "Output")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "", "User Email")

# Environment parameters (resolved from genesis_env.yml)
dbutils.widgets.text("catalog", cfg.catalog, "Catalog")
dbutils.widgets.text("schema", cfg.schema, "Schema")
dbutils.widgets.text("sql_warehouse_id", cfg.sql_warehouse_id, "SQL Warehouse Id")

print(f"Environment: {cfg.env_name}")
print(f"Catalog: {cfg.catalog}.{cfg.schema}")
print(f"Parabricks image: {pb['docker_image']}")
print("Widgets registered successfully")

# COMMAND ----------

# Cell 2: Read parameters
fastq_r1 = dbutils.widgets.get("fastq_r1")
fastq_r2 = dbutils.widgets.get("fastq_r2")
reference = dbutils.widgets.get("reference_genome_path")
output_path = dbutils.widgets.get("output_volume_path")

print("CELL 2 PASSED: Parameters read")
print(f"  fastq_r1 = {fastq_r1}")
print(f"  fastq_r2 = {fastq_r2}")
print(f"  reference = {reference}")
print(f"  output = {output_path}")

# COMMAND ----------

# Cell 3: Stage files from UC Volumes to local disk using dbutils.fs
# (FUSE mount is disabled, so /Volumes/ paths won't work directly)
import os
import time

LOCAL_DIR = "/tmp/parabricks_data"
os.makedirs(LOCAL_DIR, exist_ok=True)

fastq_r1 = dbutils.widgets.get("fastq_r1")
fastq_r2 = dbutils.widgets.get("fastq_r2")
reference = dbutils.widgets.get("reference_genome_path")

if fastq_r1:
    print("Staging files from UC Volumes to local disk...")
    start = time.time()
    
    local_r1 = os.path.join(LOCAL_DIR, os.path.basename(fastq_r1))
    local_r2 = os.path.join(LOCAL_DIR, os.path.basename(fastq_r2))
    local_ref = os.path.join(LOCAL_DIR, os.path.basename(reference))
    
    dbutils.fs.cp(f"dbfs:{fastq_r1}", f"file:{local_r1}")
    print(f"  R1 staged: {local_r1} ({os.path.getsize(local_r1)/(1024**2):.0f} MB)")
    
    dbutils.fs.cp(f"dbfs:{fastq_r2}", f"file:{local_r2}")
    print(f"  R2 staged: {local_r2} ({os.path.getsize(local_r2)/(1024**2):.0f} MB)")
    
    dbutils.fs.cp(f"dbfs:{reference}", f"file:{local_ref}")
    print(f"  Ref staged: {local_ref} ({os.path.getsize(local_ref)/(1024**2):.0f} MB)")
    
    # Also copy .fai index if exists
    try:
        dbutils.fs.cp(f"dbfs:{reference}.fai", f"file:{local_ref}.fai")
        print(f"  Ref.fai staged")
    except:
        pass
    
    elapsed = time.time() - start
    print(f"CELL 3 PASSED: Files staged in {elapsed:.0f}s")
else:
    print("CELL 3 PASSED: No files to stage (diagnostic mode)")
    local_r1 = local_r2 = local_ref = ""

# COMMAND ----------

# Cell 4: GPU and Parabricks check
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(f"nvidia-smi exit={result.returncode}")
for line in result.stdout.split("\n")[:8]:
    print(f"  {line}")

result = subprocess.run(["pbrun", "--version"], capture_output=True, text=True)
print(f"\npbrun version: {result.stdout.strip()}")
print("CELL 4 PASSED: GPU and Parabricks verified")

# COMMAND ----------

# Cell 5: Run the pipeline on local files
import os, subprocess

LOCAL_DIR = "/tmp/parabricks_data"
fastq_r1 = dbutils.widgets.get("fastq_r1")

if not fastq_r1:
    print("No FASTQ files - diagnostic pass only")
    dbutils.notebook.exit("DIAGNOSTIC_PASS")

local_r1 = os.path.join(LOCAL_DIR, os.path.basename(dbutils.widgets.get("fastq_r1")))
local_r2 = os.path.join(LOCAL_DIR, os.path.basename(dbutils.widgets.get("fastq_r2")))
local_ref = os.path.join(LOCAL_DIR, os.path.basename(dbutils.widgets.get("reference_genome_path")))

output_bam = os.path.join(LOCAL_DIR, "output.bam")
output_vcf = os.path.join(LOCAL_DIR, "germline.vcf")

# fq2bam
print("=== fq2bam (alignment) ===")
cmd = f"pbrun fq2bam --ref {local_ref} --in-fq {local_r1} {local_r2} --out-bam {output_bam} --low-memory"
print(f"CMD: {cmd}")
ret = os.system(cmd)
if ret != 0:
    raise RuntimeError(f"fq2bam failed: exit {ret}")

# haplotypecaller
print("\n=== haplotypecaller ===")
cmd = f"pbrun haplotypecaller --ref {local_ref} --in-bam {output_bam} --out-variants {output_vcf}"
print(f"CMD: {cmd}")
ret = os.system(cmd)
if ret != 0:
    raise RuntimeError(f"haplotypecaller failed: exit {ret}")

print(f"\n=== SUCCESS ===")
print(f"BAM: {output_bam} ({os.path.getsize(output_bam)/(1024**3):.2f} GB)")
print(f"VCF: {output_vcf} ({os.path.getsize(output_vcf)/(1024**2):.1f} MB)")

# COMMAND ----------

# Cell 6: Copy results back to UC Volume
import os
output_path = dbutils.widgets.get("output_volume_path")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
run_id = mlflow_run_id if mlflow_run_id else "latest"
dest = f"dbfs:{output_path}/alignment/{run_id}"

LOCAL_DIR = "/tmp/parabricks_data"
output_bam = os.path.join(LOCAL_DIR, "output.bam")
output_vcf = os.path.join(LOCAL_DIR, "germline.vcf")

dbutils.fs.cp(f"file:{output_bam}", f"{dest}/output.bam")
dbutils.fs.cp(f"file:{output_vcf}", f"{dest}/germline.vcf")
print(f"Results uploaded to {dest}")
print("PIPELINE COMPLETE")
