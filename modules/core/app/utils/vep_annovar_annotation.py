"""VEP 115 + ANNOVAR variant annotation utilities (v2).

Provides job submission, run search, and result retrieval functions
for the variant_annotation_vep_annovar job (Genesis Workbench v2).
"""

import os
import mlflow
import pandas as pd
from genesis_workbench.workbench import UserInfo, execute_workflow, execute_select_query
from genesis_workbench.models import set_mlflow_experiment
from genesis_config import GenesisConfig

_cfg = GenesisConfig.load()


# ── Job Submission ────────────────────────────────────────────────────────────

def start_vep_annovar_annotation(
    user_info: UserInfo,
    vcf_path: str,
    genome_build: str,
    annotation_tools: str,
    annovar_protocols: str,
    annovar_operations: str,
    vep_extra_flags: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
):
    """Start a VEP 115 + ANNOVAR annotation job (v2).

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None,
    )

    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("vcf_path", vcf_path)
        mlflow.log_param("genome_build", genome_build)
        mlflow.log_param("annotation_tools", annotation_tools)
        mlflow.log_param("annovar_protocols", annovar_protocols)

        job_run_id = execute_workflow(
            job_id=os.environ["VEP_ANNOVAR_ANNOTATION_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
                "vcf_path": vcf_path,
                "reference_volume": os.environ.get(
                    "VARIANT_ANNOTATION_REFERENCE_VOLUME",
                    _cfg.volume("variant_annotation_reference"),
                ),
                "genome_build": genome_build,
                "annotation_tools": annotation_tools,
                "vep_extra_flags": vep_extra_flags,
                "annovar_protocols": annovar_protocols,
                "annovar_operations": annovar_operations,
                "mlflow_run_id": mlflow_run_id,
                "run_name": mlflow_run_name,
                "user_email": user_info.user_email,
            },
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "variant_annotation_v2")
        mlflow.set_tag("annotation_tools", annotation_tools)
        mlflow.set_tag("genome_build", genome_build)
        mlflow.set_tag("run_name", mlflow_run_name)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id


# ── Run Search ────────────────────────────────────────────────────────────────

def _search_v2_runs(user_email: str, extra_filter: str = "") -> pd.DataFrame:
    """Internal helper: search VEP/ANNOVAR annotation runs."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {
        exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list
    }
    experiment_ids = list(experiments.keys())

    filter_parts = [
        "tags.feature='variant_annotation_v2'",
        f"tags.created_by='{user_email}'",
        "tags.origin='genesis_workbench'",
    ]
    if extra_filter:
        filter_parts.append(extra_filter)

    runs = mlflow.search_runs(
        filter_string=" AND ".join(filter_parts),
        experiment_ids=experiment_ids,
    )
    if len(runs) == 0:
        return pd.DataFrame()

    runs["experiment_name"] = runs["experiment_id"].map(experiments)
    return runs


def search_vep_annovar_runs_by_run_name(
    user_email: str, run_name: str
) -> pd.DataFrame:
    """Search VEP/ANNOVAR annotation runs by run name substring."""
    runs = _search_v2_runs(user_email)
    if len(runs) == 0:
        return pd.DataFrame()

    filtered = runs[
        runs["tags.mlflow.runName"].str.contains(run_name, case=False, na=False)
    ]
    if len(filtered) == 0:
        return pd.DataFrame()

    result = filtered[
        [
            "run_id",
            "tags.mlflow.runName",
            "experiment_name",
            "params.vcf_path",
            "params.annotation_tools",
            "start_time",
            "tags.job_status",
        ]
    ].copy()
    result.columns = [
        "run_id",
        "run_name",
        "experiment_name",
        "vcf_path",
        "annotation_tools",
        "start_time",
        "status",
    ]
    return result


def search_vep_annovar_runs_by_experiment_name(
    user_email: str, experiment_name: str
) -> pd.DataFrame:
    """Search VEP/ANNOVAR annotation runs by experiment name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    matching = {
        exp.experiment_id: exp.name.split("/")[-1]
        for exp in experiment_list
        if experiment_name.upper() in exp.name.split("/")[-1].upper()
    }
    if len(matching) == 0:
        return pd.DataFrame()

    all_experiments = {
        exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list
    }

    runs = mlflow.search_runs(
        filter_string=(
            "tags.feature='variant_annotation_v2' AND "
            f"tags.created_by='{user_email}' AND "
            "tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(matching.keys()),
    )
    if len(runs) == 0:
        return pd.DataFrame()

    runs["experiment_name"] = runs["experiment_id"].map(all_experiments)
    result = runs[
        [
            "run_id",
            "tags.mlflow.runName",
            "experiment_name",
            "params.vcf_path",
            "params.annotation_tools",
            "start_time",
            "tags.job_status",
        ]
    ].copy()
    result.columns = [
        "run_id",
        "run_name",
        "experiment_name",
        "vcf_path",
        "annotation_tools",
        "start_time",
        "status",
    ]
    return result


# ── Results Retrieval ─────────────────────────────────────────────────────────

def pull_vep_annovar_results(run_id: str, run_name: str = "") -> pd.DataFrame:
    """Pull VEP + ANNOVAR annotation results from the merged Delta table."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    table = f"{catalog}.{schema}.variant_annotations_v2"

    # Resolve run_name from MLflow if not provided
    if not run_name:
        try:
            run = mlflow.get_run(run_id)
            run_name = run.data.tags.get("run_name", "")
        except Exception:
            pass

    run_filter = f"WHERE run_name = '{run_name}'" if run_name else ""

    # Check what columns exist
    try:
        cols_df = execute_select_query(f"DESCRIBE {table}")
        available_cols = (
            cols_df["col_name"].tolist() if "col_name" in cols_df.columns else []
        )
    except Exception:
        return pd.DataFrame()

    # Build select columns based on what's available
    select_cols = []
    for col in [
        "variant_key",
        "combined_impact",
        "annotation_source",
        "SYMBOL",
        "Consequence",
        "IMPACT",
        "SIFT",
        "PolyPhen",
        "HGVSc",
        "HGVSp",
        "gnomADe_AF",
        "gnomADg_AF",
        "VARIANT_CLASS",
        "Existing_variation",
    ]:
        if col in available_cols:
            select_cols.append(col)

    if not select_cols:
        select_cols = ["*"]

    query = f"""
        SELECT {', '.join(select_cols)}
        FROM {table}
        {run_filter}
        ORDER BY
            CASE combined_impact
                WHEN 'HIGH' THEN 1
                WHEN 'MODERATE' THEN 2
                WHEN 'LOW' THEN 3
                ELSE 4
            END,
            variant_key
        LIMIT 5000
    """
    return execute_select_query(query)
