import streamlit as st

import os
import mlflow
import pandas as pd
from genesis_workbench.workbench import UserInfo, execute_workflow, execute_select_query
from genesis_workbench.models import set_mlflow_experiment
from genesis_config import GenesisConfig

_cfg = GenesisConfig.load()


def _derive_status(row):
    """Derive display status from job_status tag.

    The mark_success/mark_failure tasks in each job workflow update the
    job_status tag to the final value (e.g. 'alignment_complete', 'failed').
    'started' means the job is still running or the completion task hasn't run yet.
    """
    job_status = row.get("tags.job_status", "")
    if job_status:
        return job_status
    return "unknown"


# Progress visualization for search results
_PROGRESS_MAP = {
    # Variant Calling (2 steps)
    "alignment_complete":   "🟩🟩",
    # GWAS Analysis (3 main tasks)
    "phenotype_prepared":   "🟩⬜⬜",
    "gwas_complete":        "🟩🟩🟩",
    # VCF Ingestion (1 main task)
    "ingestion_complete":   "🟩🟩",
    # Variant Annotation (3 main tasks)
    "annotation_complete":  "🟩🟩🟩",
    "annotation_v2_complete": "🟩🟩🟩",
    # Terminal states
    "failed":               "🟥",
    "unknown":              "⬜",
}


def add_progress_column(df, total_steps=2):
    """Add a visual progress column to a search results DataFrame."""
    if df.empty or "status" not in df.columns:
        return df
    df = df.copy()
    df["progress"] = df["status"].map(
        lambda s: _PROGRESS_MAP.get(s, f"🟩{'⬜' * (total_steps - 1)}" if s == "started" else "⬜" * total_steps)
    )
    # Reorder so progress is right after status
    cols = list(df.columns)
    if "progress" in cols and "status" in cols:
        cols.remove("progress")
        idx = cols.index("status") + 1
        cols.insert(idx, "progress")
        df = df[cols]
    return df


_BLINKING_DOT_CSS = """
<style>
@keyframes blink-orange { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
.dot-in-progress { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                   background-color: #FF8C00; animation: blink-orange 1.2s infinite;
                   margin-right: 6px; vertical-align: middle; }
.dot-complete { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                background-color: #22C55E; margin-right: 6px; vertical-align: middle; }
.dot-failed { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
              background-color: #EF4444; margin-right: 6px; vertical-align: middle; }
.dot-unknown { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
               background-color: #9CA3AF; margin-right: 6px; vertical-align: middle; }
.run-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.run-table th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #444;
                font-weight: 600; color: #999; }
.run-table td { padding: 8px 12px; border-bottom: 1px solid #333; }
.run-table tr:hover { background-color: rgba(255,255,255,0.03); }
</style>
"""

_IN_PROGRESS_STATUSES = {"started", "phenotype_prepared"}
_COMPLETE_STATUSES = {"alignment_complete", "gwas_complete", "ingestion_complete", "annotation_complete", "annotation_v2_complete"}
_FAILED_STATUSES = {"failed"}


def _status_dot(status):
    if status in _IN_PROGRESS_STATUSES:
        return '<span class="dot-in-progress"></span>'
    elif status in _COMPLETE_STATUSES:
        return '<span class="dot-complete"></span>'
    elif status in _FAILED_STATUSES:
        return '<span class="dot-failed"></span>'
    return '<span class="dot-unknown"></span>'


def render_runs_html_table(df, hidden_columns=None):
    """Render a search results DataFrame as an HTML table with status dots."""
    if df.empty:
        return ""
    hidden_columns = hidden_columns or []
    display_cols = [c for c in df.columns if c not in hidden_columns]

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in display_cols:
            val = row.get(col, "")
            if col == "status":
                dot = _status_dot(str(val))
                label = str(val).replace("_", " ").title()
                cells.append(f"<td>{dot}{label}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    header_labels = [c.replace("_", " ").title() for c in display_cols]
    header = "".join(f"<th>{h}</th>" for h in header_labels)

    return (
        _BLINKING_DOT_CSS
        + f'<table class="run-table"><thead><tr>{header}</tr></thead>'
        + f'<tbody>{"".join(rows_html)}</tbody></table>'
    )


def start_parabricks_alignment(user_info: UserInfo,
                                fastq_r1: str,
                                fastq_r2: str,
                                reference_genome_path: str,
                                output_volume_path: str,
                                mlflow_experiment_name: str,
                                mlflow_run_name: str):
    """Start a Parabricks germline alignment job.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("fastq_r1", fastq_r1)
        mlflow.log_param("fastq_r2", fastq_r2)
        mlflow.log_param("reference_genome", reference_genome_path)

        job_run_id = execute_workflow(
            job_id=os.environ["PARABRICKS_ALIGNMENT_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "fastq_r1": fastq_r1,
                "fastq_r2": fastq_r2,
                "reference_genome_path": reference_genome_path,
                "output_volume_path": output_volume_path,
                "mlflow_run_id": mlflow_run_id,
                "user_email": user_info.user_email
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "gwas_alignment")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id


def start_gwas_analysis(user_info: UserInfo,
                         vcf_path: str,
                         phenotype_path: str,
                         phenotype_column: str,
                         contigs: str,
                         hwe_cutoff: str,
                         pvalue_threshold: str,
                         mlflow_experiment_name: str,
                         mlflow_run_name: str,
                         test_type: str = "both",
                         correction_method: str = "Firth"):
    """Start a GWAS analysis job using Glow.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("vcf_path", vcf_path)
        mlflow.log_param("phenotype_path", phenotype_path)
        mlflow.log_param("phenotype_column", phenotype_column)
        mlflow.log_param("contigs", contigs)
        mlflow.log_param("hwe_cutoff", hwe_cutoff)

        job_run_id = execute_workflow(
            job_id=os.environ["GWAS_ANALYSIS_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "vcf_path": vcf_path,
                "phenotype_path": phenotype_path,
                "phenotype_column": phenotype_column,
                "contigs": contigs,
                "hwe_cutoff": hwe_cutoff,
                "pvalue_threshold": pvalue_threshold,
                "test_type": test_type,
                "correction_method": correction_method,
                "mlflow_run_id": mlflow_run_id,
                "user_email": user_info.user_email
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "gwas")
        mlflow.set_tag("test_type", test_type)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id


def search_gwas_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search GWAS runs by run name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    # Search runs across accessible experiments — skip those with permission errors
    try:
        runs = mlflow.search_runs(
            filter_string=(
                f"tags.feature='gwas' AND "
                f"tags.origin='genesis_workbench'"
            ),
            experiment_ids=experiment_ids
        )
    except Exception:
        # If bulk search fails (permission issue on one experiment), try one by one
        import pandas as _pd
        _frames = []
        for eid in experiment_ids:
            try:
                _r = mlflow.search_runs(
                    filter_string="tags.feature='gwas' AND tags.origin='genesis_workbench'",
                    experiment_ids=[eid]
                )
                if len(_r) > 0:
                    _frames.append(_r)
            except Exception:
                continue
        runs = _pd.concat(_frames) if _frames else _pd.DataFrame()

    if len(runs) == 0:
        return pd.DataFrame()

    filtered = runs[runs['tags.mlflow.runName'].str.contains(run_name, case=False, na=False)]
    if len(filtered) == 0:
        return pd.DataFrame()

    filtered['experiment_name'] = filtered['experiment_id'].map(experiments)
    result = filtered[['run_id', 'tags.mlflow.runName', 'experiment_name',
                        'params.vcf_path', 'start_time', 'tags.job_status', 'tags.created_by']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'vcf_path', 'start_time', 'status', 'created_by']
    return result


def search_gwas_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search GWAS runs by experiment name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    matching = {exp.experiment_id: exp.name.split("/")[-1]
                for exp in experiment_list
                if experiment_name.upper() in exp.name.split("/")[-1].upper()}

    if len(matching) == 0:
        return pd.DataFrame()

    all_experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(matching.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(all_experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.vcf_path', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'vcf_path', 'start_time', 'status']
    return result


def search_variant_calling_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search variant calling runs by run name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas_alignment' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    filtered = runs[runs['tags.mlflow.runName'].str.contains(run_name, case=False, na=False)]
    if len(filtered) == 0:
        return pd.DataFrame()

    filtered['experiment_name'] = filtered['experiment_id'].map(experiments)
    result = filtered[['run_id', 'tags.mlflow.runName', 'experiment_name',
                        'params.fastq_r1', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'fastq_r1', 'start_time', 'status']
    return result


def search_variant_calling_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search variant calling runs by experiment name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    matching = {exp.experiment_id: exp.name.split("/")[-1]
                for exp in experiment_list
                if experiment_name.upper() in exp.name.split("/")[-1].upper()}

    if len(matching) == 0:
        return pd.DataFrame()

    all_experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(matching.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas_alignment' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(all_experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.fastq_r1', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'fastq_r1', 'start_time', 'status']
    return result


def list_successful_variant_calling_runs(user_email: str) -> pd.DataFrame:
    """List all successful variant calling runs for the GWAS run picker."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas_alignment' AND "
            f"tags.job_status='alignment_complete' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.output_vcf', 'start_time']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'output_vcf', 'start_time']
    return result


def pull_gwas_results(run_id: str) -> pd.DataFrame:
    """Pull GWAS results from the Delta table for a given run."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    table = f"gwas_results_{run_id.replace('-', '_')}"
    fq_table = f"{catalog}.{schema}.{table}"

    # First check what columns exist
    try:
        cols_df = execute_select_query(f"DESCRIBE {fq_table}")
        available_cols = cols_df["col_name"].tolist() if "col_name" in cols_df.columns else []
        print(f"[pull_gwas_results] Table {fq_table} columns: {available_cols}")
    except Exception as e:
        print(f"[pull_gwas_results] Table {fq_table} not found: {e}")
        return pd.DataFrame()

    # Check row counts
    try:
        count_df = execute_select_query(f"SELECT count(*) as total, count(pvalue) as non_null_pvalue FROM {fq_table}")
        print(f"[pull_gwas_results] Rows: total={count_df['total'].iloc[0]}, non_null_pvalue={count_df['non_null_pvalue'].iloc[0]}")
    except Exception as e:
        print(f"[pull_gwas_results] Count query failed: {e}")

    query = f"""
        SELECT contigName, start, pvalue, referenceAllele, alternateAlleles, effect, phenotype,
               CASE WHEN pvalue IS NOT NULL AND pvalue > 0 THEN -log(10, pvalue) ELSE NULL END as neg_log_pval
        FROM {fq_table}
        WHERE pvalue IS NOT NULL
        ORDER BY pvalue ASC
        LIMIT 10000
    """
    return execute_select_query(query)


# ── VCF Ingestion ──

def start_vcf_ingestion(user_info: UserInfo,
                         vcf_path: str,
                         output_table_name: str,
                         mlflow_experiment_name: str,
                         mlflow_run_name: str):
    """Start a VCF-to-Delta ingestion job.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("vcf_path", vcf_path)
        mlflow.log_param("output_table_name", output_table_name)

        job_run_id = execute_workflow(
            job_id=os.environ["VCF_INGESTION_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
                "vcf_path": vcf_path,
                "output_table_name": output_table_name,
                "mlflow_run_id": mlflow_run_id,
                "user_email": user_info.user_email
            }
        )
        catalog = os.environ["CORE_CATALOG_NAME"]
        schema = os.environ["CORE_SCHEMA_NAME"]
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "vcf_ingestion")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")
        mlflow.set_tag("output_table", f"{catalog}.{schema}.{output_table_name}")

    return job_run_id


def search_vcf_ingestion_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search VCF ingestion runs by run name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='vcf_ingestion' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    filtered = runs[runs['tags.mlflow.runName'].str.contains(run_name, case=False, na=False)]
    if len(filtered) == 0:
        return pd.DataFrame()

    filtered['experiment_name'] = filtered['experiment_id'].map(experiments)
    result = filtered[['run_id', 'tags.mlflow.runName', 'experiment_name',
                        'params.vcf_path', 'start_time', 'tags.job_status', 'tags.created_by']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'vcf_path', 'start_time', 'status', 'created_by']
    return result


def search_vcf_ingestion_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search VCF ingestion runs by experiment name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    matching = {exp.experiment_id: exp.name.split("/")[-1]
                for exp in experiment_list
                if experiment_name.upper() in exp.name.split("/")[-1].upper()}

    if len(matching) == 0:
        return pd.DataFrame()

    all_experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(matching.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='vcf_ingestion' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(all_experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.vcf_path', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'vcf_path', 'start_time', 'status']
    return result


def list_successful_vcf_ingestion_runs(user_email: str) -> pd.DataFrame:
    """List all successful VCF ingestion runs for the annotation tab picker."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='vcf_ingestion' AND "
            f"tags.job_status='ingestion_complete' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'tags.output_table', 'start_time']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'output_table', 'start_time']
    return result


# ── Variant Annotation ──

def start_variant_annotation(user_info: UserInfo,
                              variants_table: str,
                              gene_regions: str,
                              pathogenic_vcf_path: str,
                              mlflow_experiment_name: str,
                              mlflow_run_name: str,
                              gene_panel_mode: str = "custom"):
    """Start a variant annotation job.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.

    Args:
        gene_panel_mode: "custom" for JSON gene regions, "acmg" for ACMG SF v3.2 panel.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("variants_table", variants_table)
        mlflow.log_param("gene_panel_mode", gene_panel_mode)
        if gene_panel_mode != "acmg":
            mlflow.log_param("gene_regions", gene_regions)

        job_run_id = execute_workflow(
            job_id=os.environ["VARIANT_ANNOTATION_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
                "variants_table": variants_table,
                "gene_regions": gene_regions,
                "gene_panel_mode": gene_panel_mode,
                "pathogenic_vcf_path": pathogenic_vcf_path,
                "mlflow_run_id": mlflow_run_id,
                "run_name": mlflow_run_name,
                "user_email": user_info.user_email
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "variant_annotation")
        mlflow.set_tag("run_name", mlflow_run_name)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id



def start_vep_annovar_annotation(user_info: UserInfo,
                                  vcf_path: str,
                                  genome_build: str,
                                  annotation_tools: str,
                                  annovar_protocols: str,
                                  annovar_operations: str,
                                  vep_extra_flags: str,
                                  mlflow_experiment_name: str,
                                  mlflow_run_name: str):
    """Start a VEP 115 + ANNOVAR annotation job (v2).

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("vcf_path", vcf_path)
        mlflow.log_param("genome_build", genome_build)
        mlflow.log_param("annotation_tools", annotation_tools)

        job_run_id = execute_workflow(
            job_id=os.environ["VARIANT_ANNOTATION_V2_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
                "vcf_path": vcf_path,
                "reference_volume": _cfg.volume("variant_annotation_reference"),
                "genome_build": genome_build,
                "annotation_tools": annotation_tools,
                "vep_extra_flags": vep_extra_flags,
                "annovar_protocols": annovar_protocols,
                "annovar_operations": annovar_operations,
                "mlflow_run_id": mlflow_run_id,
                "run_name": mlflow_run_name,
                "user_email": user_info.user_email
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "variant_annotation_v2")
        mlflow.set_tag("annotation_tools", annotation_tools)
        mlflow.set_tag("genome_build", genome_build)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id




def start_sarek_variant_calling(user_info: UserInfo,
                                 fastq_dir: str,
                                 bam_dir: str,
                                 output_dir: str,
                                 genome: str,
                                 tools: str,
                                 step: str,
                                 analysis_type: str,
                                 pipeline_version: str,
                                 extra_args: str,
                                 mlflow_experiment_name: str,
                                 mlflow_run_name: str):
    """Start an nf-core/sarek somatic/germline variant calling job.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("fastq_dir", fastq_dir)
        mlflow.log_param("bam_dir", bam_dir)
        mlflow.log_param("genome", genome)
        mlflow.log_param("tools", tools)
        mlflow.log_param("step", step)
        mlflow.log_param("analysis_type", analysis_type)
        mlflow.log_param("pipeline_version", pipeline_version)

        job_run_id = execute_workflow(
            job_id=os.environ.get("SAREK_VARIANT_CALLING_JOB_ID", "64553839697857"),
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "fastq_dir": fastq_dir,
                "bam_dir": bam_dir,
                "output_dir": output_dir,
                "genome": genome,
                "tools": tools,
                "step": step,
                "analysis_type": analysis_type,
                "pipeline_version": pipeline_version,
                "extra_args": extra_args,
                "trigger_annotation": "true",
                "qc_gate_enabled": "true",
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "sarek_variant_calling")
        mlflow.set_tag("analysis_type", analysis_type)
        mlflow.set_tag("tools", tools)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id



def start_isoseq_isoform_sequencing(user_info: UserInfo,
                                     input_dir: str,
                                     output_dir: str,
                                     primers_fasta: str,
                                     genome: str,
                                     gtf_annotation: str,
                                     aligner: str,
                                     clustering: str,
                                     pipeline_version: str,
                                     extra_args: str,
                                     mlflow_experiment_name: str,
                                     mlflow_run_name: str):
    """Start an nf-core/isoseq long-read isoform sequencing job.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("input_dir", input_dir)
        mlflow.log_param("genome", genome)
        mlflow.log_param("aligner", aligner)
        mlflow.log_param("clustering", clustering)
        mlflow.log_param("pipeline_version", pipeline_version)

        job_run_id = execute_workflow(
            job_id=os.environ.get("ISOSEQ_ISOFORM_SEQUENCING_JOB_ID", "746656389496912"),
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "input_dir": input_dir,
                "output_dir": output_dir,
                "primers_fasta": primers_fasta,
                "genome": genome,
                "gtf_annotation": gtf_annotation,
                "aligner": aligner,
                "clustering": clustering,
                "pipeline_version": pipeline_version,
                "extra_args": extra_args,
                "qc_gate_enabled": "true",
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "isoseq_isoform_sequencing")
        mlflow.set_tag("aligner", aligner)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id

def search_variant_annotation_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search variant annotation runs by run name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='variant_annotation' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    filtered = runs[runs['tags.mlflow.runName'].str.contains(run_name, case=False, na=False)]
    if len(filtered) == 0:
        return pd.DataFrame()

    filtered['experiment_name'] = filtered['experiment_id'].map(experiments)
    result = filtered[['run_id', 'tags.mlflow.runName', 'experiment_name',
                        'params.variants_table', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'variants_table', 'start_time', 'status']
    return result


def search_variant_annotation_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search variant annotation runs by experiment name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    matching = {exp.experiment_id: exp.name.split("/")[-1]
                for exp in experiment_list
                if experiment_name.upper() in exp.name.split("/")[-1].upper()}

    if len(matching) == 0:
        return pd.DataFrame()

    all_experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(matching.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='variant_annotation' AND "
                        f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(all_experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.variants_table', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'variants_table', 'start_time', 'status']
    return result


def pull_annotation_results(run_id: str, run_name: str = "") -> pd.DataFrame:
    """Pull pathogenic variant results for a given annotation run."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]

    # Use provided run_name, fall back to MLflow tag
    if not run_name:
        run = mlflow.get_run(run_id)
        run_name = run.data.tags.get("run_name", "")

    table = f"{catalog}.{schema}.variant_annotation_pathogenic"

    # Check if ACMG columns (category, condition) exist in the table
    try:
        cols_df = execute_select_query(f"DESCRIBE {table}")
        available_cols = cols_df["col_name"].tolist() if "col_name" in cols_df.columns else []
    except Exception:
        available_cols = []

    extra_cols = ""
    if "category" in available_cols:
        extra_cols = "category, condition, "

    run_name_filter = f"WHERE run_name = '{run_name}'" if run_name else ""

    query = f"""
        SELECT gene, {extra_cols}chromosome, start as position, ref, alt, zygosity,
               array_join(clinical_significance, ', ') as clinical_significance,
               array_join(disease_name, ', ') as disease_name
        FROM {table}
        {run_name_filter}
        ORDER BY gene, position
    """
    return execute_select_query(query)


# ═══════════════════════════════════════════════════════════════════════════════
# GWAS Results Visualization — fetches plots from MLflow artifacts
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_gwas_plots(run_id: str) -> dict:
    """Fetch GWAS plot artifacts (Manhattan, QQ, Volcano) from an MLflow run.
    
    Returns dict of {plot_name: image_bytes}.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    
    plots = {}
    try:
        artifacts = client.list_artifacts(run_id, path="gwas_plots")
        for art in artifacts:
            if art.path.endswith(".png"):
                local_path = client.download_artifacts(run_id, art.path)
                name = os.path.basename(art.path).replace(".png", "").replace("_", " ").title()
                with open(local_path, "rb") as f:
                    plots[name] = f.read()
    except Exception as e:
        print(f"[fetch_gwas_plots] Error: {e}")
    return plots


def fetch_gwas_metrics(run_id: str) -> dict:
    """Fetch GWAS correction metrics from an MLflow run."""
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    
    try:
        run = mlflow.get_run(run_id)
        metrics = {k: v for k, v in run.data.metrics.items() if 'sig_' in k or 'min_pvalue' in k or 'total_variants' in k}
        params = {k: v for k, v in run.data.params.items() if k in ('test_type', 'fdr_threshold', 'contigs', 'hwe_cutoff', 'vcf_path')}
        return {"metrics": metrics, "params": params}
    except Exception as e:
        print(f"[fetch_gwas_metrics] Error: {e}")
        return {"metrics": {}, "params": {}}




# ═══════════════════════════════════════════════════════════════════════════════
# GWAS Inline Plotly Visualizations — render directly from Delta table results
# ═══════════════════════════════════════════════════════════════════════════════

def render_manhattan_plot(df):
    """Render an interactive Manhattan plot from GWAS results DataFrame."""
    import plotly.express as px
    import numpy as np

    if df.empty or "neg_log_pval" not in df.columns:
        return None

    pdf = df.copy()
    pdf = pdf.dropna(subset=["neg_log_pval"])
    if pdf.empty:
        return None

    # Map chromosome names to numeric for x-axis ordering
    chrom_map = {}
    for c in pdf["contigName"].unique():
        try:
            chrom_map[c] = int(c.replace("chr", "").replace("X", "23").replace("Y", "24"))
        except ValueError:
            chrom_map[c] = 99
    pdf["chrom_num"] = pdf["contigName"].map(chrom_map)
    pdf = pdf.sort_values(["chrom_num", "start"])
    pdf["x_pos"] = range(len(pdf))

    # Color alternating chromosomes
    pdf["color"] = pdf["chrom_num"].apply(lambda x: "even" if x % 2 == 0 else "odd")

    fig = px.scatter(
        pdf, x="x_pos", y="neg_log_pval", color="color",
        color_discrete_map={"even": "#1f77b4", "odd": "#00b4d8"},
        hover_data={"contigName": True, "start": True, "neg_log_pval": ":.2f", "x_pos": False, "color": False},
        labels={"neg_log_pval": "-log₁₀(p)", "x_pos": "Genomic Position"},
    )
    # Genome-wide significance line
    fig.add_hline(y=-np.log10(5e-8), line_dash="dash", line_color="red",
                  annotation_text="Genome-wide (5×10⁻⁸)", annotation_position="top left")
    # Suggestive line
    fig.add_hline(y=-np.log10(1e-5), line_dash="dot", line_color="orange",
                  annotation_text="Suggestive (10⁻⁵)", annotation_position="top left")
    fig.update_layout(
        title="Manhattan Plot", showlegend=False,
        xaxis_title="Genomic Position", yaxis_title="-log₁₀(p-value)",
        template="plotly_dark", height=450,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    return fig


def render_qq_plot(df):
    """Render a QQ plot from GWAS results DataFrame."""
    import plotly.graph_objects as go
    import numpy as np

    if df.empty or "pvalue" not in df.columns:
        return None

    pvals = df["pvalue"].dropna().values
    pvals = pvals[pvals > 0]
    if len(pvals) == 0:
        return None

    pvals = np.sort(pvals)
    n = len(pvals)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(pvals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=expected, y=observed, mode="markers",
                             marker=dict(size=4, color="#00b4d8", opacity=0.6),
                             name="Variants"))
    max_val = max(expected.max(), observed.max()) * 1.05
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                             line=dict(dash="dash", color="red"), name="Expected"))
    fig.update_layout(
        title="QQ Plot", template="plotly_dark", height=400,
        xaxis_title="Expected -log₁₀(p)", yaxis_title="Observed -log₁₀(p)",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def render_volcano_plot(df):
    """Render a Volcano plot (effect size vs significance) from GWAS results."""
    import plotly.express as px
    import numpy as np

    if df.empty or "neg_log_pval" not in df.columns or "effect" not in df.columns:
        return None

    pdf = df.dropna(subset=["neg_log_pval", "effect"]).copy()
    if pdf.empty:
        return None

    sig_thresh = -np.log10(5e-8)
    pdf["significant"] = pdf["neg_log_pval"].apply(
        lambda x: "Genome-wide" if x > sig_thresh else ("Suggestive" if x > 5 else "NS")
    )
    color_map = {"Genome-wide": "#ef4444", "Suggestive": "#f59e0b", "NS": "#6b7280"}

    fig = px.scatter(
        pdf, x="effect", y="neg_log_pval", color="significant",
        color_discrete_map=color_map,
        hover_data={"contigName": True, "start": True, "effect": ":.4f", "neg_log_pval": ":.2f", "significant": False},
        labels={"effect": "Effect Size (β)", "neg_log_pval": "-log₁₀(p)"},
    )
    fig.add_hline(y=sig_thresh, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Volcano Plot", template="plotly_dark", height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit Page UI
# ═══════════════════════════════════════════════════════════════════════════════

from utils.streamlit_helper import get_user_info
from utils.authorization import require_module_access

st.title(":material/coronavirus: Disease Biology")

user_info = get_user_info()
require_module_access("disease_biology", user_info)

gwas_tab, gwas_results_tab, vcf_tab, variant_calling_tab, sarek_tab, isoseq_tab, variant_annotation_tab = st.tabs([
    ":material/labs: GWAS Analysis",
    ":material/analytics: GWAS Results",
    ":material/upload_file: VCF Ingestion",
    ":material/biotech: Variant Calling",
    ":material/genetics: Somatic Calling",
    ":material/polyline: Iso-Seq",
    ":material/search: Variant Annotation",
])

# ── GWAS Analysis Tab ─────────────────────────────────────────────────────────

with gwas_tab:
    st.subheader("Genome-Wide Association Study")
    st.caption("Run a GWAS analysis using Glow on Spark. Supports linear, logistic (Firth/LRT), or both regression types.")

    with st.form("gwas_form"):
        col1, col2 = st.columns(2)
        with col1:
            gwas_experiment = st.text_input("MLflow Experiment Name", value="gwas_analysis")
            gwas_run_name = st.text_input("MLflow Run Name", value="")
            # --- Medallion Architecture: VCF File Path ---
            # Bronze = Source of truth (raw ingestion)
            # Silver = QC / error corrected
            # Gold = Analysis-ready (users pull from here)
            # Platinum = Scientist end results
            _GWAS_VOLUME = "/Volumes/dhbl_discovery_us_dev/genesis_schema/gwas_data"
            _VCF_OPTIONS = [
                # Gold tier — analysis-ready (default for users)
                f"{_GWAS_VOLUME}/gold/vcf/Homo_sapiens_assembly38.known_indels.vcf.gz",
                # Bronze tier — raw ingestion / source of truth
                f"{_GWAS_VOLUME}/bronze/vcf/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz",
                # Silver tier — QC'd / filtered
                f"{_GWAS_VOLUME}/silver/vcf/",
                # Platinum tier — scientist outputs
                f"{_GWAS_VOLUME}/platinum/vcf/",
                "Custom path...",
            ]

            def _vcf_label(path: str) -> str:
                if path == "Custom path...":
                    return "✏️ Enter custom path..."
                if "/gold/" in path:
                    return f"🥇 Gold · {path.split('/')[-1]}"
                if "/bronze/" in path:
                    return f"🥉 Bronze · {path.split('/')[-1]}"
                if "/silver/" in path:
                    return "🥈 Silver · (QC-filtered VCFs will appear here)"
                if "/platinum/" in path:
                    return "💎 Platinum · (scientist results will appear here)"
                return path.split("/")[-1]

            vcf_selection = st.selectbox(
                "VCF File Path",
                options=_VCF_OPTIONS,
                format_func=_vcf_label,
                key="gwas_vcf_select",
                help=(
                    "Medallion tiers: 🥉 Bronze = raw ingestion (source of truth), "
                    "🥈 Silver = QC/error-corrected, "
                    "🥇 Gold = analysis-ready (default), "
                    "💎 Platinum = scientist end results."
                )
            )
            if vcf_selection == "Custom path..." or vcf_selection.endswith("/"):
                vcf_path = st.text_input(
                    "Custom VCF Path",
                    value="" if vcf_selection == "Custom path..." else vcf_selection,
                    placeholder=f"{_GWAS_VOLUME}/gold/vcf/my_file.vcf.gz",
                    key="gwas_vcf_custom"
                )
            else:
                vcf_path = vcf_selection

            # --- Medallion Architecture: Phenotype File Path ---
            _PHENO_OPTIONS = [
                # Gold tier — analysis-ready phenotype files
                f"{_GWAS_VOLUME}/gold/phenotype/breast_cancer_phenotype.tsv",
                f"{_GWAS_VOLUME}/gold/phenotype/example_sample_list.tsv",
                # Bronze tier — raw phenotype ingestion
                f"{_GWAS_VOLUME}/bronze/phenotype/",
                # Silver tier — QC'd phenotype
                f"{_GWAS_VOLUME}/silver/phenotype/",
                "Custom path...",
            ]

            def _pheno_label(path: str) -> str:
                if path == "Custom path...":
                    return "✏️ Enter custom path..."
                if "/gold/" in path:
                    return f"🥇 Gold · {path.split('/')[-1]}"
                if "/bronze/" in path:
                    return "🥉 Bronze · (raw phenotype files)"
                if "/silver/" in path:
                    return "🥈 Silver · (QC-validated phenotype files)"
                return path.split("/")[-1]

            phenotype_selection = st.selectbox(
                "Phenotype File Path",
                options=_PHENO_OPTIONS,
                format_func=_pheno_label,
                key="gwas_pheno_select",
                help=(
                    "Medallion tiers: 🥉 Bronze = raw phenotype data, "
                    "🥈 Silver = validated/cleaned, "
                    "🥇 Gold = analysis-ready (default)."
                )
            )
            if phenotype_selection == "Custom path..." or phenotype_selection.endswith("/"):
                phenotype_path = st.text_input(
                    "Custom Phenotype Path",
                    value="" if phenotype_selection == "Custom path..." else phenotype_selection,
                    placeholder=f"{_GWAS_VOLUME}/gold/phenotype/my_phenotype.tsv",
                    key="gwas_pheno_custom"
                )
            else:
                phenotype_path = phenotype_selection
        with col2:
            phenotype_column = st.text_input("Phenotype Column", value="phenotype")
            gwas_contigs = st.text_input("Contigs (comma-separated)", value="6")
            hwe_cutoff = st.text_input("HWE Cutoff", value="0.01")
            pvalue_threshold = st.text_input("P-value Threshold", value="0.01")
            test_type_sel = st.selectbox("Regression Type", ["both", "logistic", "linear"], index=0, help="Linear for quantitative traits, logistic for case/control, or both")
            correction_method_sel = st.selectbox("Logistic Correction", ["Firth", "LRT"], index=0, help="Firth correction (default) or likelihood-ratio test")

        submitted = st.form_submit_button(":material/play_arrow: Start GWAS Analysis", use_container_width=True)
        if submitted and gwas_run_name and vcf_path and phenotype_path:
            with st.spinner("Starting GWAS job..."):
                try:
                    job_run_id = start_gwas_analysis(
                        user_info=user_info,
                        vcf_path=vcf_path,
                        phenotype_path=phenotype_path,
                        phenotype_column=phenotype_column,
                        contigs=gwas_contigs,
                        hwe_cutoff=hwe_cutoff,
                        pvalue_threshold=pvalue_threshold,
                        mlflow_experiment_name=gwas_experiment,
                        mlflow_run_name=gwas_run_name,
                        test_type=test_type_sel,
                        correction_method=correction_method_sel
                    )
                    st.success(f"GWAS job started — Run ID: `{job_run_id}`")
                except Exception as e:
                    st.error(f"Failed to start GWAS: {e}")

    st.divider()
    st.subheader("Search Past Runs")

    # Load all GWAS runs upfront for the dropdown
    with st.spinner("Loading past runs..."):
        all_past_runs = search_gwas_runs_by_run_name(user_info.user_email, "")

    if len(all_past_runs) > 0:
        # Build dropdown with created_by info
        run_labels = [
            f"{row['run_name']} — {row.get('created_by', 'unknown')} ({row['start_time']})"
            for _, row in all_past_runs.iterrows()
        ]
        selected_past_run = st.selectbox("Select a past GWAS run", [""] + run_labels, key="gwas_past_run_select")

        if selected_past_run:
            selected_row = all_past_runs.iloc[run_labels.index(selected_past_run)]
            run_with_progress = add_progress_column(pd.DataFrame([selected_row]), total_steps=3)
            st.markdown(render_runs_html_table(run_with_progress, hidden_columns=["run_id", "progress"]), unsafe_allow_html=True)
    else:
        st.caption("No past GWAS runs found.")

    # Optional text search filter
    with st.expander("Filter by keyword"):
        search_col1, search_col2 = st.columns(2)
        with search_col1:
            gwas_search_by = st.radio("Search by", ["Run Name", "Experiment Name"], horizontal=True, key="gwas_search_by")
        with search_col2:
            gwas_search_query = st.text_input("Search", placeholder="Type to search...", key="gwas_search_query")

        if gwas_search_query:
            with st.spinner("Searching..."):
                if gwas_search_by == "Run Name":
                    results = search_gwas_runs_by_run_name(user_info.user_email, gwas_search_query)
                else:
                    results = search_gwas_runs_by_experiment_name(user_info.user_email, gwas_search_query)

            if len(results) > 0:
                results = add_progress_column(results, total_steps=3)
                st.markdown(render_runs_html_table(results, hidden_columns=["run_id", "progress"]), unsafe_allow_html=True)
            else:
                st.caption("No runs found.")


# ── GWAS Results Visualization Tab ────────────────────────────────────────────

with gwas_results_tab:
    st.subheader("GWAS Results & Visualization")

    # ── Dashboard links ───────────────────────────────────────────
    dash_col1, dash_col2 = st.columns([3, 1])
    with dash_col1:
        st.caption("View Manhattan, QQ, and Volcano plots from completed GWAS runs. Includes Bonferroni and FDR correction summaries.")
    with dash_col2:
        st.link_button(
            ":material/dashboard: Open GWAS Dashboard",
            "https://eisai-lakehouse-us-dev.cloud.databricks.com/sql/dashboardsv3/01f147be34be14d28487266266b6823d?o=3015392934154839",
            use_container_width=True,
        )

    @st.fragment
    def _gwas_results_fragment():
        # Load all completed GWAS runs upfront for the dropdown
        with st.spinner("Loading completed runs..."):
            all_results = search_gwas_runs_by_run_name(user_info.user_email, "")

        all_completed = all_results[all_results["status"] == "gwas_complete"] if len(all_results) > 0 and "status" in all_results.columns else pd.DataFrame()

        if len(all_completed) == 0:
            st.info("No completed GWAS runs found. Run a GWAS analysis first.")
            return

        # Build dropdown options from all completed runs
        run_options = {f"{row['run_name']} — {row.get('created_by', 'unknown')} — {row.get('experiment_name', '')} ({row['start_time']})": row['run_id'] for _, row in all_completed.iterrows()}
        selected = st.selectbox("Select a completed GWAS run", list(run_options.keys()), key="gwas_run_select")

        if not selected:
            return

        run_id = run_options[selected]

        # Fetch metrics
        data = fetch_gwas_metrics(run_id)
        metrics = data["metrics"]
        params = data["params"]

        if params:
            st.markdown("**Run Parameters**")
            param_cols = st.columns(min(6, len(params)))
            for i, (k, v) in enumerate(params.items()):
                param_cols[i % len(param_cols)].metric(k.replace("_", " ").title(), v)

        if metrics:
            st.markdown("**Correction Summary**")
            metric_cols = st.columns(min(4, len(metrics)))
            for i, (k, v) in enumerate(sorted(metrics.items())):
                label = k.replace("_", " ").title()
                metric_cols[i % len(metric_cols)].metric(label, f"{v:,.0f}" if v > 1 else f"{v:.2e}")

        st.divider()

        # Fetch MLflow plot artifacts
        with st.spinner("Loading plots from MLflow..."):
            plots = fetch_gwas_plots(run_id)

        if plots:
            for name, img_bytes in plots.items():
                st.image(img_bytes, caption=name, use_container_width=True)
        else:
            # Fall back to inline results from Delta table
            try:
                import numpy as np
                run_suffix = run_id.replace('-', '_')
                fq = f"{_cfg.catalog}.{_cfg.schema}"
                # Try table naming conventions in order of preference
                table_candidates = [
                    (f"gwas_linear_corrected_{run_suffix}", "Linear (Corrected)"),
                    (f"gwas_logistic_corrected_{run_suffix}", "Logistic (Corrected)"),
                    (f"gwas_results_{run_suffix}", "Combined Results"),
                ]
                rendered = False
                for table_name, label in table_candidates:
                    try:
                        results_df = execute_select_query(
                            f"SELECT contigName, start, pvalue, effect "
                            f"FROM {fq}.{table_name} "
                            f"WHERE pvalue IS NOT NULL AND pvalue > 0 "
                            f"ORDER BY pvalue LIMIT 100"
                        )
                        if len(results_df) > 0:
                            st.markdown(f"**{label}** ({len(results_df)} variants)")
                            results_df["pvalue"] = pd.to_numeric(results_df["pvalue"], errors="coerce")
                            results_df["effect"] = pd.to_numeric(results_df["effect"], errors="coerce")
                            results_df["neg_log_p"] = -np.log10(results_df["pvalue"].clip(lower=1e-300))
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                column_config={
                                    "neg_log_p": st.column_config.NumberColumn("-log10(p)", format="%.2f"),
                                    "pvalue": st.column_config.NumberColumn("P-value", format="%.2e"),
                                    "effect": st.column_config.NumberColumn("Effect", format="%.4f"),
                                }
                            )
                            rendered = True
                    except Exception:
                        continue
                if not rendered:
                    st.caption("No results tables found for this run.")
                else:
                    # Render inline Plotly plots from the results data
                    try:
                        plot_query = (
                            f"SELECT contigName, start, pvalue, effect "
                            f"FROM {fq}.gwas_results_{run_suffix} "
                            f"WHERE pvalue IS NOT NULL AND pvalue > 0 "
                            f"ORDER BY pvalue LIMIT 10000"
                        )
                        plot_df = execute_select_query(plot_query)
                        if len(plot_df) == 0:
                            # Try linear corrected as fallback
                            plot_query = (
                                f"SELECT contigName, start, pvalue, effect "
                                f"FROM {fq}.gwas_linear_corrected_{run_suffix} "
                                f"WHERE pvalue IS NOT NULL AND pvalue > 0 "
                                f"ORDER BY pvalue LIMIT 10000"
                            )
                            plot_df = execute_select_query(plot_query)

                        if len(plot_df) > 0:
                            plot_df["pvalue"] = pd.to_numeric(plot_df["pvalue"], errors="coerce")
                            plot_df["effect"] = pd.to_numeric(plot_df["effect"], errors="coerce")
                            plot_df = plot_df.dropna(subset=["pvalue"])
                            plot_df["neg_log_pval"] = -np.log10(plot_df["pvalue"].clip(lower=1e-300))

                            st.divider()
                            st.markdown("**Interactive Plots**")
                            plot_col1, plot_col2 = st.columns(2)
                            with plot_col1:
                                manhattan_fig = render_manhattan_plot(plot_df)
                                if manhattan_fig:
                                    st.plotly_chart(manhattan_fig, use_container_width=True)
                            with plot_col2:
                                qq_fig = render_qq_plot(plot_df)
                                if qq_fig:
                                    st.plotly_chart(qq_fig, use_container_width=True)
                            volcano_fig = render_volcano_plot(plot_df)
                            if volcano_fig:
                                st.plotly_chart(volcano_fig, use_container_width=True)
                    except Exception as plot_err:
                        st.caption(f"Could not render plots: {plot_err}")
            except Exception as e:
                st.caption(f"Could not load results: {e}")

    _gwas_results_fragment()



# ── VCF Ingestion Tab ─────────────────────────────────────────────────────────

with vcf_tab:
    st.subheader("VCF Ingestion")
    st.caption("Convert VCF files to Delta tables for downstream analysis.")

    with st.form("vcf_form"):
        col1, col2 = st.columns(2)
        with col1:
            vcf_experiment = st.text_input("MLflow Experiment Name", value="vcf_ingestion", key="vcf_exp")
            vcf_run_name = st.text_input("MLflow Run Name", value="", key="vcf_run")
        with col2:
            vcf_input_path = st.text_input("VCF File Path", placeholder="/Volumes/...", key="vcf_input")
            vcf_output_table = st.text_input("Output Table Name", placeholder="my_variants", key="vcf_output")

        if st.form_submit_button(":material/play_arrow: Start VCF Ingestion", use_container_width=True):
            if vcf_run_name and vcf_input_path and vcf_output_table:
                with st.spinner("Starting VCF ingestion..."):
                    try:
                        job_run_id = start_vcf_ingestion(
                            user_info=user_info,
                            vcf_path=vcf_input_path,
                            output_table_name=vcf_output_table,
                            mlflow_experiment_name=vcf_experiment,
                            mlflow_run_name=vcf_run_name
                        )
                        st.success(f"VCF ingestion started — Run ID: `{job_run_id}`")
                    except Exception as e:
                        st.error(f"Failed: {e}")


# ── Variant Calling Tab ───────────────────────────────────────────────────────

with variant_calling_tab:
    st.subheader("Variant Calling (Parabricks)")
    st.caption("GPU-accelerated alignment and variant calling using NVIDIA Parabricks.")

    with st.form("parabricks_form"):
        col1, col2 = st.columns(2)
        with col1:
            pb_experiment = st.text_input("MLflow Experiment Name", value="variant_calling", key="pb_exp")
            pb_run_name = st.text_input("MLflow Run Name", value="", key="pb_run")
            fastq_r1 = st.text_input("FASTQ R1 Path", placeholder="/Volumes/...", key="pb_r1")
        with col2:
            fastq_r2 = st.text_input("FASTQ R2 Path", placeholder="/Volumes/...", key="pb_r2")
            ref_genome = st.text_input("Reference Genome", value="/dbfs/genesis_workbench/gwas/reference/grch38/GRCh38_full_analysis_set_plus_decoy_hla.fa", key="pb_ref")
            output_vol = st.text_input("Output Volume", placeholder="/Volumes/...", key="pb_out")

        if st.form_submit_button(":material/play_arrow: Start Variant Calling", use_container_width=True):
            if pb_run_name and fastq_r1 and fastq_r2:
                with st.spinner("Starting Parabricks..."):
                    try:
                        job_run_id = start_parabricks_alignment(
                            user_info=user_info,
                            fastq_r1=fastq_r1,
                            fastq_r2=fastq_r2,
                            reference_genome_path=ref_genome,
                            output_volume_path=output_vol,
                            mlflow_experiment_name=pb_experiment,
                            mlflow_run_name=pb_run_name
                        )
                        st.success(f"Variant calling started — Run ID: `{job_run_id}`")
                    except Exception as e:
                        st.error(f"Failed: {e}")


    st.divider()
    # ── VEP + ANNOVAR v2 form ───────────────────────────────────────────────────────
    st.subheader("VEP 115 + ANNOVAR Annotation (v2)")
    st.caption("Full-spectrum variant annotation: functional consequences (VEP), population frequencies, clinical significance, deleteriousness scores (ANNOVAR).")

    with st.form("vep_annovar_form"):
        v2_col1, v2_col2 = st.columns(2)
        with v2_col1:
            v2_experiment = st.text_input("MLflow Experiment Name", value="variant_annotation_v2", key="v2_exp")
            v2_run_name = st.text_input("MLflow Run Name", value="", key="v2_run")
            v2_vcf_path = st.text_input("VCF File Path", placeholder="/Volumes/...", key="v2_vcf")
            v2_genome = st.selectbox("Genome Build", ["GRCh38", "GRCh37"], index=0, key="v2_genome")
        with v2_col2:
            v2_tools = st.selectbox("Annotation Tools", ["both", "vep_only", "annovar_only"], index=0, key="v2_tools",
                                    help="Run VEP only, ANNOVAR only, or both")
            v2_protocols = st.text_input("ANNOVAR Protocols",
                value="refGene,gnomad40_genome,clinvar_20240917,dbnsfp42a,cosmic70", key="v2_protocols")
            v2_operations = st.text_input("ANNOVAR Operations", value="g,f,f,f,f", key="v2_ops")
            v2_vep_flags = st.text_input("VEP Extra Flags (optional)", value="", key="v2_flags")

        if st.form_submit_button(":material/play_arrow: Start VEP + ANNOVAR Annotation", use_container_width=True):
            if v2_run_name and v2_vcf_path:
                with st.spinner("Starting VEP + ANNOVAR annotation..."):
                    try:
                        job_run_id = start_vep_annovar_annotation(
                            user_info=user_info,
                            vcf_path=v2_vcf_path,
                            genome_build=v2_genome,
                            annotation_tools=v2_tools,
                            annovar_protocols=v2_protocols,
                            annovar_operations=v2_operations,
                            vep_extra_flags=v2_vep_flags,
                            mlflow_experiment_name=v2_experiment,
                            mlflow_run_name=v2_run_name,
                        )
                        st.success(f"VEP + ANNOVAR annotation started — Run ID: `{job_run_id}`")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    st.divider()
    st.subheader("Search VEP + ANNOVAR Runs")
    v2s_col1, v2s_col2 = st.columns(2)
    with v2s_col1:
        v2_search_by = st.radio("Search by", ["Run Name", "Experiment Name"], horizontal=True, key="v2_search_by")
    with v2s_col2:
        v2_search_query = st.text_input("Search", placeholder="Type to search...", key="v2_search_query")

    if v2_search_query:
        with st.spinner("Searching..."):
            if v2_search_by == "Run Name":
                v2_results = search_vep_annovar_runs_by_run_name(user_info.user_email, v2_search_query)
            else:
                v2_results = search_vep_annovar_runs_by_experiment_name(user_info.user_email, v2_search_query)

        if len(v2_results) > 0:
            v2_results = add_progress_column(v2_results, total_steps=3)
            st.markdown(render_runs_html_table(v2_results, hidden_columns=["run_id", "progress"]), unsafe_allow_html=True)

            # Show results for completed runs
            completed = v2_results[v2_results["status"] == "annotation_v2_complete"]
            if len(completed) > 0:
                v2_run_options = {f"{row['run_name']} ({row['start_time']})": row['run_id'] for _, row in completed.iterrows()}
                v2_selected = st.selectbox("View results for", list(v2_run_options.keys()), key="v2_result_select")
                if v2_selected:
                    v2_run_id = v2_run_options[v2_selected]
                    v2_run_name_display = v2_selected.split(" (")[0]
                    with st.spinner("Loading annotation results..."):
                        ann_df = pull_vep_annovar_results(v2_run_id, v2_run_name_display)
                    if len(ann_df) > 0:
                        # Impact summary
                        if "combined_impact" in ann_df.columns:
                            impact_counts = ann_df["combined_impact"].value_counts()
                            imp_cols = st.columns(min(4, len(impact_counts)))
                            for i, (impact, count) in enumerate(impact_counts.items()):
                                imp_cols[i % len(imp_cols)].metric(impact, f"{count:,}")
                        st.dataframe(ann_df, use_container_width=True, height=400)
                    else:
                        st.caption("No annotation results found for this run.")
        else:
            st.caption("No VEP + ANNOVAR runs found.")



# ── Somatic Variant Calling Tab (nf-core/sarek) ──────────────────────────────

with sarek_tab:
    st.subheader("Somatic & Germline Variant Calling (nf-core/sarek)")
    st.caption("Multi-caller variant calling via nf-core/sarek. Supports Mutect2, MuSE, Strelka, FreeBayes (somatic) and HaplotypeCaller, DeepVariant (germline).")

    with st.form("sarek_form"):
        col1, col2 = st.columns(2)
        with col1:
            sarek_experiment = st.text_input("MLflow Experiment Name", value="sarek_variant_calling", key="sarek_exp")
            sarek_run_name = st.text_input("MLflow Run Name", value="", key="sarek_run")
            sarek_fastq = st.text_input("FASTQ Directory", placeholder="/Volumes/catalog/schema/volume/fastqs/", key="sarek_fq",
                                        help="Directory containing paired-end FASTQ files (tumor + normal)")
            sarek_bam = st.text_input("BAM Directory (alternative)", placeholder="/Volumes/.../bams/", key="sarek_bam",
                                      help="Use pre-aligned BAMs instead of FASTQ (leave FASTQ empty)")
            sarek_output = st.text_input("Output Directory", placeholder="/Volumes/catalog/schema/volume/sarek_output/", key="sarek_out")
        with col2:
            sarek_genome = st.selectbox("Reference Genome", ["GRCh38", "GRCh37", "GRCm38", "GRCm39"], index=0, key="sarek_genome")
            sarek_analysis = st.selectbox("Analysis Type", ["somatic", "germline", "both"], index=0, key="sarek_analysis",
                                          help="Somatic: requires tumor/normal pairs. Germline: all samples treated as normal.")
            sarek_tools = st.text_input("Variant Callers (--tools)", value="mutect2,muse,strelka,vep", key="sarek_tools",
                                        help="Comma-separated: mutect2, muse, strelka, freebayes, haplotypecaller, deepvariant, manta, tiddit, ascat, controlfreec, vep, snpeff")
            sarek_step = st.selectbox("Pipeline Step", ["mapping", "markduplicates", "recalibrate", "variant_calling", "annotate"],
                                      index=0, key="sarek_step", help="Start from this step (use 'variant_calling' for pre-processed BAMs)")
            sarek_version = st.text_input("nf-core/sarek Version", value="3.5.1", key="sarek_ver")
            sarek_extra = st.text_input("Extra Arguments", value="", key="sarek_extra",
                                        help="Additional nextflow args (e.g. --wes --intervals /path/to/bed)")

        if st.form_submit_button(":material/play_arrow: Start Sarek Variant Calling", use_container_width=True):
            if sarek_run_name and (sarek_fastq or sarek_bam) and sarek_output:
                with st.spinner("Starting nf-core/sarek..."):
                    try:
                        job_run_id = start_sarek_variant_calling(
                            user_info=user_info,
                            fastq_dir=sarek_fastq,
                            bam_dir=sarek_bam,
                            output_dir=sarek_output,
                            genome=sarek_genome,
                            tools=sarek_tools,
                            step=sarek_step,
                            analysis_type=sarek_analysis,
                            pipeline_version=sarek_version,
                            extra_args=sarek_extra,
                            mlflow_experiment_name=sarek_experiment,
                            mlflow_run_name=sarek_run_name,
                        )
                        st.success(f"Sarek variant calling started — Run ID: `{job_run_id}`")
                    except Exception as e:
                        st.error(f"Failed: {e}")
            else:
                st.warning("Please provide Run Name, input path (FASTQ or BAM), and output directory.")

    st.divider()
    st.info(
        "**Naming convention for somatic analysis:**\n"
        "- Tumor samples: `*_T_*`, `*_tumor*`, or `*-T-*`\n"
        "- Normal samples: `*_N_*`, `*_normal*`, or `*-N-*`\n"
        "- Patient pairing derived from common filename prefix\n\n"
        "MuSE and Mutect2 require matched tumor/normal pairs.",
        icon=":material/info:"
    )



# ── Iso-Seq Isoform Sequencing Tab (nf-core/isoseq) ──────────────────────────

with isoseq_tab:
    st.subheader("PacBio Iso-Seq Isoform Sequencing (nf-core/isoseq)")
    st.caption("Full-length transcript isoform discovery from PacBio HiFi/CCS long reads. Processes through Lima, Refine, Cluster, Alignment, Collapse, and SQANTI3 classification.")

    with st.form("isoseq_form"):
        col1, col2 = st.columns(2)
        with col1:
            isoseq_experiment = st.text_input("MLflow Experiment Name", value="isoseq_isoform_sequencing", key="isoseq_exp")
            isoseq_run_name = st.text_input("MLflow Run Name", value="", key="isoseq_run")
            isoseq_input = st.text_input("Input Directory (BAM/FASTQ)", placeholder="/Volumes/catalog/schema/volume/pacbio_data/", key="isoseq_input",
                                         help="Directory containing PacBio HiFi BAM (.bam) or CCS FASTQ (.fastq.gz) files")
            isoseq_output = st.text_input("Output Directory", placeholder="/Volumes/catalog/schema/volume/isoseq_output/", key="isoseq_out")
            isoseq_primers = st.text_input("Primers FASTA (optional)", placeholder="/Volumes/.../primers.fasta", key="isoseq_primers",
                                           help="Primer sequences for Lima demultiplexing. Leave empty to skip Lima or use pipeline defaults.")
            isoseq_gtf = st.text_input("Reference GTF (optional)", placeholder="/Volumes/.../annotation.gtf", key="isoseq_gtf",
                                       help="Custom GTF annotation for SQANTI3 classification. Leave empty for pipeline default.")
        with col2:
            isoseq_genome = st.selectbox("Reference Genome", ["GRCh38", "GRCh37", "GRCm38", "GRCm39"], index=0, key="isoseq_genome")
            isoseq_aligner = st.selectbox("Long-Read Aligner", ["pbmm2", "minimap2"], index=0, key="isoseq_aligner",
                                          help="pbmm2: PacBio-optimized. minimap2: general long-read aligner.")
            isoseq_clustering = st.selectbox("Enable Clustering", ["true", "false"], index=0, key="isoseq_cluster",
                                             help="Cluster full-length reads into isoforms (isoseq3 cluster)")
            isoseq_version = st.text_input("nf-core/isoseq Version", value="1.0.0", key="isoseq_ver")
            isoseq_extra = st.text_input("Extra Arguments", value="", key="isoseq_extra",
                                         help="Additional nextflow args (e.g. --skip_lima, --skip_sqanti)")

        if st.form_submit_button(":material/play_arrow: Start Iso-Seq Analysis", use_container_width=True):
            if isoseq_run_name and isoseq_input and isoseq_output:
                with st.spinner("Starting nf-core/isoseq..."):
                    try:
                        job_run_id = start_isoseq_isoform_sequencing(
                            user_info=user_info,
                            input_dir=isoseq_input,
                            output_dir=isoseq_output,
                            primers_fasta=isoseq_primers,
                            genome=isoseq_genome,
                            gtf_annotation=isoseq_gtf,
                            aligner=isoseq_aligner,
                            clustering=isoseq_clustering,
                            pipeline_version=isoseq_version,
                            extra_args=isoseq_extra,
                            mlflow_experiment_name=isoseq_experiment,
                            mlflow_run_name=isoseq_run_name,
                        )
                        st.success(f"Iso-Seq analysis started — Run ID: `{job_run_id}`")
                    except Exception as e:
                        st.error(f"Failed: {e}")
            else:
                st.warning("Please provide Run Name, Input Directory, and Output Directory.")

    st.divider()
    st.info(
        "**Input formats:**\n"
        "- **HiFi BAM** (`.bam`): Raw CCS reads from PacBio Sequel II/IIe or Revio\n"
        "- **FASTQ** (`.fastq.gz`): Converted CCS reads\n\n"
        "**SQANTI3 output categories:** FSM (full-splice match), ISM (incomplete-splice match), "
        "NIC (novel in catalog), NNC (novel not in catalog), antisense, intergenic, fusion\n\n"
        "**Compute note:** Requires classic compute (m5.4xlarge). PacBio data is large — "
        "ensure sufficient disk for 50-100GB per SMRT cell.",
        icon=":material/info:"
    )


# ── Variant Annotation Tab ────────────────────────────────────────────────────

with variant_annotation_tab:
    st.subheader("Variant Annotation")

    # ── VEP 115 + ANNOVAR (v2) ────────────────────────────────────────────────
    with st.expander("🧬 VEP 115 + ANNOVAR — Full-Spectrum Annotation (v2)", expanded=True):
        st.caption("Comprehensive variant annotation using Ensembl VEP 115 (consequence, SIFT, PolyPhen, gnomAD) and ANNOVAR (gene-based, filter-based multi-database).")

        with st.form("vep_annovar_annotation_tab_form"):
            col1, col2 = st.columns(2)
            with col1:
                v2_experiment = st.text_input("MLflow Experiment Name", value="variant_annotation_v2", key="v2a_exp")
                v2_run_name = st.text_input("MLflow Run Name", value="", key="v2a_run")
                v2_vcf_path = st.text_input("VCF File Path", placeholder="/Volumes/catalog/schema/volume/file.vcf.gz", key="v2a_vcf")
                v2_genome = st.selectbox("Genome Build", ["GRCh38", "GRCh37"], index=0, key="v2a_genome")
            with col2:
                v2_tools = st.selectbox("Annotation Tools", ["both", "vep_only", "annovar_only"], index=0, key="v2a_tools",
                                        help="Run VEP only, ANNOVAR only, or both")
                v2_protocols = st.text_input("ANNOVAR Protocols", value="refGene,gnomad40_genome,clinvar_20240917,dbnsfp42a,cosmic70", key="v2a_protocols")
                v2_operations = st.text_input("ANNOVAR Operations", value="g,f,f,f,f", key="v2a_ops")
                v2_vep_flags = st.text_input("VEP Extra Flags (optional)", value="", key="v2a_vep_flags",
                                             help="Additional VEP command-line flags (space-separated)")

            if st.form_submit_button(":material/play_arrow: Start VEP + ANNOVAR Annotation", use_container_width=True):
                if v2_run_name and v2_vcf_path:
                    with st.spinner("Starting VEP + ANNOVAR annotation..."):
                        try:
                            job_run_id = start_vep_annovar_annotation(
                                user_info=user_info,
                                vcf_path=v2_vcf_path,
                                genome_build=v2_genome,
                                annotation_tools=v2_tools,
                                annovar_protocols=v2_protocols,
                                annovar_operations=v2_operations,
                                vep_extra_flags=v2_vep_flags,
                                mlflow_experiment_name=v2_experiment,
                                mlflow_run_name=v2_run_name
                            )
                            st.success(f"VEP + ANNOVAR annotation started — Run ID: `{job_run_id}`")
                        except Exception as e:
                            st.error(f"Failed: {e}")
                else:
                    st.warning("Please provide both a Run Name and VCF File Path.")

    st.divider()

    # ── ClinVar + ACMG (v1) ──────────────────────────────────────────────────
    with st.expander("🔍 ClinVar + ACMG — Pathogenicity Lookup (v1)"):
        st.caption("Cross-reference variants against ClinVar and ACMG SF v3.2 secondary findings.")

        with st.form("annotation_form"):
            col1, col2 = st.columns(2)
            with col1:
                ann_experiment = st.text_input("MLflow Experiment Name", value="variant_annotation", key="ann_exp")
                ann_run_name = st.text_input("MLflow Run Name", value="", key="ann_run")
                ann_mode = st.selectbox("Gene Panel", ["ACMG SF v3.2", "Custom"], key="ann_mode")
            with col2:
                ann_table = st.text_input("Variants Table", placeholder="gwas_results_xxx", key="ann_table")
                ann_genes = st.text_area("Custom Gene Regions (JSON)", value="", key="ann_genes", disabled=ann_mode == "ACMG SF v3.2")
                pathogenic_vcf = st.text_input("Pathogenic VCF Path", value="", key="ann_vcf")

            if st.form_submit_button(":material/play_arrow: Start Annotation", use_container_width=True):
                if ann_run_name and ann_table:
                    with st.spinner("Starting annotation..."):
                        try:
                            job_run_id = start_variant_annotation(
                                user_info=user_info,
                                variants_table=ann_table,
                                gene_regions=ann_genes,
                                pathogenic_vcf_path=pathogenic_vcf,
                                mlflow_experiment_name=ann_experiment,
                                mlflow_run_name=ann_run_name,
                                gene_panel_mode="acmg" if ann_mode == "ACMG SF v3.2" else "custom"
                            )
                            st.success(f"Annotation started — Run ID: `{job_run_id}`")
                        except Exception as e:
                            st.error(f"Failed: {e}")
