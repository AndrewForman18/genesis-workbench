"""
Nextflow Pipeline Runner — orchestrates nf-core/scrnaseq on Databricks.
Bridges the gap between SRA FASTQ downloads and scanpy h5ad analysis.

Pipeline: FASTQ files → nf-core/scrnaseq (STARsolo/Salmon/kallisto) → count matrix → h5ad
"""
import os
import re
import csv
import glob
import subprocess
import time
from typing import List, Dict, Optional
from genesis_config import GenesisConfig

_cfg = GenesisConfig.load()


# Supported aligners in nf-core/scrnaseq
ALIGNERS = {
    "starsolo": {
        "name": "STARsolo",
        "description": "STAR-based alignment with Cell Ranger-compatible output",
        "best_for": "10x Chromium, Drop-seq, high accuracy",
        "speed": "Medium",
    },
    "cellranger": {
        "name": "Cell Ranger",
        "description": "10x Genomics official pipeline (requires Cell Ranger binary)",
        "best_for": "10x Chromium (official)",
        "speed": "Slow",
    },
    "salmon": {
        "name": "Salmon alevin",
        "description": "Lightweight pseudo-alignment with Salmon + alevin-fry",
        "best_for": "Fast quantification, low memory",
        "speed": "Fast",
    },
    "kallisto": {
        "name": "kallisto bustools",
        "description": "Ultrafast pseudo-alignment with kallisto + bustools",
        "best_for": "Speed, iterative analysis",
        "speed": "Fastest",
    },
}

# Common 10x Chromium chemistry versions
CHEMISTRY_PRESETS = {
    "auto": "Auto-detect (default)",
    "10XV2": "10x Chromium Single Cell 3\' v2",
    "10XV3": "10x Chromium Single Cell 3\' v3",
    "10XV3HT": "10x Chromium Single Cell 3\' v3 HT",
    "10XV5": "10x Chromium Single Cell 5\' v1/v2",
}

# Reference genomes
GENOME_PRESETS = {
    "GRCh38": "Human (GRCh38/hg38)",
    "GRCm39": "Mouse (GRCm39/mm39)",
    "GRCm38": "Mouse (GRCm38/mm10)",
}


# ── Compute tiers for on-demand Nextflow clusters ──
# Each tier maps to an AWS instance type sized for different study volumes.
# The Streamlit UI exposes these as a simple dropdown; the Jobs API spins up
# a fresh cluster matching the selected tier and auto-terminates when done.
COMPUTE_TIERS = {
    "small": {
        "label": "Small — 8 vCPU / 64 GB",
        "node_type_id": "r5.2xlarge",
        "description": "Single sample, <20M reads",
        "vcpus": 8,
        "ram_gb": 64,
    },
    "medium": {
        "label": "Medium — 16 vCPU / 128 GB",
        "node_type_id": "r5.4xlarge",
        "description": "Typical 10x study (1–5 samples)",
        "vcpus": 16,
        "ram_gb": 128,
    },
    "large": {
        "label": "Large — 32 vCPU / 256 GB",
        "node_type_id": "r5.8xlarge",
        "description": "Multi-sample / large study",
        "vcpus": 32,
        "ram_gb": 256,
    },
    "xlarge": {
        "label": "XL — 48 vCPU / 384 GB",
        "node_type_id": "r5.12xlarge",
        "description": "Whole-cohort alignment, very large datasets",
        "vcpus": 48,
        "ram_gb": 384,
    },
}

# Init scripts required for Nextflow clusters
NEXTFLOW_INIT_SCRIPTS = [
    f"{_cfg.volume('scrnaseq_data')}/init_scripts/install_nextflow.sh",
    f"{_cfg.volume('scrnaseq_data')}/init_scripts/install_sra_tools.sh",
]


def _get_pool_ids() -> Dict:
    """
    Load instance pool IDs from the NEXTFLOW_POOL_IDS env var.

    Expected format (JSON):  {"small": "0501-...", "medium": "0501-...", ...}

    If the env var is unset or invalid, returns an empty dict and clusters
    will spin up without a pool (cold start, ~5-8 min instead of ~2 min).
    """
    import json
    raw = os.environ.get("NEXTFLOW_POOL_IDS", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def build_new_cluster_spec(tier: str = "medium", spark_version: str = "16.4.x-scala2.12") -> Dict:
    """
    Build a new_cluster spec for the Databricks Jobs API based on a compute tier.

    The cluster is single-node (num_workers=0) since Nextflow orchestrates
    its own parallelism via local executor.  Autoscaling local storage is
    enabled so large intermediate FASTQ/BAM files don't fill the root disk.

    If an instance pool is configured for this tier (via NEXTFLOW_POOL_IDS
    env var), the cluster will use it for faster spin-up (~2 min vs ~5-8 min).
    Otherwise it falls back to on-demand provisioning.

    Args:
        tier: One of small, medium, large, xlarge (see COMPUTE_TIERS)
        spark_version: Databricks Runtime version

    Returns:
        Dict suitable for the new_cluster field in SubmitTask
    """
    config = COMPUTE_TIERS.get(tier, COMPUTE_TIERS["medium"])
    pool_ids = _get_pool_ids()
    pool_id = pool_ids.get(tier)

    spec = {
        "spark_version": spark_version,
        "node_type_id": config["node_type_id"],
        "num_workers": 0,  # single-node; Nextflow handles parallelism
        "spark_conf": {
            "spark.master": "local[*]",
            "spark.databricks.cluster.profile": "singleNode",
        },
        "custom_tags": {
            "ResourceClass": "SingleNode",
            "genesis_workbench": "nextflow",
            "compute_tier": tier,
        },
        "enable_elastic_disk": True,
        "init_scripts": [
            {"volumes": {"destination": path}} for path in NEXTFLOW_INIT_SCRIPTS
        ],
        "data_security_mode": "SINGLE_USER",
    }

    if pool_id:
        spec["instance_pool_id"] = pool_id
        # When using a pool, node_type_id is inherited from the pool
        del spec["node_type_id"]

    return spec


def build_samplesheet(
    fastq_dir: str,
    output_path: str,
    sample_name: Optional[str] = None,
) -> Dict:
    """
    Scan a directory for FASTQ files and build an nf-core/scrnaseq samplesheet CSV.

    Expects 10x-style naming: {sample}_S1_L001_R1_001.fastq.gz
    Also supports SRA-style: {SRR}_1.fastq, {SRR}_2.fastq

    Args:
        fastq_dir: Directory containing FASTQ files
        output_path: Where to write the samplesheet CSV
        sample_name: Override sample name (default: infer from filenames)

    Returns:
        Dict with n_samples, n_fastq_pairs, samplesheet_path, samples list
    """
    if not os.path.isdir(fastq_dir):
        return {"error": f"Directory not found: {fastq_dir}"}

    # Find all FASTQ files
    fastq_files = []
    for pattern in ["**/*.fastq.gz", "**/*.fastq", "**/*.fq.gz", "**/*.fq"]:
        fastq_files.extend(glob.glob(os.path.join(fastq_dir, pattern), recursive=True))

    if not fastq_files:
        return {"error": f"No FASTQ files found in {fastq_dir}"}

    # Group into R1/R2 pairs
    pairs = {}
    for f in sorted(fastq_files):
        basename = os.path.basename(f)

        # 10x style: Sample_S1_L001_R1_001.fastq.gz
        m = re.match(r"(.+?)_S\d+_L(\d+)_R([12])_\d+\.fastq", basename)
        if m:
            sample = sample_name or m.group(1)
            lane = m.group(2)
            read = m.group(3)
            key = f"{sample}_L{lane}"
            pairs.setdefault(key, {"sample": sample, "fastq_1": "", "fastq_2": ""})
            pairs[key][f"fastq_{read}"] = f
            continue

        # SRA style: SRR123456_1.fastq or SRR123456_1.fastq.gz
        m = re.match(r"(SRR\d+)[_.]([12])\.f(ast)?q", basename)
        if m:
            sample = sample_name or m.group(1)
            read = m.group(2)
            key = sample
            pairs.setdefault(key, {"sample": sample, "fastq_1": "", "fastq_2": ""})
            pairs[key][f"fastq_{read}"] = f
            continue

        # Generic: anything_R1/R2 or anything_1/2
        m = re.match(r"(.+?)[_.]R?([12])\.f(ast)?q", basename)
        if m:
            sample = sample_name or m.group(1)
            read = m.group(2)
            key = sample
            pairs.setdefault(key, {"sample": sample, "fastq_1": "", "fastq_2": ""})
            pairs[key][f"fastq_{read}"] = f

    if not pairs:
        return {"error": "Could not pair FASTQ files (expected R1/R2 naming convention)"}

    # Write samplesheet CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    samples_list = []

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["sample", "fastq_1", "fastq_2"])
        writer.writeheader()

        for key, pair in sorted(pairs.items()):
            if pair["fastq_1"]:  # At minimum need R1
                writer.writerow(pair)
                samples_list.append(pair)

    return {
        "samplesheet_path": output_path,
        "n_samples": len(set(p["sample"] for p in samples_list)),
        "n_fastq_pairs": len(samples_list),
        "samples": samples_list,
    }


def build_nextflow_command(
    samplesheet_path: str,
    output_dir: str,
    aligner: str = "starsolo",
    genome: str = "GRCh38",
    chemistry: str = "auto",
    pipeline_version: str = "2.7.1",
    extra_args: Optional[str] = None,
) -> List[str]:
    """
    Build the nextflow run command for nf-core/scrnaseq.

    Args:
        samplesheet_path: Path to samplesheet CSV
        output_dir: Output directory in UC Volume
        aligner: One of: starsolo, cellranger, salmon, kallisto
        genome: Reference genome (GRCh38, GRCm39, GRCm38)
        chemistry: 10x chemistry version or 'auto'
        pipeline_version: nf-core/scrnaseq version
        extra_args: Additional Nextflow arguments

    Returns:
        List of command parts for subprocess
    """
    cmd = [
        "nextflow", "run",
        f"nf-core/scrnaseq",
        "-r", pipeline_version,
        "-profile", "conda",
        "--input", samplesheet_path,
        "--outdir", output_dir,
        "--aligner", "star" if aligner == "starsolo" else aligner,
        "--genome", genome,
    ]

    # Protocol/chemistry setting (nf-core/scrnaseq uses --protocol)
    if chemistry and chemistry != "auto":
        cmd.extend(["--protocol", chemistry])

    # Nextflow-level options
    cmd.extend([
        "-work-dir", os.path.join(output_dir, "work"),
        "-with-report", os.path.join(output_dir, "nextflow_report.html"),
        "-with-timeline", os.path.join(output_dir, "nextflow_timeline.html"),
    ])

    if extra_args:
        cmd.extend(extra_args.split())

    return cmd


def run_nextflow_pipeline(
    fastq_dir: str,
    output_dir: str,
    aligner: str = "starsolo",
    genome: str = "GRCh38",
    chemistry: str = "auto",
    sample_name: Optional[str] = None,
    pipeline_version: str = "2.7.1",
    extra_args: Optional[str] = None,
) -> Dict:
    """
    End-to-end: build samplesheet → run nf-core/scrnaseq → locate outputs.

    Args:
        fastq_dir: Directory with FASTQ files (from SRA download)
        output_dir: UC Volume output directory
        aligner: Aligner to use (starsolo, cellranger, salmon, kallisto)
        genome: Reference genome
        chemistry: 10x chemistry version
        sample_name: Override sample name
        pipeline_version: nf-core/scrnaseq version
        extra_args: Additional Nextflow arguments

    Returns:
        Dict with status, output paths, logs, and any errors
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build samplesheet
    samplesheet_path = os.path.join(output_dir, "samplesheet.csv")
    sheet_result = build_samplesheet(fastq_dir, samplesheet_path, sample_name)
    if "error" in sheet_result:
        return {"status": "failed", "step": "samplesheet", "error": sheet_result["error"]}

    # Step 2: Build command
    cmd = build_nextflow_command(
        samplesheet_path=samplesheet_path,
        output_dir=output_dir,
        aligner=aligner,
        genome=genome,
        chemistry=chemistry,
        pipeline_version=pipeline_version,
        extra_args=extra_args,
    )

    # Step 3: Run pipeline
    start_time = time.time()
    log_path = os.path.join(output_dir, "nextflow_run.log")

    try:
        with open(log_path, "w") as log_file:
            process = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=14400,  # 4-hour timeout
                cwd=output_dir,
            )

        elapsed = time.time() - start_time

        if process.returncode != 0:
            # Read last 50 lines of log for error context
            with open(log_path, "r") as f:
                lines = f.readlines()
                error_context = "".join(lines[-50:])

            return {
                "status": "failed",
                "step": "nextflow",
                "returncode": process.returncode,
                "elapsed_minutes": round(elapsed / 60, 1),
                "log_path": log_path,
                "error": f"Nextflow exited with code {process.returncode}",
                "error_context": error_context,
                "command": " ".join(cmd),
            }

        # Step 4: Locate output files
        outputs = find_pipeline_outputs(output_dir, aligner)

        return {
            "status": "success",
            "elapsed_minutes": round(elapsed / 60, 1),
            "samplesheet": sheet_result,
            "aligner": aligner,
            "genome": genome,
            "output_dir": output_dir,
            "outputs": outputs,
            "log_path": log_path,
            "report_path": os.path.join(output_dir, "nextflow_report.html"),
            "command": " ".join(cmd),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "step": "nextflow",
            "error": "Pipeline timed out after 4 hours",
            "log_path": log_path,
            "command": " ".join(cmd),
        }
    except FileNotFoundError:
        return {
            "status": "failed",
            "step": "nextflow",
            "error": (
                "Nextflow not found. Install via cluster init script: "
                f"{_cfg.volume('scrnaseq_data')}/"
                "init_scripts/install_nextflow.sh"
            ),
        }
    except Exception as e:
        return {
            "status": "failed",
            "step": "nextflow",
            "error": str(e),
            "command": " ".join(cmd),
        }


def find_pipeline_outputs(output_dir: str, aligner: str = "starsolo") -> Dict:
    """
    Locate key output files from a completed nf-core/scrnaseq run.

    Returns paths to count matrices, MultiQC report, and any h5ad files.
    """
    outputs = {
        "count_matrices": [],
        "h5ad_files": [],
        "multiqc_report": None,
        "raw_matrices": [],
    }

    # Search for count matrices (filtered/raw)
    for pattern in [
        "**/filtered_feature_bc_matrix/**",
        "**/raw_feature_bc_matrix/**",
        "**/*_matrix.mtx*",
        "**/*.h5",
        "**/*.h5ad",
    ]:
        for f in glob.glob(os.path.join(output_dir, pattern), recursive=True):
            if "filtered" in f:
                outputs["count_matrices"].append(f)
            elif "raw" in f:
                outputs["raw_matrices"].append(f)
            elif f.endswith(".h5ad"):
                outputs["h5ad_files"].append(f)

    # STARsolo-specific outputs
    if aligner == "starsolo":
        for f in glob.glob(os.path.join(output_dir, "**/Solo.out/**"), recursive=True):
            if "filtered" in f.lower() and f.endswith((".mtx", ".tsv", ".gz")):
                outputs["count_matrices"].append(f)

    # MultiQC report
    multiqc = glob.glob(os.path.join(output_dir, "**/multiqc_report.html"), recursive=True)
    if multiqc:
        outputs["multiqc_report"] = multiqc[0]

    return outputs


def convert_mtx_to_h5ad(
    mtx_dir: str,
    output_path: str,
) -> Dict:
    """
    Convert a 10x-style count matrix directory (matrix.mtx, barcodes.tsv, features.tsv)
    to h5ad format for scanpy. Runs in-process using scanpy.

    Args:
        mtx_dir: Directory containing matrix.mtx[.gz], barcodes.tsv[.gz], features/genes.tsv[.gz]
        output_path: Output h5ad file path

    Returns:
        Dict with status, output_path, n_cells, n_genes
    """
    try:
        import scanpy as sc

        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
        adata.var_names_make_unique()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        adata.write_h5ad(output_path)

        return {
            "status": "success",
            "output_path": output_path,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
        }

    except ImportError:
        return {"status": "failed", "error": "scanpy not installed"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}



# ── Parallel fan-out: split multi-sample studies across parallel tasks ──

def split_samplesheet_by_sample(samplesheet_path: str, output_dir: str) -> Dict:
    """
    Split a multi-sample samplesheet CSV into per-sample samplesheets.

    Each sample gets its own CSV file in output_dir/per_sample/<sample_name>/samplesheet.csv
    so it can be processed independently by a parallel Nextflow task.

    Args:
        samplesheet_path: Path to the combined samplesheet CSV
        output_dir: Base output directory (per-sample sheets go in subdirectories)

    Returns:
        Dict with sample_sheets: {sample_name: samplesheet_path, ...} and sample_count
    """
    import csv as csv_mod

    with open(samplesheet_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        rows_by_sample = {}
        for row in reader:
            sample = row.get("sample", "unknown")
            rows_by_sample.setdefault(sample, []).append(row)

    if not rows_by_sample:
        return {"error": "No samples found in samplesheet", "sample_sheets": {}, "sample_count": 0}

    sample_sheets = {}
    for sample_name, rows in rows_by_sample.items():
        sample_dir = os.path.join(output_dir, "per_sample", sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        sheet_path = os.path.join(sample_dir, "samplesheet.csv")

        with open(sheet_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=["sample", "fastq_1", "fastq_2"])
            writer.writeheader()
            writer.writerows(rows)

        sample_sheets[sample_name] = sheet_path

    return {
        "sample_sheets": sample_sheets,
        "sample_count": len(sample_sheets),
    }


def build_parallel_tasks(
    samplesheet_path: str,
    output_dir: str,
    aligner: str = "starsolo",
    genome: str = "GRCh38",
    chemistry: str = "auto",
    pipeline_version: str = "2.7.1",
    extra_args: Optional[str] = None,
    compute_tier: str = "medium",
    merge_notebook_path: str = "",
) -> Dict:
    """
    Build parallel SubmitTask dicts for a multi-sample Nextflow run.

    Splits the samplesheet by sample, creates one Nextflow task per sample
    (each with its own cluster), and adds a final merge task that depends on
    all sample tasks completing.

    Args:
        samplesheet_path: Path to the combined samplesheet CSV
        output_dir: Base output directory in UC Volume
        aligner: Aligner to use
        genome: Reference genome
        chemistry: 10x chemistry version
        pipeline_version: nf-core/scrnaseq version
        extra_args: Additional Nextflow args (already includes resource limits)
        compute_tier: Compute tier for per-sample clusters
        merge_notebook_path: Path to the merge/collect notebook

    Returns:
        Dict with tasks (list of task dicts), sample_count, sample_names
    """
    # Split samplesheet
    split_result = split_samplesheet_by_sample(samplesheet_path, output_dir)
    if "error" in split_result:
        return split_result

    sample_sheets = split_result["sample_sheets"]
    if split_result["sample_count"] <= 1:
        # Single sample — no need for parallel fan-out
        return {"tasks": [], "sample_count": 1, "single_sample": True}

    alignment_notebook = (
        "/Users/andrew_forman@eisai.com/genesis-workbench"
        "/modules/single_cell/scanpy/scanpy_v0.0.1/notebooks/run_nextflow_scrnaseq"
    )

    cluster_spec = build_new_cluster_spec(tier=compute_tier)
    tasks = []
    task_keys = []

    # One task per sample
    for sample_name, sheet_path in sample_sheets.items():
        safe_key = sample_name.replace("-", "_").replace(".", "_")[:50]
        task_key = f"nf_{safe_key}"
        task_keys.append(task_key)

        sample_output = os.path.join(output_dir, "per_sample", sample_name, "output")

        tasks.append({
            "task_key": task_key,
            "notebook_task": {
                "notebook_path": alignment_notebook,
                "base_parameters": {
                    "fastq_dir": os.path.dirname(sheet_path),
                    "output_dir": sample_output,
                    "aligner": aligner,
                    "genome": genome,
                    "chemistry": chemistry,
                    "sample_name": sample_name,
                    "pipeline_version": pipeline_version,
                    "extra_args": extra_args or "",
                },
                "source": "WORKSPACE",
            },
            "new_cluster": cluster_spec,
        })

    # Merge/collect task — depends on all sample tasks
    if merge_notebook_path:
        merge_cluster = build_new_cluster_spec(tier="small")
        tasks.append({
            "task_key": "merge_outputs",
            "depends_on": [{"task_key": k} for k in task_keys],
            "notebook_task": {
                "notebook_path": merge_notebook_path,
                "base_parameters": {
                    "output_dir": output_dir,
                    "sample_names": ",".join(sample_sheets.keys()),
                },
                "source": "WORKSPACE",
            },
            "new_cluster": merge_cluster,
        })

    return {
        "tasks": tasks,
        "sample_count": len(sample_sheets),
        "sample_names": list(sample_sheets.keys()),
        "task_keys": task_keys,
    }
