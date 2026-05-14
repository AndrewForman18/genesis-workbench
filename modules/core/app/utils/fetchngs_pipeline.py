"""
fetchngs_pipeline.py — Utilities for nf-core/fetchngs integration in Genesis Workbench.

Provides:
1. ENA Programmatic API metadata fetching (structured JSON, replaces BeautifulSoup scraping)
2. Accession ID validation and classification
3. fetchngs samplesheet parsing
4. Job submission helper for Streamlit integration

Supported accession types:
- SRR/ERR/DRR (individual runs)
- SRP/ERP/DRP (studies)
- PRJNA/PRJEB/PRJDB (BioProjects)
- GSE (GEO Series — converted to SRP via ENA)
- syn (Synapse IDs)
"""
import os
import re
import time
import json
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone


# ── ENA Programmatic API ──
ENA_PORTAL_API = "https://www.ebi.ac.uk/ena/portal/api"
ENA_BROWSER_API = "https://www.ebi.ac.uk/ena/browser/api"

# Accession patterns
ACCESSION_PATTERNS = {
    "run": re.compile(r"^[SED]RR\d{6,}$"),
    "experiment": re.compile(r"^[SED]RX\d{6,}$"),
    "study": re.compile(r"^[SED]RP\d{6,}$"),
    "bioproject": re.compile(r"^PRJ[A-Z]{1,2}\d+$"),
    "geo": re.compile(r"^GSE\d+$"),
    "synapse": re.compile(r"^syn\d+$"),
}

# nf-core/fetchngs supported downstream pipelines
SUPPORTED_DOWNSTREAM_PIPELINES = [
    "scrnaseq",
    "rnaseq",
    "taxprofiler",
    "atacseq",
    "none",
]

DOWNLOAD_METHODS = ["sratools", "aspera", "ftp"]


def classify_accession(accession: str) -> Tuple[str, str]:
    """
    Classify an accession ID by type and source repository.
    
    Returns:
        Tuple of (accession_type, repository)
        e.g., ('run', 'SRA'), ('study', 'ENA'), ('geo', 'GEO')
    """
    cleaned = accession.strip()
    accession = cleaned.upper()
    
    for acc_type, pattern in ACCESSION_PATTERNS.items():
        # Synapse IDs are case-sensitive (lowercase 'syn')
        test_value = cleaned if acc_type == "synapse" else accession
        if pattern.match(test_value):
            if acc_type in ("geo", "synapse"):
                return acc_type, acc_type.upper()
            # Determine repository from prefix
            if accession.startswith("S"):
                repo = "SRA"
            elif accession.startswith("E"):
                repo = "ENA"
            elif accession.startswith("D"):
                repo = "DDBJ"
            elif accession.startswith("PRJ"):
                repo = "BioProject"
            else:
                repo = "Unknown"
            return acc_type, repo
    
    return "unknown", "Unknown"


def validate_accessions(accession_list: List[str]) -> Dict:
    """
    Validate and classify a list of accession IDs.
    
    Args:
        accession_list: List of accession strings
    
    Returns:
        Dict with 'valid', 'invalid', 'summary' keys
    """
    valid = []
    invalid = []
    type_counts = {}
    
    for acc in accession_list:
        acc = acc.strip()
        if not acc:
            continue
        acc_type, repo = classify_accession(acc)
        if acc_type == "unknown":
            invalid.append({"accession": acc, "reason": "Unrecognized format"})
        else:
            valid.append({"accession": acc.upper(), "type": acc_type, "repository": repo})
            type_counts[acc_type] = type_counts.get(acc_type, 0) + 1
    
    return {
        "valid": valid,
        "invalid": invalid,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "type_counts": type_counts,
    }


def fetch_ena_metadata(accession: str, limit: int = 500) -> Dict:
    """
    Fetch metadata for an accession using the ENA Programmatic Portal API.
    Returns structured JSON — far more reliable than HTML scraping.
    
    Supports: SRP/ERP/DRP, PRJNA/PRJEB/PRJDB, SRR/ERR/DRR, GSE (converted)
    
    Args:
        accession: Any supported accession ID
        limit: Max number of run records to return
    
    Returns:
        Dict with study metadata and list of runs with FASTQ URLs
    """
    accession = accession.strip().upper()
    acc_type, repo = classify_accession(accession)
    
    if acc_type == "unknown":
        return {"error": f"Unrecognized accession format: {accession}"}
    
    if acc_type == "synapse":
        return {
            "accession": accession,
            "type": "synapse",
            "note": "Synapse IDs are handled directly by fetchngs (requires SYNAPSE_AUTH_TOKEN env var).",
            "runs": [],
        }
    
    # For GEO accessions, try to find the linked SRA study
    if acc_type == "geo":
        return _fetch_geo_via_ena(accession, limit)
    
    # Build the ENA Portal API query
    try:
        # Determine the query field based on accession type
        if acc_type == "run":
            query = f"run_accession={accession}"
            result_type = "read_run"
        elif acc_type == "experiment":
            query = f"experiment_accession={accession}"
            result_type = "read_run"
        elif acc_type in ("study", "bioproject"):
            # Both study accessions and BioProject IDs can be queried as study_accession
            query = f"study_accession={accession} OR secondary_study_accession={accession}"
            result_type = "read_run"
        else:
            return {"error": f"Unsupported accession type for ENA query: {acc_type}"}
        
        # Query ENA Portal API
        params = {
            "result": result_type,
            "query": query,
            "fields": ",".join([
                "run_accession", "experiment_accession", "study_accession",
                "sample_accession", "secondary_sample_accession",
                "instrument_platform", "instrument_model",
                "library_layout", "library_strategy", "library_source",
                "library_name", "read_count", "base_count",
                "fastq_ftp", "fastq_aspera", "fastq_md5", "fastq_bytes",
                "study_title", "experiment_title", "sample_title",
                "scientific_name", "tax_id",
            ]),
            "format": "json",
            "limit": limit,
        }
        
        resp = requests.get(f"{ENA_PORTAL_API}/search", params=params, timeout=30)
        
        if resp.status_code == 204:
            # No content — try alternative query
            return {"error": f"No records found in ENA for: {accession}"}
        
        resp.raise_for_status()
        records = resp.json()
        
        if not records:
            return {"error": f"No records found in ENA for: {accession}"}
        
        # Extract study-level metadata from first record
        first = records[0]
        study_meta = {
            "accession": accession,
            "type": acc_type,
            "repository": repo,
            "study_accession": first.get("study_accession", ""),
            "title": first.get("study_title", "") or first.get("experiment_title", ""),
            "organism": first.get("scientific_name", ""),
            "tax_id": first.get("tax_id", ""),
            "platform": first.get("instrument_platform", ""),
            "instrument": first.get("instrument_model", ""),
            "library_strategy": first.get("library_strategy", ""),
            "library_layout": first.get("library_layout", ""),
            "n_runs": len(records),
        }
        
        # Parse run-level metadata
        runs = []
        for rec in records:
            fastq_bytes = rec.get("fastq_bytes", "")
            size_mb = 0
            if fastq_bytes:
                try:
                    size_mb = round(sum(int(b) for b in fastq_bytes.split(";") if b) / (1024 * 1024), 1)
                except (ValueError, TypeError):
                    pass
            
            runs.append({
                "run_accession": rec.get("run_accession", ""),
                "experiment_accession": rec.get("experiment_accession", ""),
                "sample_accession": rec.get("sample_accession", ""),
                "sample_title": rec.get("sample_title", ""),
                "library_layout": rec.get("library_layout", ""),
                "library_strategy": rec.get("library_strategy", ""),
                "read_count": int(rec.get("read_count", 0) or 0),
                "base_count": int(rec.get("base_count", 0) or 0),
                "size_mb": size_mb,
                "fastq_ftp": rec.get("fastq_ftp", ""),
                "fastq_md5": rec.get("fastq_md5", ""),
            })
        
        study_meta["runs"] = runs
        study_meta["total_size_gb"] = round(sum(r["size_mb"] for r in runs) / 1024, 2)
        
        return study_meta
    
    except requests.exceptions.HTTPError as e:
        return {"error": f"ENA API HTTP error for {accession}: {e}"}
    except requests.exceptions.Timeout:
        return {"error": f"ENA API timeout for {accession}. Try again later."}
    except Exception as e:
        return {"error": f"ENA API error for {accession}: {str(e)}"}


def _fetch_geo_via_ena(gse_id: str, limit: int = 500) -> Dict:
    """
    Fetch GEO series metadata via ENA by searching for the linked SRA study.
    More reliable than scraping NCBI HTML.
    """
    try:
        # ENA stores GEO links as secondary_study_accession or via text search
        params = {
            "result": "study",
            "query": f'secondary_study_accession="{gse_id}" OR study_alias="{gse_id}"',
            "fields": "study_accession,secondary_study_accession,study_title,scientific_name,tax_id",
            "format": "json",
            "limit": 5,
        }
        resp = requests.get(f"{ENA_PORTAL_API}/search", params=params, timeout=15)
        
        if resp.status_code == 204 or not resp.text.strip():
            # Fallback: try NCBI E-utilities to find SRP
            return _geo_to_srp_ncbi(gse_id, limit)
        
        resp.raise_for_status()
        studies = resp.json()
        
        if not studies:
            return _geo_to_srp_ncbi(gse_id, limit)
        
        # Found the linked study — now fetch run details
        srp = studies[0].get("study_accession", "")
        if srp:
            result = fetch_ena_metadata(srp, limit)
            if "error" not in result:
                result["original_accession"] = gse_id
                result["note"] = f"GEO {gse_id} resolved to ENA study {srp}"
            return result
        
        return {"error": f"Could not resolve GEO {gse_id} to an SRA/ENA study accession."}
    
    except Exception as e:
        return {"error": f"Failed to resolve GEO {gse_id} via ENA: {str(e)}"}


def _geo_to_srp_ncbi(gse_id: str, limit: int = 500) -> Dict:
    """Fallback: use NCBI E-utilities to find SRP for a GSE accession."""
    try:
        from xml.etree import ElementTree
        
        # Search SRA for the GEO accession
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "sra", "term": gse_id, "retmode": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        
        if not ids:
            return {"error": f"No SRA records found for GEO accession {gse_id}"}
        
        # Get summary to find the SRP
        time.sleep(0.35)
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "sra", "id": ids[0], "retmode": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        entry = result.get(ids[0], {})
        
        exp_xml = entry.get("expxml", "")
        if exp_xml:
            wrapped = f"<root>{exp_xml}</root>"
            root = ElementTree.fromstring(wrapped)
            study_el = root.find(".//Study")
            if study_el is not None:
                srp = study_el.get("acc", "")
                if srp:
                    ena_result = fetch_ena_metadata(srp, limit)
                    if "error" not in ena_result:
                        ena_result["original_accession"] = gse_id
                        ena_result["note"] = f"GEO {gse_id} resolved to SRA study {srp} (via NCBI)"
                    return ena_result
        
        return {"error": f"Could not extract SRP from GEO {gse_id} NCBI metadata"}
    
    except Exception as e:
        return {"error": f"NCBI fallback failed for {gse_id}: {str(e)}"}


def parse_fetchngs_samplesheet(samplesheet_path: str) -> Dict:
    """
    Parse the auto-generated samplesheet from nf-core/fetchngs output.
    
    The samplesheet format depends on the --nf_core_pipeline parameter:
    - scrnaseq: sample, fastq_1, fastq_2
    - rnaseq: sample, fastq_1, fastq_2, strandedness
    - atacseq: sample, fastq_1, fastq_2, replicate
    - none: full metadata columns
    
    Args:
        samplesheet_path: Path to the fetchngs output samplesheet.csv
    
    Returns:
        Dict with parsed samples and samplesheet metadata
    """
    import csv
    
    if not os.path.exists(samplesheet_path):
        return {"error": f"Samplesheet not found: {samplesheet_path}"}
    
    samples = []
    with open(samplesheet_path, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for row in reader:
            sample = dict(row)
            # Compute FASTQ sizes if files exist
            for fq_col in ["fastq_1", "fastq_2"]:
                if fq_col in sample and sample[fq_col] and os.path.exists(sample[fq_col]):
                    sample[f"{fq_col}_size_mb"] = round(
                        os.path.getsize(sample[fq_col]) / (1024 * 1024), 2
                    )
            samples.append(sample)
    
    return {
        "samplesheet_path": samplesheet_path,
        "columns": columns,
        "n_samples": len(samples),
        "samples": samples,
    }


def parse_fetchngs_metadata(metadata_dir: str) -> Dict:
    """
    Parse the raw SRA/ENA metadata files from fetchngs output.
    Typically found at: {outdir}/custom/ or {outdir}/metadata/
    
    Returns:
        Dict with parsed metadata records
    """
    import csv
    
    metadata = {"files_found": [], "records": []}
    
    # Look for common metadata file patterns
    patterns = [
        os.path.join(metadata_dir, "custom", "*.tsv"),
        os.path.join(metadata_dir, "metadata", "*.tsv"),
        os.path.join(metadata_dir, "samplesheet", "*.csv"),
    ]
    
    import glob
    for pattern in patterns:
        for f in glob.glob(pattern):
            metadata["files_found"].append(f)
            try:
                delimiter = "\t" if f.endswith(".tsv") else ","
                with open(f, "r") as fh:
                    reader = csv.DictReader(fh, delimiter=delimiter)
                    for row in reader:
                        metadata["records"].append(dict(row))
            except Exception as e:
                metadata.setdefault("errors", []).append(f"Error reading {f}: {e}")
    
    metadata["n_records"] = len(metadata["records"])
    return metadata


def start_fetchngs_job(
    accession_ids: List[str],
    output_dir: str,
    nf_core_pipeline: str = "scrnaseq",
    download_method: str = "sratools",
    pipeline_version: str = "1.12.0",
    genome: str = "GRCh38",
    trigger_downstream: bool = True,
    project_name: str = "",
    extra_args: str = "",
    catalog: str = "dhbl_discovery_us_dev",
    schema: str = "genesis_schema",
    user_info: Optional[Dict] = None,
) -> Tuple[int, int]:
    """
    Submit the fetchngs sandwich wrapper notebook as a Databricks job.
    Called from the Streamlit form.
    
    Args:
        accession_ids: List of accession IDs to fetch
        output_dir: UC Volume output path
        nf_core_pipeline: Target downstream pipeline
        download_method: aspera/ftp/sratools
        pipeline_version: nf-core/fetchngs version
        genome: Reference genome (for downstream trigger)
        trigger_downstream: Auto-trigger downstream pipeline on success
        project_name: Project name for audit trail
        extra_args: Extra nextflow arguments
        catalog: Unity Catalog name
        schema: Schema name
        user_info: User info dict from Streamlit
    
    Returns:
        Tuple of (job_id, run_id)
    """
    from databricks.sdk import WorkspaceClient
    
    w = WorkspaceClient()
    
    notebook_path = (
        "/Users/andrew_forman@eisai.com/genesis-workbench"
        "/modules/single_cell/scanpy/scanpy_v0.0.1/notebooks"
        "/nextflow_sandwich_wrapper_fetchngs"
    )
    
    # Join accession IDs with newline for the widget
    ids_str = "\n".join(accession_ids)
    
    run = w.jobs.submit(
        run_name=f"fetchngs: {project_name or 'unnamed'} ({len(accession_ids)} accessions)",
        tasks=[{
            "task_key": "fetchngs_run",
            "notebook_task": {
                "notebook_path": notebook_path,
                "base_parameters": {
                    "accession_ids": ids_str,
                    "output_dir": output_dir,
                    "nf_core_pipeline": nf_core_pipeline,
                    "download_method": download_method,
                    "pipeline_version": pipeline_version,
                    "genome": genome,
                    "trigger_downstream": str(trigger_downstream).lower(),
                    "project_name": project_name,
                    "extra_args": extra_args,
                    "catalog": catalog,
                    "schema": schema,
                    "qc_gate_enabled": "true",
                    "min_fastq_size_mb": "10",
                },
                "source": "WORKSPACE",
            },
            "new_cluster": {
                "spark_version": "16.4.x-scala2.12",
                "node_type_id": "r5.2xlarge",
                "num_workers": 0,
                "spark_conf": {
                    "spark.master": "local[*]",
                    "spark.databricks.cluster.profile": "singleNode",
                },
                "data_security_mode": "SINGLE_USER",
            },
        }],
    )
    
    # The submit API returns a run_id; job_id is accessible from the run
    return (0, run.run_id)  # job_id=0 for one-time runs


def estimate_download_size(metadata: Dict) -> Dict:
    """
    Estimate total download size from ENA metadata.
    
    Args:
        metadata: Output from fetch_ena_metadata()
    
    Returns:
        Dict with size estimates and time estimates
    """
    if "error" in metadata:
        return metadata
    
    runs = metadata.get("runs", [])
    total_size_mb = sum(r.get("size_mb", 0) for r in runs)
    total_size_gb = total_size_mb / 1024
    
    # Estimate download time (conservative: 50 MB/s for aspera, 10 MB/s for FTP/HTTPS)
    time_aspera_min = total_size_mb / (50 * 60)
    time_ftp_min = total_size_mb / (10 * 60)
    
    return {
        "n_runs": len(runs),
        "total_size_gb": round(total_size_gb, 2),
        "estimated_time_aspera_min": round(time_aspera_min, 1),
        "estimated_time_ftp_min": round(time_ftp_min, 1),
        "paired_end": sum(1 for r in runs if r.get("library_layout") == "PAIRED"),
        "single_end": sum(1 for r in runs if r.get("library_layout") == "SINGLE"),
    }
