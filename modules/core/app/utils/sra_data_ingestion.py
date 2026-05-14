"""
SRA Data Ingestion Utilities — companion to geo_data_ingestion.py
Provides functions to fetch SRA metadata from NCBI, download FASTQ files,
and stage them to UC Volumes for downstream analysis.

Supports accession types: SRP (study), SRR (run), SRX (experiment), PRJNA (BioProject)
Download methods: AWS S3 mirror (fastest on AWS), HTTPS fallback
"""
import os
import re
import time
import subprocess
import requests
from typing import List, Dict
from xml.etree import ElementTree


# ── NCBI E-utilities base URLs ──
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# AWS S3 mirror for SRA data (public, no auth required)
SRA_S3_BUCKET = "https://sra-pub-run-odp.s3.amazonaws.com"


def _ncbi_get(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """Make a rate-limited GET to NCBI E-utilities (max 3 req/sec without API key)."""
    time.sleep(0.35)
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def fetch_sra_metadata(accession: str) -> Dict:
    """
    Fetch metadata for an SRA accession (SRP, SRR, SRX, or PRJNA).
    Uses NCBI E-utilities and the SRA Run Selector API.

    Args:
        accession: SRA accession ID (e.g., 'SRP123456', 'SRR1234567', 'PRJNA123456')

    Returns:
        Dict with keys: accession, accession_type, title, organism, platform,
        library_strategy, n_runs, runs (list of dicts with srr_id, bases, spots, etc.)
    """
    accession = accession.strip().upper()

    # Determine accession type
    if re.match(r"SRP\d+", accession):
        acc_type = "study"
        db_term = accession
    elif re.match(r"SRR\d+", accession):
        acc_type = "run"
        db_term = accession
    elif re.match(r"SRX\d+", accession):
        acc_type = "experiment"
        db_term = accession
    elif re.match(r"PRJNA\d+", accession):
        acc_type = "bioproject"
        db_term = accession
    elif re.match(r"GSE\d+", accession):
        return {"error": f"'{accession}' is a GEO accession. Use the GEO tab instead."}
    else:
        return {"error": f"Unrecognized accession format: {accession}. Expected SRP, SRR, SRX, or PRJNA."}

    try:
        # Step 1: Search SRA database for the accession
        search_resp = _ncbi_get(ESEARCH_URL, {
            "db": "sra",
            "term": db_term,
            "retmax": 500,
            "retmode": "json",
            "usehistory": "y"
        })
        search_data = search_resp.json()
        result = search_data.get("esearchresult", {})
        id_list = result.get("idlist", [])

        if not id_list:
            return {"error": f"No SRA records found for accession: {accession}"}

        # Step 2: Fetch summary for the IDs
        summary_resp = _ncbi_get(ESUMMARY_URL, {
            "db": "sra",
            "id": ",".join(id_list[:50]),
            "retmode": "json"
        })
        summary_data = summary_resp.json().get("result", {})

        # Step 3: Parse run information
        runs = []
        study_title = ""
        organism = ""
        platform = ""
        library_strategy = ""

        for uid in id_list[:50]:
            entry = summary_data.get(uid, {})
            if not entry:
                continue

            # Parse the XML in ExpXml field
            exp_xml = entry.get("expxml", "")
            if exp_xml:
                try:
                    wrapped = f"<root>{exp_xml}</root>"
                    root = ElementTree.fromstring(wrapped)

                    title_el = root.find(".//Title")
                    if title_el is not None and title_el.text:
                        study_title = study_title or title_el.text

                    org_el = root.find(".//Organism")
                    if org_el is not None:
                        organism = organism or org_el.get("CommonName", org_el.get("ScientificName", ""))

                    platform_el = root.find(".//Platform")
                    if platform_el is not None:
                        platform = platform or platform_el.get("instrument_model", "")

                    lib_el = root.find(".//Library_descriptor/LIBRARY_STRATEGY")
                    if lib_el is not None and lib_el.text:
                        library_strategy = library_strategy or lib_el.text

                except ElementTree.ParseError:
                    pass

            # Parse runs from the Runs field
            runs_xml = entry.get("runs", "")
            if runs_xml:
                try:
                    wrapped = f"<root>{runs_xml}</root>"
                    root = ElementTree.fromstring(wrapped)
                    for run_el in root.findall(".//Run"):
                        srr_id = run_el.get("acc", "")
                        total_spots = run_el.get("total_spots", "0")
                        total_bases = run_el.get("total_bases", "0")
                        # total_size not always present in API response;
                        # estimate compressed SRA size as ~0.25 bytes/base
                        total_size = run_el.get("total_size", "")

                        if srr_id:
                            if total_size and total_size.isdigit() and int(total_size) > 0:
                                size_mb = round(int(total_size) / (1024 * 1024), 1)
                            elif total_bases.isdigit() and int(total_bases) > 0:
                                # Estimate: compressed SRA ≈ 0.25 bytes per base
                                size_mb = round(int(total_bases) * 0.25 / (1024 * 1024), 1)
                            else:
                                size_mb = 0

                            runs.append({
                                "srr_id": srr_id,
                                "spots": int(total_spots) if total_spots.isdigit() else 0,
                                "bases": int(total_bases) if total_bases.isdigit() else 0,
                                "size_mb": size_mb,
                            })
                except ElementTree.ParseError:
                    pass

        # Deduplicate runs by srr_id
        seen = set()
        unique_runs = []
        for r in runs:
            if r["srr_id"] not in seen:
                seen.add(r["srr_id"])
                unique_runs.append(r)

        return {
            "accession": accession,
            "accession_type": acc_type,
            "title": study_title or "N/A",
            "organism": organism or "N/A",
            "platform": platform or "N/A",
            "library_strategy": library_strategy or "N/A",
            "n_runs": len(unique_runs),
            "runs": unique_runs,
        }

    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error querying NCBI for {accession}: {e}"}
    except Exception as e:
        return {"error": f"Failed to fetch SRA metadata for {accession}: {str(e)}"}


def download_sra_to_volume(
    srr_ids: List[str],
    volume_path: str,
    project_name: str = "",
    method: str = "https"
) -> Dict:
    """
    Download SRA run data to a UC Volume directory.

    Supports two methods:
    - "https": Download .sra files from NCBI HTTPS (works everywhere, reliable)
    - "aws_s3": Download from s3://sra-pub-run-odp (faster on AWS, uses boto3)

    After download, runs fasterq-dump if sra-tools is available to convert to FASTQ.

    Args:
        srr_ids: List of SRR accession IDs to download
        volume_path: UC Volume base path
        project_name: Optional project subdirectory
        method: Download method ("https" or "aws_s3")

    Returns:
        Dict with download status per run and output paths
    """
    if project_name:
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", project_name)
        output_dir = os.path.join(volume_path, "sra_downloads", safe_name)
    else:
        output_dir = os.path.join(volume_path, "sra_downloads")

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for srr_id in srr_ids:
        srr_id = srr_id.strip()
        run_dir = os.path.join(output_dir, srr_id)
        os.makedirs(run_dir, exist_ok=True)

        try:
            if method == "aws_s3":
                # Download from AWS S3 public mirror using boto3
                sra_path = _download_sra_s3(srr_id, run_dir)
            else:
                # Download via HTTPS from NCBI
                sra_path = _download_sra_https(srr_id, run_dir)

            # Try to convert .sra to FASTQ if sra-tools is available
            fastq_files = _try_fasterq_dump(sra_path, run_dir)

            # List all files in run directory
            all_files = os.listdir(run_dir)
            total_size_mb = sum(
                os.path.getsize(os.path.join(run_dir, f)) / (1024 * 1024)
                for f in all_files if os.path.isfile(os.path.join(run_dir, f))
            )

            results.append({
                "srr_id": srr_id,
                "status": "success",
                "sra_file": os.path.basename(sra_path) if sra_path else None,
                "fastq_files": fastq_files,
                "all_files": all_files,
                "size_mb": round(total_size_mb, 2),
                "path": run_dir
            })

        except Exception as e:
            results.append({
                "srr_id": srr_id,
                "status": "failed",
                "error": str(e),
                "path": run_dir
            })

    return {
        "output_dir": output_dir,
        "method": method,
        "n_downloaded": sum(1 for r in results if r["status"] == "success"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "runs": results
    }


def _download_sra_https(srr_id: str, run_dir: str) -> str:
    """Download .sra file from NCBI via HTTPS."""
    # NCBI SRA HTTPS URL pattern
    prefix = srr_id[:6]
    if len(srr_id) > 9:
        sub = f"0{srr_id[-2:]}" if len(srr_id) == 10 else f"00{srr_id[-1]}" if len(srr_id) == 11 else ""
        url = f"https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos5/sra-pub-zq-11/{prefix}/{sub}/{srr_id}/{srr_id}.sralite"
    else:
        url = f"https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos5/sra-pub-zq-11/{prefix}/{srr_id}/{srr_id}.sralite"

    # Try the AWS S3 public URL as primary HTTPS source (more reliable)
    s3_url = f"{SRA_S3_BUCKET}/sra/{srr_id}/{srr_id}"
    out_path = os.path.join(run_dir, f"{srr_id}.sra")

    for attempt_url in [s3_url, url]:
        try:
            resp = requests.get(attempt_url, stream=True, timeout=60)
            if resp.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                return out_path
        except Exception:
            continue

    raise RuntimeError(f"Failed to download {srr_id} from NCBI. Check the accession ID.")


def _download_sra_s3(srr_id: str, run_dir: str) -> str:
    """Download .sra file from AWS S3 public mirror using boto3."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        key = f"sra/{srr_id}/{srr_id}"
        out_path = os.path.join(run_dir, f"{srr_id}.sra")

        s3.download_file("sra-pub-run-odp", key, out_path)
        return out_path

    except ImportError:
        # Fall back to HTTPS if boto3 not available
        return _download_sra_https(srr_id, run_dir)
    except Exception as e:
        # Fall back to HTTPS on any S3 error
        return _download_sra_https(srr_id, run_dir)


def _try_fasterq_dump(sra_path: str, run_dir: str) -> List[str]:
    """Attempt to convert .sra to FASTQ using fasterq-dump. Returns list of FASTQ files created."""
    if not sra_path or not os.path.exists(sra_path):
        return []

    try:
        result = subprocess.run(
            ["fasterq-dump", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    try:
        subprocess.run(
            ["fasterq-dump", sra_path,
             "--outdir", run_dir,
             "--split-3",
             "--threads", "4",
             "--force"],
            capture_output=True, text=True, timeout=3600
        )

        fastq_files = [f for f in os.listdir(run_dir) if f.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz"))]
        return fastq_files

    except Exception:
        return []


def detect_sra_data(directory: str) -> Dict:
    """
    Detect SRA/FASTQ data formats in a directory.

    Args:
        directory: Path to scan

    Returns:
        Dict with format type, file paths, and recommendations
    """
    import glob

    if not os.path.isdir(directory):
        return {"error": f"Directory not found: {directory}", "n_formats_found": 0}

    formats_found = []

    # Check for FASTQ files
    for pattern in ["**/*.fastq", "**/*.fastq.gz", "**/*.fq", "**/*.fq.gz"]:
        for f in glob.glob(os.path.join(directory, pattern), recursive=True):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            is_paired = "_1.fastq" in f or "_2.fastq" in f or "_R1" in f or "_R2" in f
            formats_found.append({
                "format": "FASTQ (paired-end)" if is_paired else "FASTQ",
                "path": f,
                "size_mb": round(size_mb, 2),
                "recommendation": (
                    "Ready for alignment. Use in Disease Biology \u2192 Parabricks pipeline "
                    "or Cell Ranger for scRNA-seq."
                )
            })

    # Check for .sra files (not yet converted)
    for f in glob.glob(os.path.join(directory, "**/*.sra"), recursive=True):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        formats_found.append({
            "format": "SRA (unconverted)",
            "path": f,
            "size_mb": round(size_mb, 2),
            "recommendation": (
                "Raw SRA format. Install sra-tools on your cluster and run "
                "fasterq-dump to convert to FASTQ."
            )
        })

    # Check for BAM/CRAM files (sometimes in SRA)
    for pattern in ["**/*.bam", "**/*.cram"]:
        for f in glob.glob(os.path.join(directory, pattern), recursive=True):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            formats_found.append({
                "format": "BAM" if f.endswith(".bam") else "CRAM",
                "path": f,
                "size_mb": round(size_mb, 2),
                "recommendation": "Aligned reads. Can be used directly for variant calling or converted to FASTQ."
            })

    return {
        "n_formats_found": len(formats_found),
        "formats": formats_found
    }
