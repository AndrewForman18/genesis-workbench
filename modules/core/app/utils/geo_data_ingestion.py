"""
GEO Data Ingestion Utilities — integrated from CellAtria/CellExpress pipeline
Provides functions to fetch GEO metadata, download datasets, and stage to UC Volumes.
"""
import os
import re
import glob
import shutil
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple


def fetch_geo_metadata(gse_id: str) -> Dict:
    """
    Fetch metadata for a GEO Series accession (GSE ID).
    Scrapes NCBI GEO for title, organism, summary, and sample information.
    
    Args:
        gse_id: GEO Series accession (e.g., 'GSE204716')
    
    Returns:
        Dict with keys: title, organism, summary, link, samples (list of dicts)
    """
    gse_id = gse_id.strip().upper()
    if not re.match(r"GSE\d+", gse_id):
        return {"error": f"Invalid GEO accession: {gse_id}. Expected format: GSEnnnnnn"}
    
    geo_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
    
    try:
        response = requests.get(geo_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract basic metadata
        def get_field(label):
            cell = soup.find("td", string=label)
            if cell:
                return cell.find_next_sibling("td").get_text(strip=True)
            return "N/A"
        
        title = get_field("Title")
        organism = get_field("Organism")
        summary = get_field("Summary")
        
        # Extract overall design
        design = get_field("Overall design")
        
        # Extract sample rows (GSM accessions)
        sample_rows = []
        for row in soup.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 2:
                gsm_id = cols[0].get_text(strip=True)
                gsm_desc = cols[1].get_text(strip=True)
                if re.match(r"GSM\d{6,}", gsm_id):
                    sample_rows.append({
                        "gsm_id": gsm_id,
                        "description": gsm_desc
                    })
        
        # Extract supplementary file links
        supp_files = []
        for link in soup.find_all("a"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if "suppl/" in href or "filelist" in href.lower():
                supp_files.append({"name": text, "url": href})
        
        return {
            "gse_id": gse_id,
            "title": title,
            "organism": organism,
            "summary": summary,
            "design": design,
            "link": geo_url,
            "n_samples": len(sample_rows),
            "samples": sample_rows,
            "supplementary_files": supp_files[:10],  # Limit to 10
        }
    
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error fetching {gse_id}: {e}"}
    except Exception as e:
        return {"error": f"Failed to fetch metadata for {gse_id}: {str(e)}"}


def download_geo_to_volume(
    gse_id: str,
    gsm_ids: List[str],
    volume_path: str,
    project_name: str = ""
) -> Dict:
    """
    Download GEO supplementary files for selected samples to a UC Volume directory.
    Uses GEOparse for robust downloading and organizes files per sample.
    
    Args:
        gse_id: GEO Series accession
        gsm_ids: List of GSM sample IDs to download
        volume_path: UC Volume base path (e.g., /Volumes/catalog/schema/vol_name)
        project_name: Optional project name for subdirectory
    
    Returns:
        Dict with download status per sample and output paths
    """
    try:
        import GEOparse
    except ImportError:
        return {"error": "GEOparse not installed. Add 'GEOparse' to requirements.txt."}
    
    # Create output directory
    if project_name:
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", project_name)
        output_dir = os.path.join(volume_path, "geo_downloads", safe_name, gse_id)
    else:
        output_dir = os.path.join(volume_path, "geo_downloads", gse_id)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    original_dir = os.getcwd()
    
    for gsm_id in gsm_ids:
        sample_dir = os.path.join(output_dir, gsm_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        try:
            os.chdir(sample_dir)
            gsm = GEOparse.get_GEO(geo=gsm_id, destdir=sample_dir, how="full")
            gsm.download_supplementary_files()
            
            # Flatten subdirectories
            for root, dirs, files in os.walk(sample_dir):
                for f in files:
                    src = os.path.join(root, f)
                    dst = os.path.join(sample_dir, f)
                    if src != dst:
                        shutil.move(src, dst)
            
            # Remove empty dirs and descriptor txt
            for root, dirs, _ in os.walk(sample_dir, topdown=False):
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
            
            txt_file = os.path.join(sample_dir, f"{gsm_id}.txt")
            if os.path.exists(txt_file):
                os.remove(txt_file)
            
            # List downloaded files
            downloaded_files = os.listdir(sample_dir)
            total_size_mb = sum(
                os.path.getsize(os.path.join(sample_dir, f)) / (1024 * 1024)
                for f in downloaded_files if os.path.isfile(os.path.join(sample_dir, f))
            )
            
            results.append({
                "gsm_id": gsm_id,
                "status": "success",
                "files": downloaded_files,
                "size_mb": round(total_size_mb, 2),
                "path": sample_dir
            })
            
        except Exception as e:
            results.append({
                "gsm_id": gsm_id,
                "status": "failed",
                "error": str(e),
                "path": sample_dir
            })
        finally:
            os.chdir(original_dir)
    
    return {
        "gse_id": gse_id,
        "output_dir": output_dir,
        "n_downloaded": sum(1 for r in results if r["status"] == "success"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "samples": results
    }


def detect_data_format(directory: str) -> Dict:
    """
    Detect single-cell data format in a directory.
    Identifies: 10X Genomics (matrix.mtx.gz), h5ad, h5, CSV/TSV.
    
    Args:
        directory: Path to scan for data files
    
    Returns:
        Dict with format type, file paths, and recommendations
    """
    if not os.path.isdir(directory):
        return {"error": f"Directory not found: {directory}"}
    
    formats_found = []
    
    # Check for 10X Genomics format (matrix.mtx.gz + features/genes + barcodes)
    mtx_files = glob.glob(os.path.join(directory, "**", "matrix.mtx.gz"), recursive=True)
    for mtx in mtx_files:
        parent = os.path.dirname(mtx)
        features = glob.glob(os.path.join(parent, "*features*.tsv.gz")) or glob.glob(os.path.join(parent, "*genes*.tsv.gz"))
        barcodes = glob.glob(os.path.join(parent, "*barcodes*.tsv.gz"))
        if features and barcodes:
            formats_found.append({
                "format": "10X Genomics",
                "path": parent,
                "files": {"matrix": mtx, "features": features[0], "barcodes": barcodes[0]},
                "recommendation": "Use sc.read_10x_mtx() or convert to h5ad with scanpy"
            })
    
    # Check for h5ad files
    h5ad_files = glob.glob(os.path.join(directory, "**", "*.h5ad"), recursive=True)
    for f in h5ad_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        formats_found.append({
            "format": "h5ad (AnnData)",
            "path": f,
            "size_mb": round(size_mb, 2),
            "recommendation": "Ready for scanpy — use this path in Run Analysis"
        })
    
    # Check for h5 files (10X HDF5)
    h5_files = glob.glob(os.path.join(directory, "**", "*.h5"), recursive=True)
    for f in h5_files:
        if not f.endswith(".h5ad"):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            formats_found.append({
                "format": "HDF5 (10X)",
                "path": f,
                "size_mb": round(size_mb, 2),
                "recommendation": "Use sc.read_10x_h5() to load, then save as h5ad"
            })
    
    # Check for CSV/TSV (count matrices)
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv.gz"), recursive=True)
    csv_files += glob.glob(os.path.join(directory, "**", "*.tsv.gz"), recursive=True)
    csv_files += glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    for f in csv_files[:5]:  # Limit
        size_mb = os.path.getsize(f) / (1024 * 1024)
        formats_found.append({
            "format": "CSV/TSV",
            "path": f,
            "size_mb": round(size_mb, 2),
            "recommendation": "Load with pd.read_csv() and convert to AnnData"
        })
    
    return {
        "directory": directory,
        "n_formats_found": len(formats_found),
        "formats": formats_found
    }


def create_metadata_csv(samples: List[Dict], output_path: str) -> str:
    """
    Create a CellExpress-compatible metadata.csv from GEO sample info.
    
    Args:
        samples: List of sample dicts with gsm_id, description
        output_path: Directory to write metadata.csv
    
    Returns:
        Path to the created metadata.csv
    """
    import csv
    
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "metadata.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "description", "gsm_id"])
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "sample": s["gsm_id"],
                "description": s.get("description", ""),
                "gsm_id": s["gsm_id"]
            })
    
    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
# ENA API Integration (added for nf-core/fetchngs compatibility)
# ══════════════════════════════════════════════════════════════════════════════

ENA_PORTAL_API = "https://www.ebi.ac.uk/ena/portal/api"


def fetch_geo_metadata_ena(gse_id: str) -> Dict:
    """
    Fetch GEO metadata using the ENA Programmatic API (structured JSON).
    More reliable than HTML scraping — returns machine-readable data.
    
    This resolves a GSE accession to its linked SRA study in ENA,
    then returns structured run-level metadata including FASTQ URLs.
    
    Args:
        gse_id: GEO Series accession (e.g., 'GSE204716')
    
    Returns:
        Dict with structured metadata including runs with FASTQ download URLs.
        Falls back to None if ENA has no record (caller should use BeautifulSoup fallback).
    """
    gse_id = gse_id.strip().upper()
    if not re.match(r"GSE\d+", gse_id):
        return {"error": f"Invalid GEO accession: {gse_id}. Expected format: GSEnnnnnn"}
    
    try:
        # Step 1: Find linked SRA study in ENA
        params = {
            "result": "study",
            "query": f"secondary_study_accession={gse_id} OR study_alias={gse_id}",
            "fields": "study_accession,secondary_study_accession,study_title,scientific_name,tax_id",
            "format": "json",
            "limit": 5,
        }
        resp = requests.get(f"{ENA_PORTAL_API}/search", params=params, timeout=15)
        
        if resp.status_code == 204 or not resp.text.strip():
            return None  # No ENA record — caller should fall back
        
        resp.raise_for_status()
        studies = resp.json()
        
        if not studies:
            return None  # Fall back to BeautifulSoup
        
        srp = studies[0].get("study_accession", "")
        study_title = studies[0].get("study_title", "")
        organism = studies[0].get("scientific_name", "")
        
        if not srp:
            return None
        
        # Step 2: Get run-level details
        run_params = {
            "result": "read_run",
            "query": f"study_accession={srp}",
            "fields": ",".join([
                "run_accession", "experiment_accession", "sample_accession",
                "instrument_platform", "instrument_model",
                "library_layout", "library_strategy", "library_source",
                "sample_title", "read_count", "base_count",
                "fastq_ftp", "fastq_bytes", "fastq_md5",
            ]),
            "format": "json",
            "limit": 500,
        }
        run_resp = requests.get(f"{ENA_PORTAL_API}/search", params=run_params, timeout=30)
        
        if run_resp.status_code == 204:
            records = []
        else:
            run_resp.raise_for_status()
            records = run_resp.json()
        
        # Build sample list in the same format as fetch_geo_metadata
        samples = []
        for rec in records:
            samples.append({
                "gsm_id": rec.get("sample_accession", rec.get("run_accession", "")),
                "description": rec.get("sample_title", ""),
                "run_accession": rec.get("run_accession", ""),
                "library_layout": rec.get("library_layout", ""),
                "library_strategy": rec.get("library_strategy", ""),
                "read_count": int(rec.get("read_count", 0) or 0),
                "fastq_ftp": rec.get("fastq_ftp", ""),
            })
        
        return {
            "gse_id": gse_id,
            "sra_study": srp,
            "title": study_title,
            "organism": organism,
            "summary": f"ENA study {srp} linked to GEO {gse_id}",
            "design": "",
            "link": f"https://www.ebi.ac.uk/ena/browser/view/{srp}",
            "n_samples": len(samples),
            "samples": samples,
            "supplementary_files": [],
            "source": "ENA_API",  # Flag indicating this came from ENA
        }
    
    except requests.exceptions.Timeout:
        return None  # Fall back on timeout
    except Exception:
        return None  # Fall back on any error


def fetch_geo_metadata_robust(gse_id: str) -> Dict:
    """
    Robust GEO metadata fetch — tries ENA API first (structured JSON),
    falls back to NCBI HTML scraping (BeautifulSoup) if ENA has no record.
    
    This is the recommended entry point for the Streamlit form.
    
    Args:
        gse_id: GEO Series accession (e.g., 'GSE204716')
    
    Returns:
        Dict with metadata (same format as fetch_geo_metadata)
    """
    # Try ENA first (faster, structured, more reliable)
    ena_result = fetch_geo_metadata_ena(gse_id)
    
    if ena_result is not None and "error" not in ena_result:
        return ena_result
    
    # Fall back to BeautifulSoup scraping
    return fetch_geo_metadata(gse_id)

