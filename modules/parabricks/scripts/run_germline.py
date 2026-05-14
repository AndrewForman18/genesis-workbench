#!/usr/bin/env python3
"""
Parabricks Germline Pipeline - DCS Container Script
Runs as spark_python_task to avoid IPython kernel instability on NGC containers.
"""
import os
import sys
import subprocess

def main():
    # Read parameters from sys.argv: --key=value format
    params = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            params[key] = val

    fastq_r1 = params.get("fastq_r1", "")
    fastq_r2 = params.get("fastq_r2", "")
    reference_genome_path = params.get("reference_genome_path", "")
    output_volume_path = params.get("output_volume_path", "")
    mlflow_run_id = params.get("mlflow_run_id", "")

    print(f"=== Parabricks Germline Pipeline ===")
    print(f"FASTQ R1: {fastq_r1}")
    print(f"FASTQ R2: {fastq_r2}")
    print(f"Reference: {reference_genome_path}")
    print(f"Output: {output_volume_path}")

    # Validate inputs
    if not fastq_r1 or not fastq_r2 or not reference_genome_path:
        print("ERROR: Required parameters missing!")
        sys.exit(1)

    for label, path in [("FASTQ R1", fastq_r1), ("FASTQ R2", fastq_r2), ("Reference", reference_genome_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  {label}: {size_mb:.0f} MB")

    # Setup output directory
    run_id = mlflow_run_id if mlflow_run_id else "latest"
    run_output_dir = os.path.join(output_volume_path, "alignment", run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    output_bam = os.path.join(run_output_dir, "output.bam")
    output_vcf = os.path.join(run_output_dir, "germline.vcf")

    # Check GPU
    print("\n=== GPU Check ===")
    subprocess.run(["nvidia-smi"], check=True)
    print("")
    subprocess.run(["pbrun", "--version"], check=True)

    # Step 1: fq2bam (alignment)
    print("\n=== Step 1: fq2bam (FASTQ to BAM alignment) ===")
    fq2bam_cmd = [
        "pbrun", "fq2bam",
        "--ref", reference_genome_path,
        "--in-fq", fastq_r1, fastq_r2,
        "--out-bam", output_bam,
        "--low-memory"
    ]
    print(f"Command: {' '.join(fq2bam_cmd)}")
    result = subprocess.run(fq2bam_cmd)
    if result.returncode != 0:
        print(f"ERROR: fq2bam failed with exit code {result.returncode}")
        sys.exit(1)

    # Step 2: haplotypecaller
    print("\n=== Step 2: HaplotypeCaller (variant calling) ===")
    hc_cmd = [
        "pbrun", "haplotypecaller",
        "--ref", reference_genome_path,
        "--in-bam", output_bam,
        "--out-variants", output_vcf
    ]
    print(f"Command: {' '.join(hc_cmd)}")
    result = subprocess.run(hc_cmd)
    if result.returncode != 0:
        print(f"ERROR: haplotypecaller failed with exit code {result.returncode}")
        sys.exit(1)

    # Verify
    print("\n=== Pipeline Complete ===")
    if os.path.exists(output_bam):
        print(f"BAM: {output_bam} ({os.path.getsize(output_bam) / (1024**3):.2f} GB)")
    if os.path.exists(output_vcf):
        print(f"VCF: {output_vcf} ({os.path.getsize(output_vcf) / (1024**2):.1f} MB)")
    print("\nSUCCESS!")

if __name__ == "__main__":
    main()
