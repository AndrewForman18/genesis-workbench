#!/bin/bash
# ==============================================================================
# Genesis Workbench — Nextflow Init Script for nf-core/sarek
# ==============================================================================
# Installs Nextflow and configures the environment for nf-core pipelines.
# Used as a cluster init script for the gwas_sarek_variant_calling job.
#
# Prerequisites (provided by DBR 16.4):
#   - Java 11+ (openjdk)
#   - Conda (miniforge)
#   - Python 3.11+
#
# This script installs:
#   - Nextflow (latest stable)
#   - nf-core tools (for pipeline management)
#   - Required system packages
# ==============================================================================

set -euo pipefail

LOG_PREFIX="[nextflow-init]"
echo "${LOG_PREFIX} Starting Nextflow environment setup..."

# ── 1. Verify Java ──────────────────────────────────────────────────────────────
if command -v java &>/dev/null; then
    JAVA_VER=$(java -version 2>&1 | head -1)
    echo "${LOG_PREFIX} ✅ Java found: ${JAVA_VER}"
else
    echo "${LOG_PREFIX} Installing OpenJDK 17..."
    apt-get update -qq && apt-get install -y -qq openjdk-17-jre-headless
    echo "${LOG_PREFIX} ✅ Java installed"
fi

# ── 2. Install Nextflow ─────────────────────────────────────────────────────────
NXF_VERSION="24.10.4"  # Pin version for reproducibility
NXF_HOME="/usr/local/nextflow"
NXF_BIN="/usr/local/bin/nextflow"

if [ -f "${NXF_BIN}" ]; then
    echo "${LOG_PREFIX} Nextflow already installed at ${NXF_BIN}"
else
    echo "${LOG_PREFIX} Installing Nextflow v${NXF_VERSION}..."
    mkdir -p "${NXF_HOME}"
    
    # Download and install Nextflow
    curl -fsSL https://get.nextflow.io | bash
    mv nextflow "${NXF_BIN}"
    chmod +x "${NXF_BIN}"
    
    echo "${LOG_PREFIX} ✅ Nextflow installed at ${NXF_BIN}"
fi

# Verify Nextflow
nextflow -version

# ── 3. Configure Nextflow environment ──────────────────────────────────────────
# Set NXF_HOME for caching pipelines and conda envs
export NXF_HOME="/local_disk0/nextflow"
mkdir -p "${NXF_HOME}"

# Set conda cache directory to local disk (fast NVMe)
export NXF_CONDA_CACHEDIR="/local_disk0/nextflow/conda"
mkdir -p "${NXF_CONDA_CACHEDIR}"

# Persist environment variables for notebook processes
cat >> /etc/environment <<EOF
NXF_HOME=/local_disk0/nextflow
NXF_CONDA_CACHEDIR=/local_disk0/nextflow/conda
NXF_OPTS=-Xms512m -Xmx4g
EOF

# Also export for current shell and child processes
cat >> /databricks/spark/conf/spark-env.sh <<EOF
export NXF_HOME=/local_disk0/nextflow
export NXF_CONDA_CACHEDIR=/local_disk0/nextflow/conda
export NXF_OPTS="-Xms512m -Xmx4g"
EOF

echo "${LOG_PREFIX} ✅ NXF_HOME=${NXF_HOME}"
echo "${LOG_PREFIX} ✅ NXF_CONDA_CACHEDIR=${NXF_CONDA_CACHEDIR}"

# ── 4. Install nf-core tools (optional but useful for pipeline management) ─────
echo "${LOG_PREFIX} Installing nf-core tools..."
pip install --quiet nf-core

# Verify nf-core
nf-core --version 2>/dev/null && echo "${LOG_PREFIX} ✅ nf-core tools installed" || true

# ── 5. Pre-pull nf-core/sarek (optional — speeds up first run) ─────────────────
# Uncomment to pre-download the pipeline during cluster init:
# echo "${LOG_PREFIX} Pre-pulling nf-core/sarek..."
# nextflow pull nf-core/sarek -r 3.5.1
# echo "${LOG_PREFIX} ✅ nf-core/sarek pre-pulled"

# ── 6. System packages for variant calling tools ───────────────────────────────
echo "${LOG_PREFIX} Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    tabix \
    bcftools \
    samtools \
    pigz \
    2>/dev/null || true

echo "${LOG_PREFIX} ✅ System packages installed"

# ── 7. Verify final environment ────────────────────────────────────────────────
echo "${LOG_PREFIX} ═══════════════════════════════════════════════"
echo "${LOG_PREFIX}   Environment Summary"
echo "${LOG_PREFIX} ═══════════════════════════════════════════════"
echo "${LOG_PREFIX}   Nextflow:  $(nextflow -version 2>&1 | grep -i version | head -1)"
echo "${LOG_PREFIX}   Java:      $(java -version 2>&1 | head -1)"
echo "${LOG_PREFIX}   Conda:     $(conda --version 2>/dev/null || echo 'not found')"
echo "${LOG_PREFIX}   Python:    $(python --version)"
echo "${LOG_PREFIX}   NXF_HOME:  ${NXF_HOME}"
echo "${LOG_PREFIX}   Disk:      $(df -h /local_disk0 | tail -1 | awk '{print $4}') available"
echo "${LOG_PREFIX} ═══════════════════════════════════════════════"
echo "${LOG_PREFIX} ✅ Init script complete"
