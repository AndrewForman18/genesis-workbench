# =============================================================================
# Parabricks on Databricks — Multi-Stage Docker Build
# =============================================================================
# 
# This image combines:
#   1. databricksruntime/gpu-standard:14.3-LTS  (Databricks Python/JVM/REPL)
#   2. NVIDIA Clara Parabricks 4.5.1            (GPU-accelerated genomics)
#
# The Databricks base provides all cluster infrastructure (virtualenv, PySpark,
# JDK, getfacl, etc.) so NO init script is needed.
#
# Build:
#   docker build -t <registry>/parabricks-databricks:4.5.1-14.3 #     --build-arg NGC_API_KEY=<your-key> .
#
# Push (to ECR, for example):
#   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
#   docker tag parabricks-databricks:4.5.1-14.3 <account>.dkr.ecr.us-east-1.amazonaws.com/parabricks-databricks:4.5.1-14.3
#   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/parabricks-databricks:4.5.1-14.3
#
# Job Config (no init script needed):
#   docker_image.url = "<account>.dkr.ecr.us-east-1.amazonaws.com/parabricks-databricks:4.5.1-14.3"
#   spark_version    = "14.3.x-scala2.12"
#   node_type_id     = "g5.16xlarge" (or p4d.24xlarge for multi-GPU)
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Parabricks source (NGC image)
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1 AS parabricks

# -----------------------------------------------------------------------------
# Stage 2: Databricks GPU base + Parabricks tools
# -----------------------------------------------------------------------------
FROM databricksruntime/gpu-standard:14.3-LTS

LABEL maintainer="andrew_forman@eisai.com"
LABEL description="Databricks-compatible image with NVIDIA Parabricks 4.5.1"
LABEL module="genesis-workbench/parabricks"

# System dependencies required by Parabricks
RUN apt-get update -qq && apt-get install -y --no-install-recommends     libcurl4-openssl-dev     libbz2-dev     liblzma-dev     libdeflate-dev     zlib1g-dev     libncurses5-dev     libnss-sss     && rm -rf /var/lib/apt/lists/*

# Copy Parabricks binaries and libraries from NGC image
COPY --from=parabricks /usr/local/parabricks /usr/local/parabricks

# Copy BWA-MEM2, minimap2, and other bundled tools if present
COPY --from=parabricks /usr/local/bin/bwa-mem2* /usr/local/bin/
COPY --from=parabricks /usr/local/bin/minimap2* /usr/local/bin/
COPY --from=parabricks /usr/local/bin/samtools* /usr/local/bin/
COPY --from=parabricks /usr/local/bin/bcftools* /usr/local/bin/

# Add Parabricks to PATH
ENV PATH="/usr/local/parabricks/bin:/usr/local/parabricks:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/parabricks/lib:${LD_LIBRARY_PATH}"

# Install Python genomics packages into the Databricks Python environment
RUN /databricks/python3/bin/pip install --no-cache-dir     pysam     biopython     pyvcf3     databricks-sdk

# Verify Parabricks is accessible
RUN pbrun version || echo "WARN: pbrun not in expected location, check COPY paths"

# Verify Databricks Python env is intact
RUN /databricks/python3/bin/python3 --version &&     /databricks/python3/bin/python3 -c "import virtualenv; print(f'virtualenv {virtualenv.__version__}')"
