#!/bin/bash
# Databricks Container Services compatibility init script
# Installs JDK 8, iproute2, R, Python virtualenv BEFORE Spark/JVM starts.
# Used with NGC images that don't include Databricks requirements.
set -ex

echo "[init] Installing Databricks Container Services requirements..."

# Install core system packages (JDK 8 is critical for Spark)
apt-get update && apt-get install --yes --no-install-recommends \
  openjdk-8-jdk \
  iproute2 \
  bash \
  sudo \
  coreutils \
  procps \
  acl \
  wget \
  software-properties-common \
  apt-transport-https \
  libssl-dev

# Configure Java certificates
/var/lib/dpkg/info/ca-certificates-java.postinst configure 2>/dev/null || true

# Install R (required for Databricks driver setup)
export DEBIAN_FRONTEND=noninteractive
gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 2>/dev/null || true
gpg -a --export E298A3A825C0D65DFD57CBB651716619E084DAB9 | sudo apt-key add - 2>/dev/null || true
add-apt-repository -y "deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu $(lsb_release -cs)-cran40/" 2>/dev/null || true
apt-get update && apt-get install --yes --no-install-recommends r-base r-base-dev 2>/dev/null || true

# Create libraries user (required by Databricks)
useradd libraries 2>/dev/null || true
usermod -L libraries 2>/dev/null || true

# Create /databricks/python3 virtualenv if it doesn't exist
mkdir -p /databricks
if [ ! -f /databricks/python3/bin/python3 ]; then
  python -m venv --system-site-packages /databricks/python3
fi

# Install core Databricks Python packages
/databricks/python3/bin/pip install --quiet --no-cache-dir \
  six jedi ipython ipython-genutils numpy pandas pyarrow \
  matplotlib Jinja2 ipykernel protobuf pyccolo \
  grpcio grpcio-status databricks-sdk

# Set PYSPARK_PYTHON
export PYSPARK_PYTHON=/databricks/python3/bin/python3
echo "export PYSPARK_PYTHON=/databricks/python3/bin/python3" >> /etc/environment

# Create Python LSP venv for notebook autocomplete
if [ ! -f /databricks/python-lsp/bin/python3 ]; then
  python -m venv --system-site-packages /databricks/python-lsp
fi
/databricks/python-lsp/bin/pip install --quiet --no-cache-dir \
  python-lsp-server pylsp-mypy python-lsp-black 2>/dev/null || true

# Cleanup to save disk
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

echo "[init] Databricks DCS requirements installed successfully"
