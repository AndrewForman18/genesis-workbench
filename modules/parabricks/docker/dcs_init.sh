#!/bin/bash
# DCS init for NGC Parabricks container on DBR 14.3
# All checks non-fatal - logs to Volume for debugging
set -x

# Log everything to a Volume so we can debug after cluster terminates
LOG_DIR="/Volumes/dhbl_discovery_us_dev/genesis_schema/gwas_data/init_logs"
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG_FILE="$LOG_DIR/dcs_init_$(date +%Y%m%d_%H%M%S).log"

# Tee all output to log file AND stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[init] DCS init for NGC Parabricks starting..."
echo "[init] Date: $(date)"
echo "[init] Hostname: $(hostname)"

# ============================================================
# 1. System packages
# ============================================================
apt-get update -qq 2>/dev/null || true
apt-get install --yes --no-install-recommends \
  iproute2 sudo procps coreutils bash acl wget curl \
  mawk gawk findutils 2>/dev/null || true

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/conda/bin:$PATH"

# ============================================================
# 2. Fix getfacl for Docker overlay filesystem
# ============================================================
REAL_GETFACL=$(which getfacl 2>/dev/null || echo "/usr/bin/getfacl")
if [ -f "$REAL_GETFACL" ]; then
  mv "$REAL_GETFACL" "${REAL_GETFACL}.real" 2>/dev/null || true
fi

cat > /usr/bin/getfacl << 'WRAPPER'
#!/bin/bash
if [ -f /usr/bin/getfacl.real ]; then
  /usr/bin/getfacl.real "$@" 2>/dev/null && exit 0
fi
for arg in "$@"; do
  [[ "$arg" == -* ]] && continue
  if [ -e "$arg" ]; then
    OWNER=$(stat -c '%U' "$arg" 2>/dev/null || echo "root")
    GROUP=$(stat -c '%G' "$arg" 2>/dev/null || echo "root")
    echo "# file: $arg"
    echo "# owner: $OWNER"
    echo "# group: $GROUP"
    echo "user::rwx"
    echo "group::r-x"
    echo "other::r-x"
    echo ""
  fi
done
exit 0
WRAPPER
chmod +x /usr/bin/getfacl
echo "[init] getfacl wrapper installed"

# ============================================================
# 3. Install JDK (critical for Spark)
# ============================================================
if ! command -v java &>/dev/null; then
  echo "[init] Java not found, installing..."
  apt-get install --yes --no-install-recommends openjdk-8-jdk 2>/dev/null || \
    apt-get install --yes --no-install-recommends openjdk-11-jdk 2>/dev/null || \
    echo "[init] WARNING: Could not install JDK via apt"
fi

if command -v java &>/dev/null; then
  JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
  export JAVA_HOME
  echo "export JAVA_HOME=${JAVA_HOME}" >> /etc/environment
  echo "[init] Java: $(java -version 2>&1 | head -1)"
else
  echo "[init] WARNING: No Java available - Spark may fail"
fi

# ============================================================
# 4. Discover container Python
# ============================================================
mkdir -p /databricks

echo "[init] Searching for Python in container..."
echo "[init] /opt/conda/bin/python3 exists: $(test -x /opt/conda/bin/python3 && echo YES || echo NO)"
echo "[init] /usr/bin/python3 exists: $(test -x /usr/bin/python3 && echo YES || echo NO)"
echo "[init] /usr/local/bin/python3 exists: $(test -x /usr/local/bin/python3 && echo YES || echo NO)"
echo "[init] which python3: $(which python3 2>/dev/null || echo NOT_FOUND)"
echo "[init] which python: $(which python 2>/dev/null || echo NOT_FOUND)"

CONDA_PYTHON=""
for candidate in /opt/conda/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if [ -x "$candidate" ]; then
    CONDA_PYTHON="$candidate"
    break
  fi
done

if [ -z "$CONDA_PYTHON" ]; then
  echo "[init] FATAL: No Python found anywhere in container!"
  echo "[init] ls /opt/conda/bin/ :"
  ls /opt/conda/bin/ 2>/dev/null | head -20
  echo "[init] ls /usr/bin/python* :"
  ls /usr/bin/python* 2>/dev/null
  # Don't exit - let it fail naturally
  exit 0
fi

echo "[init] Using Python: $CONDA_PYTHON"
echo "[init] Python version: $($CONDA_PYTHON --version 2>&1)"
echo "[init] Python pip: $($CONDA_PYTHON -m pip --version 2>&1 || echo 'pip NOT available')"

# ============================================================
# 5. Create /databricks/python3
# ============================================================
rm -rf /databricks/python3

echo "[init] Attempting: virtualenv --system-site-packages..."
$CONDA_PYTHON -m virtualenv /databricks/python3 --system-site-packages 2>&1
VENV_RC=$?
echo "[init] virtualenv returned: $VENV_RC"

if [ $VENV_RC -ne 0 ] || [ ! -x /databricks/python3/bin/python3 ]; then
  echo "[init] virtualenv failed, trying venv..."
  rm -rf /databricks/python3
  $CONDA_PYTHON -m venv /databricks/python3 --system-site-packages --without-pip 2>&1
  VENV_RC=$?
  echo "[init] venv returned: $VENV_RC"
fi

if [ ! -x /databricks/python3/bin/python3 ]; then
  echo "[init] venv also failed, copying conda env..."
  rm -rf /databricks/python3
  cp -a $(dirname $(dirname $CONDA_PYTHON)) /databricks/python3
  echo "[init] cp exit code: $?"
fi

# Final check
if [ -x /databricks/python3/bin/python3 ]; then
  echo "[init] SUCCESS: /databricks/python3/bin/python3 exists"
  echo "[init] Version: $(/databricks/python3/bin/python3 --version 2>&1)"
else
  echo "[init] FAILED: /databricks/python3/bin/python3 does not exist!"
  echo "[init] Contents of /databricks/python3/bin/:"
  ls -la /databricks/python3/bin/ 2>/dev/null | head -20
fi

# Ensure activate script exists
if [ ! -f /databricks/python3/bin/activate ]; then
  cat > /databricks/python3/bin/activate << 'ACTIVATE'
VIRTUAL_ENV="/databricks/python3"
export VIRTUAL_ENV
export PATH="$VIRTUAL_ENV/bin:$PATH"
ACTIVATE
fi

# ============================================================
# 6. Install required packages (all non-fatal)
# ============================================================
echo "[init] Installing packages..."
/databricks/python3/bin/python3 -m pip install --quiet --upgrade pip 2>&1 || echo "[init] pip upgrade failed (non-fatal)"

/databricks/python3/bin/python3 -m pip install --no-cache-dir \
  virtualenv ipython ipykernel pyarrow "py4j==0.10.9.7" \
  pandas protobuf databricks-sdk pyyaml \
  urllib3 requests certifi six jedi ipython-genutils \
  grpcio grpcio-status 2>&1 || echo "[init] pip install had errors (non-fatal)"

echo "[init] Installed packages:"
/databricks/python3/bin/python3 -m pip list 2>&1 | grep -iE "ipykernel|pyarrow|py4j|numpy|grpcio|pandas" || true

# ============================================================
# 7. Create /databricks/python (real copy)
# ============================================================
rm -rf /databricks/python
cp -a /databricks/python3 /databricks/python
echo "[init] /databricks/python created"

# ============================================================
# 8. Environment variables
# ============================================================
export PYSPARK_PYTHON=/databricks/python3/bin/python3
echo "export PYSPARK_PYTHON=/databricks/python3/bin/python3" >> /etc/environment
echo "PYSPARK_PYTHON=/databricks/python3/bin/python3" >> /etc/environment
echo "PATH=/databricks/python3/bin:/databricks/python/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/conda/bin" >> /etc/environment

useradd libraries 2>/dev/null || true
usermod -L libraries 2>/dev/null || true

# ============================================================
# 9. Virtualenv wrapper
# ============================================================
cat > /usr/bin/virtualenv << 'VENVWRAP'
#!/bin/bash
# Databricks calls: virtualenv <path> -p <python> --no-download --no-setuptools --no-wheel
# Try real virtualenv module first, then fall back to venv --without-pip

/databricks/python3/bin/python3 -m virtualenv "$@" 2>/dev/null
RC=$?
if [ $RC -eq 0 ]; then exit 0; fi

# Parse arguments
TARGET=""
PYPATH="/databricks/python3/bin/python3"
SKIP_NEXT=false
for arg in "$@"; do
  if $SKIP_NEXT; then PYPATH="$arg"; SKIP_NEXT=false; continue; fi
  case "$arg" in
    -p|--python) SKIP_NEXT=true ;;
    --no-download|--no-setuptools|--no-wheel|--clear|--system-site-packages) ;;
    -*) ;;
    *) TARGET="$arg" ;;
  esac
done

if [ -n "$TARGET" ]; then
  # CRITICAL: --without-pip avoids ensurepip dependency (not available in conda Python)
  "$PYPATH" -m venv "$TARGET" --without-pip 2>&1
  VENV_RC=$?
  if [ $VENV_RC -eq 0 ]; then
    # Copy pip/setuptools from source python into the new venv
    SITE_SRC=$("$PYPATH" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    SITE_DST="$TARGET/lib/python3.10/site-packages"
    if [ -d "$SITE_SRC" ] && [ -d "$TARGET/lib" ]; then
      # Find actual python version directory
      PY_DIR=$(ls "$TARGET/lib/" 2>/dev/null | head -1)
      SITE_DST="$TARGET/lib/$PY_DIR/site-packages"
      mkdir -p "$SITE_DST"
      # Copy essential packages from source
      for pkg in pip setuptools virtualenv pkg_resources _virtualenv; do
        if [ -d "$SITE_SRC/$pkg" ]; then
          cp -a "$SITE_SRC/$pkg" "$SITE_DST/" 2>/dev/null
        fi
      done
      # Copy .dist-info directories
      for dist in "$SITE_SRC"/pip-*.dist-info "$SITE_SRC"/setuptools-*.dist-info "$SITE_SRC"/virtualenv-*.dist-info; do
        if [ -d "$dist" ]; then
          cp -a "$dist" "$SITE_DST/" 2>/dev/null
        fi
      done
    fi
    exit 0
  fi
  exit $VENV_RC
fi
exit 1
VENVWRAP
chmod +x /usr/bin/virtualenv
ln -sf /usr/bin/virtualenv /usr/local/bin/virtualenv
ln -sf /databricks/python3/bin/python3 /usr/local/bin/python3
ln -sf /databricks/python/bin/python3 /usr/local/bin/python 2>/dev/null || true

# ============================================================
# 10. Non-fatal validation (log only, never exit non-zero)
# ============================================================
echo "[init] === VALIDATION ==="
/databricks/python3/bin/python3 -c "
import sys; print(f'Python: {sys.version}')
try:
    import ipykernel; print(f'ipykernel: {ipykernel.__version__}')
except Exception as e: print(f'ipykernel MISSING: {e}')
try:
    import pyarrow; print(f'pyarrow: {pyarrow.__version__}')
except Exception as e: print(f'pyarrow MISSING: {e}')
try:
    import numpy; print(f'numpy: {numpy.__version__}')
except Exception as e: print(f'numpy MISSING: {e}')
try:
    import py4j; print('py4j: OK')
except Exception as e: print(f'py4j MISSING: {e}')
try:
    import grpc; print(f'grpcio: {grpc.__version__}')
except Exception as e: print(f'grpcio MISSING: {e}')
" 2>&1 || echo "[init] Python validation had issues"

echo "[init] getfacl test:"
getfacl /databricks/python/bin/python3 2>&1 || echo "[init] getfacl failed (non-fatal)"

echo "[init] COMPLETE (exit 0 regardless)"
echo "[init] Log saved to: $LOG_FILE"
exit 0
