#!/bin/bash
# =============================================================================
# Build & Push Parabricks-Databricks Docker Image
# =============================================================================
# Prerequisites:
#   - Docker installed locally (or use CI/CD)
#   - NGC API key (for pulling Parabricks image): https://ngc.nvidia.com/setup
#   - AWS ECR access (or DockerHub/other registry)
#
# Usage:
#   export NGC_API_KEY="nvapi-..."
#   export ECR_REGISTRY="<account-id>.dkr.ecr.us-east-1.amazonaws.com"
#   ./build_and_push.sh
# =============================================================================

set -euo pipefail

# Configuration
IMAGE_NAME="parabricks-databricks"
IMAGE_TAG="4.5.1-14.3"
NGC_REGISTRY="nvcr.io"

# Validate environment
if [ -z "${NGC_API_KEY:-}" ]; then
  echo "ERROR: NGC_API_KEY not set. Get one at https://ngc.nvidia.com/setup"
  exit 1
fi

if [ -z "${ECR_REGISTRY:-}" ]; then
  echo "ERROR: ECR_REGISTRY not set (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com)"
  exit 1
fi

echo "=== Step 1: Login to NGC Registry ==="
echo "$NGC_API_KEY" | docker login ${NGC_REGISTRY} --username '$oauthtoken' --password-stdin

echo "=== Step 2: Login to ECR ==="
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ${ECR_REGISTRY}

echo "=== Step 3: Build Image ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build \
  -t ${IMAGE_NAME}:${IMAGE_TAG} \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

echo "=== Step 4: Tag & Push ==="
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "=== DONE ==="
echo "Image: ${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Update your Databricks job with:"
echo "  docker_image.url = \"${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}\""
echo "  docker_image.basic_auth.username = \"AWS\""
echo "  docker_image.basic_auth.password = <ecr-token or use instance profile>"
echo "  spark_version = \"14.3.x-scala2.12\""
echo "  init_scripts = []  (remove init script!)"
