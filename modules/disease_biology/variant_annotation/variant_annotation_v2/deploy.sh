#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [Variant Annotation v2 — VEP + ANNOVAR] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [Variant Annotation v2 — VEP + ANNOVAR] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS --force

echo ""
echo "▶️ Granting SP permissions on all Genesis jobs"
echo ""

python3 ../../../../grant_sp_permissions.py --host "$workspace_url" --sp-client-id "${GWB_SP_CLIENT_ID:-7485fd58-7d9d-49b7-ab99-096fba381657}"


echo ""
echo "▶️ [Variant Annotation v2] Deployment complete"
echo "   Jobs:"
echo "   • vep_annovar_annotation_job (annotation pipeline)"
echo "   • vep_annovar_setup_job (reference database setup)"
echo ""
echo "ℹ️  To install reference databases (one-time, ~45-90 min):"
echo "   databricks bundle run vep_annovar_setup_job $EXTRA_PARAMS --no-wait"
echo ""
