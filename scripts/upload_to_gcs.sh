#!/usr/bin/env bash
# BrahmX SDG — Upload pipeline assets to GCS before running on Vertex AI
#
# Uploads:
#   configs/specs/*.json          → gs://BUCKET/specs/
#   data/kb/chunks/*.json         → gs://BUCKET/kb/chunks/
#   configs/routing/models.yaml   → gs://BUCKET/configs/models.yaml
#
# Run from the repo root:
#   source pipelines/vertex/.env
#   bash scripts/upload_to_gcs.sh

set -euo pipefail

ENV_FILE="pipelines/vertex/.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${PIPELINE_BUCKET:?Set PIPELINE_BUCKET}"

BUCKET="gs://${PIPELINE_BUCKET}"

echo "=== BrahmX SDG — GCS Asset Upload ==="
echo "  Bucket: $BUCKET"
echo "  Project: $PROJECT_ID"
echo ""

# ── Task specs ────────────────────────────────────────────────────────────────
SPECS_COUNT=$(find configs/specs -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$SPECS_COUNT" -gt 0 ]]; then
    echo "[1/3] Uploading $SPECS_COUNT task spec(s) → $BUCKET/specs/ ..."
    gsutil -m cp configs/specs/*.json "$BUCKET/specs/"
else
    echo "[1/3] No task specs found in configs/specs/ — skipping."
fi

# ── KB chunks ─────────────────────────────────────────────────────────────────
KB_COUNT=$(find data/kb/chunks -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$KB_COUNT" -gt 0 ]]; then
    echo "[2/3] Uploading $KB_COUNT KB chunk file(s) → $BUCKET/kb/chunks/ ..."
    gsutil -m cp data/kb/chunks/*.json "$BUCKET/kb/chunks/"
else
    echo "[2/3] No KB chunks found in data/kb/chunks/ — skipping."
    echo "      Run 'brahmx-sdg ingest run ...' first, or add chunks manually."
fi

# ── Routing config ────────────────────────────────────────────────────────────
if [[ -f "configs/routing/models.yaml" ]]; then
    echo "[3/3] Uploading routing config → $BUCKET/configs/models.yaml ..."
    gsutil cp configs/routing/models.yaml "$BUCKET/configs/models.yaml"
else
    echo "[3/3] configs/routing/models.yaml not found — skipping."
fi

echo ""
echo "=== Upload complete ==="
echo ""
echo "Pipeline parameter values to use:"
echo "  spec_gcs_uris:          Each gs://${PIPELINE_BUCKET}/specs/<name>.json"
echo "  kb_gcs_prefix:          gs://${PIPELINE_BUCKET}/kb/"
echo "  gold_gcs_prefix:        gs://${PIPELINE_BUCKET}/gold/"
echo "  corpus_gcs_prefix:      gs://${PIPELINE_BUCKET}/corpus/v0.1/"
echo "  routing_config_gcs_uri: gs://${PIPELINE_BUCKET}/configs/models.yaml"
echo ""
echo "To list uploaded specs:"
echo "  gsutil ls $BUCKET/specs/"
