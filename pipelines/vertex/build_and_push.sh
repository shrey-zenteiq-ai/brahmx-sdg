#!/usr/bin/env bash
# BrahmX SDG — Build and push component Docker image to Artifact Registry
#
# Must be run from the REPO ROOT (not from pipelines/vertex/):
#   bash pipelines/vertex/build_and_push.sh [TAG]
#
# TAG defaults to "latest". Pass a version tag for production builds:
#   bash pipelines/vertex/build_and_push.sh v0.1.0
#
# After a successful push, the IMAGE_URI is written to:
#   pipelines/vertex/.image_uri.env
# Source that file or copy the value into BRAHMX_IMAGE_URI before running
# pipeline.py.

set -euo pipefail

# ── Load env ──────────────────────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

: "${PROJECT_ID:?Set PROJECT_ID (in env or pipelines/vertex/.env)}"
: "${LOCATION:?Set LOCATION}"

ARTIFACT_REPO="${ARTIFACT_REPO:-brahmx}"
IMAGE_NAME="${IMAGE_NAME:-sdg}"
TAG="${1:-latest}"

IMAGE_URI="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${IMAGE_NAME}:${TAG}"

echo "=== BrahmX SDG — Docker Build + Push ==="
echo "  Image: $IMAGE_URI"
echo "  Build context: $(pwd)  (must be repo root)"
echo ""

# Verify we're at the repo root
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: Run this script from the repository root, not from pipelines/vertex/."
    exit 1
fi

# ── Configure Docker auth for Artifact Registry ───────────────────────────────
echo "[1/3] Configuring Docker auth for $LOCATION-docker.pkg.dev ..."
gcloud auth configure-docker "${LOCATION}-docker.pkg.dev" --quiet

# ── Build ─────────────────────────────────────────────────────────────────────
echo "[2/3] Building $IMAGE_URI ..."
docker build \
    --file "pipelines/vertex/Dockerfile" \
    --tag "$IMAGE_URI" \
    --platform linux/amd64 \
    .

# ── Push ──────────────────────────────────────────────────────────────────────
echo "[3/3] Pushing $IMAGE_URI ..."
docker push "$IMAGE_URI"

# Write URI for pipeline.py to consume
URI_FILE="$(dirname "$0")/.image_uri.env"
echo "BRAHMX_IMAGE_URI=${IMAGE_URI}" > "$URI_FILE"

echo ""
echo "=== Done ==="
echo "  Image: $IMAGE_URI"
echo "  URI saved to: $URI_FILE"
echo ""
echo "Load it before compiling the pipeline:"
echo "  source $URI_FILE"
echo "  python pipelines/vertex/pipeline.py"
