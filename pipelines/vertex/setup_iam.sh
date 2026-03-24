#!/usr/bin/env bash
# BrahmX SDG — Vertex AI IAM + GCP API Setup
#
# Run this ONCE before the first pipeline submission.
# Requires: gcloud CLI authenticated with an account that has Owner or IAM Admin.
#
# Usage:
#   source pipelines/vertex/.env   # or export vars manually
#   bash pipelines/vertex/setup_iam.sh

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${SA_NAME:?Set SA_NAME}"
: "${LOCATION:?Set LOCATION}"
: "${PIPELINE_BUCKET:?Set PIPELINE_BUCKET}"
: "${ARTIFACT_REPO:=brahmx}"          # Artifact Registry repo name
: "${OPENAI_SECRET_ID:=openai-api-key}"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== BrahmX SDG — GCP Setup ==="
echo "  Project:  $PROJECT_ID"
echo "  Location: $LOCATION"
echo "  SA:       $SA_EMAIL"
echo "  Bucket:   gs://$PIPELINE_BUCKET"
echo "  AR Repo:  $LOCATION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO"
echo ""

gcloud config set project "$PROJECT_ID"

# ── 1. Enable required APIs ───────────────────────────────────────────────────
echo "[1/6] Enabling GCP APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    firestore.googleapis.com \
    logging.googleapis.com \
    --project="$PROJECT_ID"

# ── 2. Create GCS bucket (pipeline root + artifact storage) ──────────────────
echo "[2/6] Creating GCS bucket gs://$PIPELINE_BUCKET ..."
if ! gsutil ls -b "gs://$PIPELINE_BUCKET" &>/dev/null; then
    gsutil mb -l "$LOCATION" -p "$PROJECT_ID" "gs://$PIPELINE_BUCKET"
    gsutil versioning set on "gs://$PIPELINE_BUCKET"
    echo "  Created."
else
    echo "  Already exists — skipping."
fi

# ── 3. Create Artifact Registry repository ───────────────────────────────────
echo "[3/6] Creating Artifact Registry repo '$ARTIFACT_REPO' in $LOCATION ..."
if ! gcloud artifacts repositories describe "$ARTIFACT_REPO" \
        --location="$LOCATION" --project="$PROJECT_ID" &>/dev/null; then
    gcloud artifacts repositories create "$ARTIFACT_REPO" \
        --repository-format=docker \
        --location="$LOCATION" \
        --project="$PROJECT_ID" \
        --description="BrahmX SDG component images"
    echo "  Created."
else
    echo "  Already exists — skipping."
fi

# ── 4. Create Service Account ─────────────────────────────────────────────────
echo "[4/6] Creating service account $SA_NAME ..."
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="BrahmX Vertex Pipeline SA" \
        --project="$PROJECT_ID"
    echo "  Created."
else
    echo "  Already exists — skipping role bindings to avoid duplicates."
fi

# ── 5. Grant IAM roles ────────────────────────────────────────────────────────
echo "[5/6] Assigning IAM roles to $SA_EMAIL ..."

ROLES=(
    # Core Vertex AI
    "roles/aiplatform.user"
    # GCS: read/write pipeline artifacts, bundles, corpus
    "roles/storage.objectAdmin"
    # Secret Manager: read OPENAI_API_KEY at runtime
    "roles/secretmanager.secretAccessor"
    # Firestore: write human review queue
    "roles/datastore.user"
    # Logs
    "roles/logging.logWriter"
    # Pull images from Artifact Registry
    "roles/artifactregistry.reader"
)

for ROLE in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="$ROLE" \
        --condition=None \
        --quiet
    echo "  Granted $ROLE"
done

# ── 6. Create OpenAI API Key Secret ──────────────────────────────────────────
echo "[6/6] Setting up Secret Manager secret '$OPENAI_SECRET_ID' ..."
if ! gcloud secrets describe "$OPENAI_SECRET_ID" --project="$PROJECT_ID" &>/dev/null; then
    gcloud secrets create "$OPENAI_SECRET_ID" \
        --replication-policy="automatic" \
        --project="$PROJECT_ID"
    echo ""
    echo "  Secret created. Add your key with:"
    echo "    printf 'sk-...' | gcloud secrets versions add $OPENAI_SECRET_ID --data-file=-"
else
    echo "  Secret already exists."
    echo "  To rotate: printf 'sk-...' | gcloud secrets versions add $OPENAI_SECRET_ID --data-file=-"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Add your OpenAI key to Secret Manager:"
echo "       printf 'sk-...' | gcloud secrets versions add $OPENAI_SECRET_ID --data-file=-"
echo ""
echo "  2. Build and push the component Docker image:"
echo "       bash pipelines/vertex/build_and_push.sh"
echo ""
echo "  3. Upload pipeline assets to GCS:"
echo "       bash scripts/upload_to_gcs.sh"
echo ""
echo "  4. Compile and submit the pipeline:"
echo "       export BRAHMX_IMAGE_URI=\$(cat pipelines/vertex/.image_uri.env | cut -d= -f2)"
echo "       python pipelines/vertex/pipeline.py"
