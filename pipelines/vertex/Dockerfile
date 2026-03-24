# BrahmX SDG — Vertex AI Pipeline Component Image
#
# Build context must be the repo root:
#   docker build -f pipelines/vertex/Dockerfile -t <IMAGE_URI> .
#
# The image pre-installs the brahmx_sdg package plus GCP client libs so every
# KFP component can import them without a per-component `packages_to_install`
# runtime download (faster cold-start, reproducible versions).

FROM python:3.11-slim

WORKDIR /app

# Build tools required by some transitive deps (e.g. pint, sympy C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Install brahmx_sdg package ────────────────────────────────────────────────
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e "."

# ── GCP client libraries (not in brahmx_sdg deps, needed by components) ──────
RUN pip install --no-cache-dir \
    "google-cloud-storage>=2.16" \
    "google-cloud-secret-manager>=2.20" \
    "google-cloud-firestore>=2.16" \
    "pyyaml>=6.0"

# ── KFP executor (no-deps to avoid version conflicts with installed packages) ─
RUN pip install --no-cache-dir "kfp==2.16.0" --no-deps \
    "typing-extensions>=3.7.4,<5"

# Vertex AI runs the KFP executor; no CMD needed
