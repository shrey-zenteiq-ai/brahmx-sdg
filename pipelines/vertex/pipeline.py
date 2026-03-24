"""
BrahmX SDG — Vertex AI Pipeline DAG
====================================

Usage
-----
1.  Export required environment variables (or create pipelines/vertex/.env and
    run `source pipelines/vertex/.env`):

        export PROJECT_ID=your-project-id
        export SA_EMAIL=brahmx-vertex-sa@your-project-id.iam.gserviceaccount.com
        export LOCATION=us-central1
        export PIPELINE_BUCKET=your-pipeline-bucket
        export BRAHMX_IMAGE_URI=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/brahmx/sdg:latest

2.  Compile the pipeline YAML:

        python pipelines/vertex/pipeline.py --compile-only

3.  Compile and submit:

        python pipelines/vertex/pipeline.py

Pipeline DAG
------------

    [ParallelFor over spec_gcs_uris (max 4 concurrent)]
        generate_gold_bundle  ──►  write_to_human_review_queue
        generate_gold_bundle  ──►  write_to_human_review_queue
        ...

    [After all bundles complete]
        assemble_corpus

The corpus assembly step automatically waits for all generate_gold_bundle tasks
because it is defined outside the dsl.ParallelFor block.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from kfp import compiler, dsl
from google.cloud import aiplatform

# Load .env from pipelines/vertex/ if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    load_dotenv(str(_env_file))

# ── Required env vars ─────────────────────────────────────────────────────────
PROJECT_ID      = os.environ.get("PROJECT_ID", "")
SA_EMAIL        = os.environ.get("SA_EMAIL", "")
LOCATION        = os.environ.get("LOCATION", "us-central1")
PIPELINE_BUCKET = os.environ.get("PIPELINE_BUCKET", "")

if not all([PROJECT_ID, SA_EMAIL, PIPELINE_BUCKET]):
    print(
        "ERROR: Set PROJECT_ID, SA_EMAIL, and PIPELINE_BUCKET env vars before running.\n"
        "       Copy pipelines/vertex/.env.example → pipelines/vertex/.env and fill it in.",
        file=sys.stderr,
    )
    sys.exit(1)

# Import components AFTER env is loaded so _IMAGE_URI is resolved correctly
from components import (  # noqa: E402
    generate_gold_bundle,
    assemble_corpus,
    write_to_human_review_queue,
)


# ── Pipeline Definition ────────────────────────────────────────────────────────

@dsl.pipeline(
    name="brahmx-sdg-gold-generation",
    description=(
        "BrahmX SDG: ingest → 3-teacher gold generation → Dean/Auditor → "
        "JSONL corpus assembly → human review queue"
    ),
)
def brahmx_sdg_pipeline(
    project_id: str,
    spec_gcs_uris: list,
    kb_gcs_prefix: str,
    gold_gcs_prefix: str,
    corpus_gcs_prefix: str,
    routing_config_gcs_uri: str,
    openai_secret_id: str = "openai-api-key",
    no_llm_dean: bool = False,
    corpus_version: str = "v0.1",
) -> None:
    """
    Args:
        project_id:             GCP project (used for Secret Manager + Firestore).
        spec_gcs_uris:          List of gs:// URIs pointing to task spec JSON files.
                                Example:
                                  ["gs://bucket/specs/thermo.json",
                                   "gs://bucket/specs/bio.json"]
        kb_gcs_prefix:          gs:// prefix of the KB chunks directory.
                                Chunks must be JSON files under this prefix.
        gold_gcs_prefix:        gs:// prefix where GOLD bundles + slices are written.
        corpus_gcs_prefix:      gs:// prefix where assembled JSONL corpus is written.
        routing_config_gcs_uri: gs:// URI of configs/routing/models.yaml.
        openai_secret_id:       Secret Manager secret name holding OPENAI_API_KEY.
                                Create with: gcloud secrets create openai-api-key ...
        no_llm_dean:            Skip LLM rubric scoring in Dean (faster / cheaper).
                                Citation and symbolic checks still run.
        corpus_version:         Version tag embedded in AssemblyResult.
        generation_parallelism: Max concurrent generate_gold_bundle tasks.
    """
    # ── Fan-out: one generate_gold_bundle per spec ────────────────────────────
    with dsl.ParallelFor(
        items=spec_gcs_uris,
        parallelism=4,
    ) as spec_uri:
        gen_task = generate_gold_bundle(
            project_id=project_id,
            spec_gcs_uri=spec_uri,
            kb_gcs_prefix=kb_gcs_prefix,
            gold_gcs_prefix=gold_gcs_prefix,
            routing_config_gcs_uri=routing_config_gcs_uri,
            openai_secret_id=openai_secret_id,
            no_llm_dean=no_llm_dean,
        )
        # Passing bundles go to the human review queue immediately
        write_to_human_review_queue(
            project_id=project_id,
            generation_result_json=gen_task.output,
            gold_gcs_prefix=gold_gcs_prefix,
        )

    # ── Corpus assembly: runs after ALL generate tasks complete ───────────────
    # Tasks defined outside the `with dsl.ParallelFor` block automatically wait
    # for the loop to finish — no explicit .after() needed.
    assemble_corpus(
        gold_gcs_prefix=gold_gcs_prefix,
        output_gcs_prefix=corpus_gcs_prefix,
        corpus_version=corpus_version,
    )


# ── Compile + Submit ───────────────────────────────────────────────────────────

def _default_parameter_values() -> dict:
    """Build parameter_values from env vars / convention."""
    bucket = PIPELINE_BUCKET
    return {
        "project_id": PROJECT_ID,
        "spec_gcs_uris": [
            f"gs://{bucket}/specs/thermodynamics_first_law.json",
            f"gs://{bucket}/specs/cell_biology_mitosis.json",
            f"gs://{bucket}/specs/linear_algebra_eigenvalues.json",
        ],
        "kb_gcs_prefix":          f"gs://{bucket}/kb/",
        "gold_gcs_prefix":        f"gs://{bucket}/gold/",
        "corpus_gcs_prefix":      f"gs://{bucket}/corpus/v0.1/",
        "routing_config_gcs_uri": f"gs://{bucket}/configs/models.yaml",
        "openai_secret_id":       "openai-api-key",
        "no_llm_dean":            False,
        "corpus_version":         "v0.1",
    }


def compile_pipeline(output_path: str = "brahmx_sdg_pipeline.yaml") -> None:
    compiler.Compiler().compile(
        pipeline_func=brahmx_sdg_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled → {output_path}")


def submit_pipeline(
    compiled_yaml: str = "brahmx_sdg_pipeline.yaml",
    parameter_values: dict | None = None,
    enable_caching: bool = True,
) -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    job = aiplatform.PipelineJob(
        display_name="brahmx-sdg-gold-generation",
        template_path=compiled_yaml,
        pipeline_root=f"gs://{PIPELINE_BUCKET}/pipeline-root",
        parameter_values=parameter_values or _default_parameter_values(),
        enable_caching=enable_caching,
    )
    job.submit(service_account=SA_EMAIL)
    print(f"Pipeline submitted: {job.resource_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and/or submit the BrahmX SDG pipeline.")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile the pipeline YAML but do not submit.",
    )
    parser.add_argument(
        "--output",
        default="brahmx_sdg_pipeline.yaml",
        help="Path for the compiled pipeline YAML (default: brahmx_sdg_pipeline.yaml).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Vertex AI step caching.",
    )
    args = parser.parse_args()

    compile_pipeline(output_path=args.output)

    if not args.compile_only:
        submit_pipeline(
            compiled_yaml=args.output,
            enable_caching=not args.no_cache,
        )
