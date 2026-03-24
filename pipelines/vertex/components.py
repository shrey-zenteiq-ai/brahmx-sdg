"""
BrahmX SDG — Vertex AI KFP Component Definitions
=================================================

Each function decorated with @dsl.component is serialized by KFP and executed
inside the Docker container built from pipelines/vertex/Dockerfile.

Rules that apply to every component:
  - ALL imports must be inside the function body (KFP serialises the function
    in isolation; module-level imports are not included).
  - GCS I/O pattern: download inputs → run brahmx_sdg code → upload outputs.
  - Secrets (OpenAI key) are fetched from Secret Manager at runtime so they
    never appear in pipeline parameter logs.
  - Return values are JSON strings so KFP can pass them between tasks.

IMAGE_URI is resolved at compile time from the BRAHMX_IMAGE_URI env var.
Set it before running pipeline.py:
    export BRAHMX_IMAGE_URI=us-central1-docker.pkg.dev/PROJECT/brahmx/sdg:latest
"""

import os

from kfp import dsl

# Resolved at pipeline compile time — set BRAHMX_IMAGE_URI in your env.
_IMAGE_URI = os.environ.get(
    "BRAHMX_IMAGE_URI",
    "us-central1-docker.pkg.dev/zenteiq-lxp-1722918338008/brahmx/sdg:latest",
)


# ── Shared GCS helpers (copied into each component as inner functions) ─────────
# KFP components are self-contained; we cannot share a helper module across them.
# Keep these functions identical in each component that needs GCS access.

_GCS_HELPERS = """
def _parse_gcs_uri(uri: str):
    assert uri.startswith("gs://"), f"Expected gs:// URI, got: {uri}"
    parts = uri[5:].split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")

def _gcs_download_file(client, bucket_name: str, blob_name: str, local_path: str):
    from pathlib import Path
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)

def _gcs_download_prefix(client, bucket_name: str, prefix: str, local_dir: str,
                          skip_subdirs: bool = False):
    from pathlib import Path
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue
        if skip_subdirs and "/" in rel:
            continue
        dest = Path(local_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

def _gcs_upload_file(client, local_path: str, bucket_name: str, blob_name: str):
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)

def _gcs_upload_dir(client, local_dir: str, bucket_name: str, prefix: str):
    from pathlib import Path
    for f in Path(local_dir).rglob("*"):
        if f.is_file():
            rel = f.relative_to(local_dir)
            blob_name = f"{prefix.rstrip('/')}/{rel}"
            client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(f))
"""


# ── 1. Ingest Sources ──────────────────────────────────────────────────────────

@dsl.component(base_image=_IMAGE_URI)
def ingest_sources(
    source_gcs_uri: str,
    kb_gcs_prefix: str,
    domain: str = "",
    chunk_size: int = 2048,
    license_tag: str = "unknown",
) -> str:
    """
    Download source documents from GCS, chunk them into KB atoms, and upload
    the resulting chunk JSON files back to GCS.

    Args:
        source_gcs_uri:  gs:// URI of a single file OR a directory prefix to ingest.
        kb_gcs_prefix:   gs:// prefix where KB chunks will be written
                         (e.g. gs://bucket/kb/).
        domain:          Domain tag applied to all ingested atoms.
        chunk_size:      Characters per chunk (~512 tokens).
        license_tag:     License tag for provenance.

    Returns:
        JSON string with keys: atom_count, doc_count, rejected_count.
    """
    import json
    import tempfile
    from pathlib import Path
    from google.cloud import storage

    # ── inline GCS helpers ────────────────────────────────────────────────────
    def _parse_gcs_uri(uri):
        assert uri.startswith("gs://"), f"Expected gs:// URI, got: {uri}"
        parts = uri[5:].split("/", 1)
        return parts[0], (parts[1] if len(parts) > 1 else "")

    def _gcs_download_file(client, bucket_name, blob_name, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)

    def _gcs_download_prefix(client, bucket_name, prefix, local_dir):
        bucket = client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix):
            rel = blob.name[len(prefix):].lstrip("/")
            if not rel:
                continue
            dest = Path(local_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))

    def _gcs_upload_dir(client, local_dir, bucket_name, prefix):
        for f in Path(local_dir).rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_dir)
                blob_name = f"{prefix.rstrip('/')}/{rel}"
                client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(f))

    # ── main logic ────────────────────────────────────────────────────────────
    gcs = storage.Client()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        local_source = tmp / "source"
        local_kb = tmp / "kb"
        local_source.mkdir()
        local_kb.mkdir()

        # Download source — single file or directory prefix
        src_bucket, src_key = _parse_gcs_uri(source_gcs_uri)
        _SINGLE_FILE_EXTS = {".txt", ".md", ".pdf", ".json", ".jsonl", ".tex", ".py"}
        if any(src_key.endswith(ext) for ext in _SINGLE_FILE_EXTS):
            local_file = local_source / Path(src_key).name
            _gcs_download_file(gcs, src_bucket, src_key, str(local_file))
        else:
            _gcs_download_prefix(gcs, src_bucket, src_key, str(local_source))

        # Run ingestion
        from brahmx_sdg.ingestion.source_ingestion import SourceIngestionPipeline

        pipeline = SourceIngestionPipeline(
            chunk_size=chunk_size,
            domain=domain,
            license_tag=license_tag,
            redistribution_allowed=True,
        )
        result = pipeline.run(source_path=local_source, output_dir=local_kb)

        # Upload KB chunks
        kb_bucket, kb_prefix = _parse_gcs_uri(kb_gcs_prefix)
        _gcs_upload_dir(gcs, str(local_kb), kb_bucket, kb_prefix)

    return json.dumps(result)


# ── 2. Generate Gold Bundle ────────────────────────────────────────────────────

@dsl.component(base_image=_IMAGE_URI)
def generate_gold_bundle(
    project_id: str,
    spec_gcs_uri: str,
    kb_gcs_prefix: str,
    gold_gcs_prefix: str,
    routing_config_gcs_uri: str,
    openai_secret_id: str = "openai-api-key",
    no_llm_dean: bool = False,
) -> str:
    """
    Download a task spec + KB chunks from GCS, run the full gold generation
    pipeline, and upload the resulting GOLD bundle + slice files to GCS.

    The OpenAI API key is fetched from Secret Manager (never logged).

    Args:
        project_id:             GCP project ID (used for Secret Manager).
        spec_gcs_uri:           gs:// URI of the task spec JSON file.
        kb_gcs_prefix:          gs:// prefix of the knowledge base chunks dir.
        gold_gcs_prefix:        gs:// prefix where bundles will be written.
        routing_config_gcs_uri: gs:// URI of configs/routing/models.yaml.
        openai_secret_id:       Secret Manager secret name holding OPENAI_API_KEY.
        no_llm_dean:            If True, skips LLM rubric scoring in Dean.

    Returns:
        JSON string: {success, bundle_id, reason, spec_gcs_uri,
                      bundle_gcs_uri, slices_gcs_prefix}.
    """
    import json
    import os
    import tempfile
    from pathlib import Path
    from google.cloud import secretmanager, storage

    # ── inline GCS helpers ────────────────────────────────────────────────────
    def _parse_gcs_uri(uri):
        assert uri.startswith("gs://"), f"Expected gs:// URI, got: {uri}"
        parts = uri[5:].split("/", 1)
        return parts[0], (parts[1] if len(parts) > 1 else "")

    def _gcs_download_file(client, bucket_name, blob_name, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)

    def _gcs_download_prefix(client, bucket_name, prefix, local_dir):
        bucket = client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix):
            rel = blob.name[len(prefix):].lstrip("/")
            if not rel:
                continue
            dest = Path(local_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))

    def _gcs_upload_file(client, local_path, bucket_name, blob_name):
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)

    def _gcs_upload_dir(client, local_dir, bucket_name, prefix):
        for f in Path(local_dir).rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_dir)
                blob_name = f"{prefix.rstrip('/')}/{rel}"
                client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(f))

    # ── fetch secret ──────────────────────────────────────────────────────────
    sm = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project_id}/secrets/{openai_secret_id}/versions/latest"
    api_key = sm.access_secret_version(request={"name": secret_name}).payload.data.decode()
    os.environ["OPENAI_API_KEY"] = api_key

    gcs = storage.Client()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Download task spec
        spec_bucket, spec_blob = _parse_gcs_uri(spec_gcs_uri)
        local_spec = tmp / "spec.json"
        _gcs_download_file(gcs, spec_bucket, spec_blob, str(local_spec))

        # Download KB chunks
        kb_bucket, kb_prefix = _parse_gcs_uri(kb_gcs_prefix)
        local_kb = tmp / "kb"
        local_kb.mkdir()
        _gcs_download_prefix(gcs, kb_bucket, kb_prefix, str(local_kb))

        # Download routing config
        rc_bucket, rc_blob = _parse_gcs_uri(routing_config_gcs_uri)
        local_rc = tmp / "models.yaml"
        _gcs_download_file(gcs, rc_bucket, rc_blob, str(local_rc))

        # Output dir
        local_gold = tmp / "gold"
        local_gold.mkdir()

        # Run gold generation
        from brahmx_sdg.generation.gold_generator import GoldGenerator

        generator = GoldGenerator(routing_config=str(local_rc))
        if no_llm_dean:
            from brahmx_sdg.verification.dean import Dean
            generator._dean = Dean(routing_config=str(local_rc), use_llm_scoring=False)

        result = generator.run(
            spec_path=local_spec,
            kb_path=local_kb,
            output_dir=local_gold,
        )

        gold_bucket, gold_prefix = _parse_gcs_uri(gold_gcs_prefix)

        if result.success:
            # Upload GOLD-*.json bundle file
            for f in local_gold.glob("GOLD-*.json"):
                _gcs_upload_file(
                    gcs, str(f),
                    gold_bucket,
                    f"{gold_prefix.rstrip('/')}/{f.name}",
                )

            # Upload slice files
            slices_dir = local_gold / "slices"
            if slices_dir.exists():
                _gcs_upload_dir(
                    gcs, str(slices_dir),
                    gold_bucket,
                    f"{gold_prefix.rstrip('/')}/slices",
                )

        bundle_gcs_uri = (
            f"{gold_gcs_prefix.rstrip('/')}/{result.bundle_id}.json"
            if result.success else ""
        )
        slices_gcs_prefix = (
            f"{gold_gcs_prefix.rstrip('/')}/slices/"
            if result.success else ""
        )

        return json.dumps({
            "success": result.success,
            "bundle_id": result.bundle_id if result.success else "",
            "reason": result.reason if not result.success else "",
            "spec_gcs_uri": spec_gcs_uri,
            "bundle_gcs_uri": bundle_gcs_uri,
            "slices_gcs_prefix": slices_gcs_prefix,
        })


# ── 3. Assemble JSONL Corpus ───────────────────────────────────────────────────

@dsl.component(base_image=_IMAGE_URI)
def assemble_corpus(
    gold_gcs_prefix: str,
    output_gcs_prefix: str,
    corpus_version: str = "v0.1",
) -> str:
    """
    Download all GOLD-*.json bundles from GCS, assemble them into JSONL corpus
    files (one per slice type + all_slices.jsonl), and upload to GCS.

    Slice subdirectory is intentionally excluded from download — the
    CorpusAssembler re-emits slices from each bundle's content directly.

    Args:
        gold_gcs_prefix:    gs:// prefix where GOLD-*.json files live.
        output_gcs_prefix:  gs:// prefix where JSONL files will be written.
        corpus_version:     Version tag embedded in the assembly result.

    Returns:
        JSON string: {version, total_examples, slices_by_type, output_gcs_prefix}.
    """
    import json
    import tempfile
    from pathlib import Path
    from google.cloud import storage
    import yaml

    # ── inline GCS helpers ────────────────────────────────────────────────────
    def _parse_gcs_uri(uri):
        assert uri.startswith("gs://"), f"Expected gs:// URI, got: {uri}"
        parts = uri[5:].split("/", 1)
        return parts[0], (parts[1] if len(parts) > 1 else "")

    def _gcs_download_gold_bundles(client, bucket_name, prefix, local_dir):
        """Download only top-level GOLD-*.json files (skip slices/ subdir)."""
        from pathlib import Path
        bucket = client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix):
            rel = blob.name[len(prefix):].lstrip("/")
            if not rel or "/" in rel:
                # Skip empty prefix match and subdirectory entries (slices/)
                continue
            if not rel.startswith("GOLD-") or not rel.endswith(".json"):
                continue
            dest = Path(local_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))

    def _gcs_upload_dir(client, local_dir, bucket_name, prefix):
        for f in Path(local_dir).rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_dir)
                blob_name = f"{prefix.rstrip('/')}/{rel}"
                client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(f))

    gcs = storage.Client()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        local_gold = tmp / "gold"
        local_gold.mkdir()
        local_corpus = tmp / "corpus"
        local_corpus.mkdir()

        # Download gold bundles
        gold_bucket, gold_prefix = _parse_gcs_uri(gold_gcs_prefix)
        _gcs_download_gold_bundles(gcs, gold_bucket, gold_prefix, str(local_gold))

        bundle_count = len(list(local_gold.glob("GOLD-*.json")))
        if bundle_count == 0:
            return json.dumps({
                "version": corpus_version,
                "total_examples": 0,
                "slices_by_type": {},
                "output_gcs_prefix": output_gcs_prefix,
                "warning": "No GOLD-*.json files found at the given prefix.",
            })

        # Write a minimal manifest so CorpusAssembler can find the bundles
        manifest_data = {
            "version": corpus_version,
            "gold_paths": [str(local_gold)],
            "silver_paths": [],
            "mixture": {"task_types": [], "max_per_type": 0},
            "format": "jsonl",
        }
        manifest_path = tmp / "manifest.yaml"
        manifest_path.write_text(yaml.dump(manifest_data))

        # Assemble
        from brahmx_sdg.corpus.corpus_assembler import CorpusAssembler

        assembler = CorpusAssembler()
        result = assembler.run(manifest_path=manifest_path, output_path=local_corpus)

        # Upload JSONL files
        out_bucket, out_prefix = _parse_gcs_uri(output_gcs_prefix)
        _gcs_upload_dir(gcs, str(local_corpus), out_bucket, out_prefix)

        return json.dumps({
            "version": result.version,
            "total_examples": result.total_examples,
            "slices_by_type": result.slices_by_type,
            "output_gcs_prefix": output_gcs_prefix,
        })


# ── 4. Write to Human Review Queue ────────────────────────────────────────────

@dsl.component(base_image=_IMAGE_URI)
def write_to_human_review_queue(
    project_id: str,
    generation_result_json: str,
    gold_gcs_prefix: str,
) -> str:
    """
    Read the result from generate_gold_bundle. If it succeeded, download the
    bundle from GCS and write a review record to Firestore.

    Skips silently for failed bundles (audit FAIL or generation error) so the
    pipeline does not break on individual spec failures.

    Args:
        project_id:              GCP project ID.
        generation_result_json:  Output JSON string from generate_gold_bundle.
        gold_gcs_prefix:         gs:// prefix where GOLD-*.json files live.

    Returns:
        JSON string: {status, record_id, bundle_id} or {status, reason}.
    """
    import json
    import uuid
    import datetime
    from google.cloud import firestore, storage

    def _parse_gcs_uri(uri):
        assert uri.startswith("gs://"), f"Expected gs:// URI, got: {uri}"
        parts = uri[5:].split("/", 1)
        return parts[0], (parts[1] if len(parts) > 1 else "")

    # Parse upstream result
    try:
        gen_result = json.loads(generation_result_json)
    except json.JSONDecodeError:
        return json.dumps({"status": "ERROR", "reason": "Could not parse generation_result_json."})

    if not gen_result.get("success"):
        return json.dumps({
            "status": "SKIPPED",
            "reason": gen_result.get("reason", "Generation failed."),
            "spec_gcs_uri": gen_result.get("spec_gcs_uri", ""),
        })

    bundle_id = gen_result["bundle_id"]
    bundle_gcs_uri = gen_result["bundle_gcs_uri"]

    # Download the bundle to read its auditor status
    gcs = storage.Client()
    try:
        bucket_name, blob_name = _parse_gcs_uri(bundle_gcs_uri)
        bundle_text = gcs.bucket(bucket_name).blob(blob_name).download_as_text()
        bundle_data = json.loads(bundle_text)
    except Exception as exc:
        return json.dumps({"status": "ERROR", "reason": f"Could not download bundle: {exc}"})

    auditor_status = bundle_data.get("auditor_report", {}).get("status", "UNKNOWN")
    escalate = bundle_data.get("auditor_report", {}).get("escalate_to_human", False)

    # Only queue bundles that passed (or that the Auditor flagged for escalation)
    if auditor_status != "PASS" and not escalate:
        return json.dumps({
            "status": "SKIPPED",
            "reason": f"Auditor status {auditor_status} — not queued.",
            "bundle_id": bundle_id,
        })

    # Write to Firestore
    db = firestore.Client(project=project_id)
    record_id = str(uuid.uuid4())

    doc_data = {
        "record_id": record_id,
        "bundle_id": bundle_id,
        "status": "PENDING_REVIEW",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "bundle_gcs_uri": bundle_gcs_uri,
        "slices_gcs_prefix": gen_result.get("slices_gcs_prefix", ""),
        "spec_gcs_uri": gen_result.get("spec_gcs_uri", ""),
        "auditor_status": auditor_status,
        "escalate_to_human": escalate,
        "domain": bundle_data.get("task_spec", {}).get("domain", ""),
        "section_id": bundle_data.get("task_spec", {}).get("section_id", ""),
        "selected_candidate_id": bundle_data.get("selected_candidate_id", ""),
    }

    try:
        db.collection("human_review_queue").document(record_id).set(doc_data)
    except Exception as exc:
        return json.dumps({"status": "ERROR", "reason": f"Firestore write failed: {exc}"})

    return json.dumps({
        "status": "SUCCESS",
        "record_id": record_id,
        "bundle_id": bundle_id,
    })
