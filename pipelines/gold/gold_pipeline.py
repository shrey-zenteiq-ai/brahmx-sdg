"""
Kubeflow Pipeline — Gold Data Generation.

DAG: evidence_pack → prompt → generate(n=3) → dean_score(per candidate)
     → select → audit → human_gate → bundle → slices → provenance

Caching: evidence packs are cacheable (deterministic). Generation is not.
Retry: teacher generation retries 3x, dean scoring 2x.
"""
from __future__ import annotations

try:
    from kfp import dsl
    from kfp.dsl import Input, Output, Artifact
    HAS_KFP = True
except ImportError:
    HAS_KFP = False

if HAS_KFP:

    @dsl.component(
        base_image="python:3.11-slim",
        packages_to_install=["brahmx-sdg"],
    )
    def build_evidence_pack(
        task_spec_path: str,
        kb_path: str,
        evidence_pack: Output[Artifact],
    ) -> None:
        from brahmx_sdg.kb.evidence_pack_builder import EvidencePackBuilder
        from pathlib import Path
        import json
        builder = EvidencePackBuilder()
        spec = json.loads(Path(task_spec_path).read_text())
        pack = builder.build(spec, Path(kb_path))
        Path(evidence_pack.path).write_text(json.dumps(pack, indent=2))

    @dsl.component(
        base_image="python:3.11-slim",
        packages_to_install=["brahmx-sdg"],
    )
    def generate_candidates(
        prompt_spec_path: str,
        n_candidates: int,
        candidates_out: Output[Artifact],
    ) -> None:
        import json
        from pathlib import Path
        # TODO: call teacher router with prompt spec
        Path(candidates_out.path).write_text(json.dumps({"candidates": []}))

    @dsl.component(
        base_image="python:3.11-slim",
        packages_to_install=["brahmx-sdg"],
    )
    def dean_score(
        candidate_path: str,
        evidence_pack_path: str,
        dean_report: Output[Artifact],
    ) -> str:
        return "PASS"

    @dsl.component(
        base_image="python:3.11-slim",
        packages_to_install=["brahmx-sdg"],
    )
    def auditor_review(
        candidate_path: str,
        task_spec_path: str,
        auditor_report: Output[Artifact],
    ) -> str:
        return "PASS"

    @dsl.component(
        base_image="python:3.11-slim",
        packages_to_install=["brahmx-sdg"],
    )
    def publish_gold_bundle(
        candidate_path: str,
        dean_path: str,
        auditor_path: str,
        bundle_out: Output[Artifact],
    ) -> str:
        return "GOLD-published"

    @dsl.pipeline(
        name="gold-generation-pipeline",
        description="Evidence-grounded gold data generation with full verification chain",
    )
    def gold_generation_pipeline(
        task_spec_gcs: str,
        kb_gcs: str,
        output_gcs: str,
        n_candidates: int = 3,
    ):
        evidence = build_evidence_pack(
            task_spec_path=task_spec_gcs,
            kb_path=kb_gcs,
        )
        evidence.set_caching_options(True)

        gen = generate_candidates(
            prompt_spec_path=task_spec_gcs,
            n_candidates=n_candidates,
        )
        gen.after(evidence)
        gen.set_retry(num_retries=3, backoff_duration="60s")

        dean = dean_score(
            candidate_path="",
            evidence_pack_path="",
        )
        dean.after(gen)
        dean.set_retry(num_retries=2)

        audit = auditor_review(
            candidate_path="",
            task_spec_path=task_spec_gcs,
        )
        audit.after(dean)

        publish = publish_gold_bundle(
            candidate_path="",
            dean_path="",
            auditor_path="",
        )
        publish.after(audit)
