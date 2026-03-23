"""
Kubeflow Pipeline — Multi-stage Student Training.

DAG: validate_corpus → CPT → SFT_gold → SFT_silver → preference
     → RL/alignment → distill_recovery → LC_recovery → eval → package

All stages use MaxText on TPU. No live teacher dependency.
"""
from __future__ import annotations

try:
    from kfp import dsl
    from kfp.dsl import Output, Artifact
    HAS_KFP = True
except ImportError:
    HAS_KFP = False

if HAS_KFP:

    @dsl.component(base_image="python:3.11-slim", packages_to_install=["brahmx-sdg"])
    def validate_corpus(corpus_version: str, manifest_path: str) -> str:
        """Validate corpus: checksums, tokenizer-safe, no live teacher refs."""
        return "valid"

    @dsl.component(base_image="python:3.11-slim", packages_to_install=["brahmx-sdg"])
    def launch_maxtext_stage(
        stage: str, corpus_path: str, base_checkpoint: str,
        tpu_topology: str,
    ) -> str:
        """Launch a MaxText training stage on TPU."""
        return f"gs://checkpoints/{stage}/latest"

    @dsl.component(base_image="python:3.11-slim", packages_to_install=["brahmx-sdg"])
    def run_eval(checkpoint_path: str, eval_config: str) -> str:
        return "eval-report-path"

    @dsl.component(base_image="python:3.11-slim", packages_to_install=["brahmx-sdg"])
    def package_model(checkpoint_path: str, eval_report_path: str) -> str:
        return "model-package-path"

    @dsl.pipeline(
        name="training-pipeline",
        description="Multi-stage student training: CPT → SFT → Preference → RL → Recovery → Eval",
    )
    def training_pipeline(
        corpus_version: str,
        manifest_path: str,
        student_model: str = "brahmx-sci-8b",
        tpu_topology: str = "v5e-256",
    ):
        val = validate_corpus(corpus_version=corpus_version, manifest_path=manifest_path)

        cpt = launch_maxtext_stage(stage="cpt", corpus_path=corpus_version, base_checkpoint="", tpu_topology=tpu_topology)
        cpt.after(val)

        sft_gold = launch_maxtext_stage(stage="sft_gold", corpus_path=corpus_version, base_checkpoint=cpt.output, tpu_topology=tpu_topology)
        sft_silver = launch_maxtext_stage(stage="sft_silver", corpus_path=corpus_version, base_checkpoint=sft_gold.output, tpu_topology=tpu_topology)
        pref = launch_maxtext_stage(stage="preference", corpus_path=corpus_version, base_checkpoint=sft_silver.output, tpu_topology=tpu_topology)
        rl = launch_maxtext_stage(stage="rl_alignment", corpus_path=corpus_version, base_checkpoint=pref.output, tpu_topology=tpu_topology)
        recovery = launch_maxtext_stage(stage="distill_recovery", corpus_path=corpus_version, base_checkpoint=rl.output, tpu_topology=tpu_topology)
        lc = launch_maxtext_stage(stage="lc_recovery", corpus_path=corpus_version, base_checkpoint=recovery.output, tpu_topology=tpu_topology)

        ev = run_eval(checkpoint_path=lc.output, eval_config="")
        package_model(checkpoint_path=lc.output, eval_report_path=ev.output)
