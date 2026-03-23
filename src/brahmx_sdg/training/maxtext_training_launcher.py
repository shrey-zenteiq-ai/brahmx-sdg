"""
MaxText Training Launcher — launches training jobs on TPU via MaxText.

MaxText is the primary Model Factory runtime for:
CPT, SFT, preference, RL, distillation recovery, long-context recovery.
Training NEVER calls live teachers.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml, json
import structlog
from brahmx_sdg.schemas import TrainingRunSpec, TrainingStage

logger = structlog.get_logger()

class MaxTextTrainingLauncher:
    def launch(self, run_spec_path: Path) -> str:
        """Launch a MaxText training job from a run spec."""
        with open(run_spec_path) as f:
            raw = yaml.safe_load(f)
        spec = TrainingRunSpec(**raw)

        # Validate: corpus must exist, no live teacher references
        self._validate_spec(spec)

        # Build MaxText config
        maxtext_config = self._build_maxtext_config(spec)

        # Launch job (in production: submit to GKE / Vertex AI)
        job_id = f"train-{spec.run_id}"
        logger.info("training_launched", job_id=job_id, stage=spec.stage.value,
                     model=spec.student_model, tpu=spec.tpu_topology)

        # TODO: actually submit to training cluster
        return job_id

    def _validate_spec(self, spec: TrainingRunSpec) -> None:
        """Validate training spec before launch."""
        assert spec.corpus_version, "Corpus version is required"
        assert spec.student_model, "Student model is required"
        # Critical: no live teacher references allowed
        assert "teacher" not in json.dumps(spec.maxtext_config).lower(), \
            "Training spec must not reference live teachers"

    def _build_maxtext_config(self, spec: TrainingRunSpec) -> dict:
        """Build MaxText YAML config from run spec."""
        base_config = {
            "model_name": spec.student_model,
            "dataset_path": spec.corpus_version,
            "steps": spec.num_steps,
            "checkpoint_period": spec.checkpoint_interval,
            "eval_period": spec.eval_interval,
            "hardware": spec.tpu_topology,
        }
        base_config.update(spec.hyperparameters)
        base_config.update(spec.maxtext_config)
        return base_config
