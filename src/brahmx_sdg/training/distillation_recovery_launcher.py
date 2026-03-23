"""
Hard Distillation Recovery — recover quality lost during alignment/compression.

Uses response-based offline distillation ONLY (text corpora from stronger checkpoints).
No soft KD, no token-logit matching, no hidden-state matching.
"""
from __future__ import annotations
from pathlib import Path
import structlog
from brahmx_sdg.schemas import TrainingRunSpec, TrainingStage

logger = structlog.get_logger()

class DistillationRecoveryLauncher:
    def launch(self, base_checkpoint: str, recovery_corpus: str, run_spec_path: Path) -> str:
        """Launch a hard distillation recovery stage."""
        # Recovery corpora are pre-generated text from stronger checkpoints
        # They are static artifacts, NOT live teacher outputs
        logger.info("distillation_recovery", base=base_checkpoint, corpus=recovery_corpus)
        # TODO: delegate to MaxTextTrainingLauncher with recovery-specific config
        return f"recovery-{Path(run_spec_path).stem}"
