"""
Silver Generator — Breadth expansion on top of gold truth.

Silver data is source-grounded, validator-backed, and never promoted
to training without tokenizer checks, provenance, and release approval.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import structlog

from brahmx_sdg.schemas import SilverBundle, TaskSpec, LaneName

logger = structlog.get_logger()


@dataclass
class SilverResult:
    bundle_count: int = 0
    rejected_count: int = 0


class SilverGenerator:
    """Orchestrates silver data expansion across specialized lanes."""

    def __init__(self, lane_configs: Optional[dict] = None) -> None:
        self.lane_configs = lane_configs or {}

    def run(
        self,
        spec_path: Path,
        gold_ref: Path,
        output_dir: Path,
    ) -> dict[str, int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Load gold bundles as reference
        gold_bundles = self._load_gold_refs(gold_ref)
        result = SilverResult()

        for gold_bundle in gold_bundles:
            for lane in self._get_applicable_lanes(gold_bundle):
                try:
                    silver = self._generate_lane(gold_bundle, lane)
                    if silver and silver.validator_pass:
                        self._write_bundle(silver, output_dir)
                        result.bundle_count += 1
                    else:
                        result.rejected_count += 1
                except Exception as e:
                    logger.error("silver_lane_failed", lane=lane, error=str(e))
                    result.rejected_count += 1

        return {"bundle_count": result.bundle_count, "rejected_count": result.rejected_count}

    def _load_gold_refs(self, gold_ref: Path) -> list[dict]:
        import json
        bundles = []
        for f in sorted(gold_ref.glob("GOLD-*.json")):
            bundles.append(json.loads(f.read_text()))
        return bundles

    def _get_applicable_lanes(self, gold_bundle: dict) -> list[LaneName]:
        """Determine which lanes apply based on source content."""
        # TODO: implement lane applicability logic
        return [LaneName.SIMULATION_JSON, LaneName.CURRICULUM_DIFFICULTY]

    def _generate_lane(self, gold_bundle: dict, lane: LaneName) -> Optional[SilverBundle]:
        """Generate silver data for a specific lane."""
        from brahmx_sdg.lanes.base_lane import get_lane_processor
        processor = get_lane_processor(lane)
        return processor.process(gold_bundle)

    def _write_bundle(self, bundle: SilverBundle, output_dir: Path) -> None:
        path = output_dir / f"{bundle.record_id}.json"
        path.write_text(bundle.model_dump_json(indent=2))
