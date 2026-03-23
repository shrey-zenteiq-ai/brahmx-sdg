"""
Abstract Base Lane — Interface for all specialized scientific generation lanes.

Each lane transforms a scientific source into lane-specific training artifacts.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from brahmx_sdg.schemas import SilverBundle, LaneName


@dataclass
class LaneConfig:
    lane_name: LaneName
    enabled: bool = True
    max_outputs_per_source: int = 10
    validator_strict: bool = True
    extra: dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class BaseLane(ABC):
    """Abstract lane processor. All scientific lanes implement this interface."""

    def __init__(self, config: Optional[LaneConfig] = None):
        self.config = config or LaneConfig(lane_name=self.lane_name)

    @property
    @abstractmethod
    def lane_name(self) -> LaneName:
        ...

    @abstractmethod
    def process(self, source: dict) -> Optional[SilverBundle]:
        """Process a single source into a silver bundle. Returns None on failure."""
        ...

    @abstractmethod
    def validate(self, output: dict) -> bool:
        """Run lane-specific validation on generated output."""
        ...

    def emit_products(self, bundle: SilverBundle) -> list[dict]:
        """Emit derived data products (pretraining, SFT, QA, etc.)."""
        return []


def get_lane_processor(lane: LaneName) -> BaseLane:
    """Factory: return the processor for a given lane."""
    from brahmx_sdg.lanes.simulation_json_lane import SimulationJSONLane
    from brahmx_sdg.lanes.dialogue_lane import DialogueLane
    from brahmx_sdg.lanes.latex_lane import LaTeXLane
    from brahmx_sdg.lanes.code_tool_lane import CodeToolLane
    from brahmx_sdg.lanes.multilingual_lane import MultilingualLane
    from brahmx_sdg.lanes.reasoning_budget_lane import ReasoningBudgetLane
    from brahmx_sdg.lanes.curriculum_lane import CurriculumLane

    _registry = {
        LaneName.SIMULATION_JSON: SimulationJSONLane,
        LaneName.BOUNDED_DIALOGUE: DialogueLane,
        LaneName.LATEX_ARTIFACT: LaTeXLane,
        LaneName.CODE_TOOL_USE: CodeToolLane,
        LaneName.MULTILINGUAL: MultilingualLane,
        LaneName.REASONING_BUDGET: ReasoningBudgetLane,
        LaneName.CURRICULUM_DIFFICULTY: CurriculumLane,
    }
    cls = _registry.get(lane)
    if cls is None:
        raise ValueError(f"Unknown lane: {lane}")
    return cls()
