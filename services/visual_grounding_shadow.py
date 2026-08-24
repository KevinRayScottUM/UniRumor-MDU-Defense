"""Shadow-only wrapper for deterministic visual-grounding artifacts."""

from typing import Any, List

from adapters.visual_grounding_adapter import VisualGroundingAdapter
from schemas import GroundedVisualUnit, VisualObservationSnapshot


VISUAL_GROUNDING_SHADOW_FAILURE_WARNING = (
    "visual grounding shadow generation failed; verification was unaffected"
)


class VisualGroundingShadowRunner:
    """Run the metadata-only grounding adapter outside the prediction path."""

    def __init__(self, adapter: Any = None) -> None:
        if adapter is None:
            adapter = VisualGroundingAdapter()
        if not callable(getattr(adapter, "ground", None)):
            raise TypeError("adapter must provide a callable ground method")
        self.adapter = adapter

    def run(
        self, visual_snapshots: List[VisualObservationSnapshot]
    ) -> List[GroundedVisualUnit]:
        if not isinstance(visual_snapshots, list) or not all(
            isinstance(snapshot, VisualObservationSnapshot)
            for snapshot in visual_snapshots
        ):
            raise TypeError(
                "visual grounding shadow runner requires isolated visual snapshots"
            )
        grounded_units = self.adapter.ground(visual_snapshots)
        if not isinstance(grounded_units, list):
            raise TypeError("visual grounding adapter must return a list")
        if not all(isinstance(unit, GroundedVisualUnit) for unit in grounded_units):
            raise TypeError(
                "visual grounding adapter must return only GroundedVisualUnit artifacts"
            )
        return grounded_units
