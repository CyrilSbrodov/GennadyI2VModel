from __future__ import annotations

from learned.interfaces import TemporalConsistencyModel, TemporalRefinementOutput, TemporalRefinementRequest
from rendering.compositor import TemporalStabilizer


class BaselineTemporalConsistencyModel(TemporalConsistencyModel):
    def __init__(self, stabilizer: TemporalStabilizer | None = None) -> None:
        self.stabilizer = stabilizer or TemporalStabilizer()

    def refine_temporal(self, request: TemporalRefinementRequest) -> TemporalRefinementOutput:
        changed = request.changed_regions
        region_conf = 0.7 if not changed else min(0.95, 0.5 + 0.05 * len(changed))
        refined = self.stabilizer.refine(
            previous_frame=request.previous_frame,
            new_frame=request.current_composed_frame,
            memory=request.memory_state,
            enabled=True,
            updated_regions=changed,
            region_confidence=region_conf,
        )
        scores = {r.region_id: max(0.0, min(1.0, region_conf - 0.02 * idx)) for idx, r in enumerate(changed)}
        return TemporalRefinementOutput(refined_frame=refined, region_consistency_scores=scores, metadata={"backend": "deterministic_temporal_stabilizer"})
