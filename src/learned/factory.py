from __future__ import annotations

from dataclasses import dataclass

from dynamics.learned_bridge import LegacyHeuristicDynamicsTransitionModel, LearnedDynamicsTransitionModel
from learned.interfaces import (
    DynamicsTransitionModel,
    GraphEncoder,
    IdentityAppearanceEncoder,
    PatchSynthesisModel,
    TemporalConsistencyModel,
    TextEncoder,
)
from rendering.learned_bridge import LegacyDeterministicPatchSynthesisModel, TrainablePatchSynthesisModel
from rendering.temporal_bridge import LegacyBaselineTemporalConsistencyModel, TrainableTemporalConsistencyBackend
from representation.learned_bridge import BaselineGraphEncoder, BaselineIdentityAppearanceEncoder
from text.learned_bridge import BaselineTextEncoderAdapter


@dataclass(slots=True)
class BackendConfig:
    text_encoder: str = "baseline"
    graph_encoder: str = "baseline"
    identity_encoder: str = "baseline"
    dynamics_backend: str = "learned_graph_delta"
    patch_backend: str = "trainable_local"
    temporal_backend: str = "trainable_temporal"


@dataclass(slots=True)
class BackendBundle:
    text_encoder: TextEncoder
    graph_encoder: GraphEncoder
    identity_encoder: IdentityAppearanceEncoder
    dynamics_backend: DynamicsTransitionModel
    patch_backend: PatchSynthesisModel
    temporal_backend: TemporalConsistencyModel
    backend_names: dict[str, str]


class LearnedBackendFactory:
    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig()

    def build(self) -> BackendBundle:
        resolved_names = {
            "text_encoder": self._resolve_alias("text_encoder", self.config.text_encoder),
            "graph_encoder": self._resolve_alias("graph_encoder", self.config.graph_encoder),
            "identity_encoder": self._resolve_alias("identity_encoder", self.config.identity_encoder),
            "dynamics_backend": self._resolve_alias("dynamics_backend", self.config.dynamics_backend),
            "patch_backend": self._resolve_alias("patch_backend", self.config.patch_backend),
            "temporal_backend": self._resolve_alias("temporal_backend", self.config.temporal_backend),
        }
        return BackendBundle(
            text_encoder=self._build_text_encoder(resolved_names["text_encoder"]),
            graph_encoder=self._build_graph_encoder(resolved_names["graph_encoder"]),
            identity_encoder=self._build_identity_encoder(resolved_names["identity_encoder"]),
            dynamics_backend=self._build_dynamics(resolved_names["dynamics_backend"]),
            patch_backend=self._build_patch(resolved_names["patch_backend"]),
            temporal_backend=self._build_temporal(resolved_names["temporal_backend"]),
            backend_names={
                **resolved_names,
                "requested_dynamics_backend": self.config.dynamics_backend,
                "requested_patch_backend": self.config.patch_backend,
                "requested_temporal_backend": self.config.temporal_backend,
            },
        )

    @staticmethod
    def _resolve_alias(stage: str, name: str) -> str:
        normalized = str(name).strip()
        alias_map = {
            ("dynamics_backend", "baseline"): "legacy_heuristic",
            ("patch_backend", "baseline"): "legacy_deterministic",
            ("temporal_backend", "baseline"): "legacy_baseline",
        }
        return alias_map.get((stage, normalized), normalized)

    def _build_text_encoder(self, name: str) -> TextEncoder:
        if name == "baseline":
            return BaselineTextEncoderAdapter()
        raise ValueError(f"Unknown text encoder backend: {name}")

    def _build_graph_encoder(self, name: str) -> GraphEncoder:
        if name == "baseline":
            return BaselineGraphEncoder()
        raise ValueError(f"Unknown graph encoder backend: {name}")

    def _build_identity_encoder(self, name: str) -> IdentityAppearanceEncoder:
        if name == "baseline":
            return BaselineIdentityAppearanceEncoder()
        raise ValueError(f"Unknown identity encoder backend: {name}")

    def _build_dynamics(self, name: str) -> DynamicsTransitionModel:
        if name in {"learned_graph_delta", "learned_primary"}:
            return LearnedDynamicsTransitionModel()
        if name in {"legacy_heuristic", "legacy"}:
            return LegacyHeuristicDynamicsTransitionModel()
        raise ValueError(f"Unknown dynamics backend: {name}")

    def _build_patch(self, name: str) -> PatchSynthesisModel:
        if name in {"trainable_local", "learned_primary"}:
            return TrainablePatchSynthesisModel()
        if name in {"legacy", "legacy_deterministic"}:
            return LegacyDeterministicPatchSynthesisModel()
        raise ValueError(f"Unknown patch backend: {name}")

    def _build_temporal(self, name: str) -> TemporalConsistencyModel:
        if name in {"trainable_temporal", "learned_primary"}:
            return TrainableTemporalConsistencyBackend()
        if name in {"legacy", "legacy_baseline"}:
            return LegacyBaselineTemporalConsistencyModel()
        raise ValueError(f"Unknown temporal backend: {name}")
