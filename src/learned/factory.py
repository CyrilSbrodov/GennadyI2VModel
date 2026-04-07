from __future__ import annotations

from dataclasses import dataclass

from dynamics.learned_bridge import BaselineDynamicsTransitionModel
from learned.interfaces import (
    DynamicsTransitionModel,
    GraphEncoder,
    IdentityAppearanceEncoder,
    PatchSynthesisModel,
    TemporalConsistencyModel,
    TextEncoder,
)
from rendering.learned_bridge import BaselinePatchSynthesisModel
from rendering.temporal_bridge import BaselineTemporalConsistencyModel
from representation.learned_bridge import BaselineGraphEncoder, BaselineIdentityAppearanceEncoder
from text.learned_bridge import BaselineTextEncoderAdapter


@dataclass(slots=True)
class BackendConfig:
    text_encoder: str = "baseline"
    graph_encoder: str = "baseline"
    identity_encoder: str = "baseline"
    dynamics_backend: str = "baseline"
    patch_backend: str = "baseline"
    temporal_backend: str = "baseline"


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
        return BackendBundle(
            text_encoder=self._build_text_encoder(self.config.text_encoder),
            graph_encoder=self._build_graph_encoder(self.config.graph_encoder),
            identity_encoder=self._build_identity_encoder(self.config.identity_encoder),
            dynamics_backend=self._build_dynamics(self.config.dynamics_backend),
            patch_backend=self._build_patch(self.config.patch_backend),
            temporal_backend=self._build_temporal(self.config.temporal_backend),
            backend_names={
                "text_encoder": self.config.text_encoder,
                "graph_encoder": self.config.graph_encoder,
                "identity_encoder": self.config.identity_encoder,
                "dynamics_backend": self.config.dynamics_backend,
                "patch_backend": self.config.patch_backend,
                "temporal_backend": self.config.temporal_backend,
            },
        )

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
        if name == "baseline":
            return BaselineDynamicsTransitionModel()
        raise ValueError(f"Unknown dynamics backend: {name}")

    def _build_patch(self, name: str) -> PatchSynthesisModel:
        if name == "baseline":
            return BaselinePatchSynthesisModel()
        raise ValueError(f"Unknown patch backend: {name}")

    def _build_temporal(self, name: str) -> TemporalConsistencyModel:
        if name == "baseline":
            return BaselineTemporalConsistencyModel()
        raise ValueError(f"Unknown temporal backend: {name}")
