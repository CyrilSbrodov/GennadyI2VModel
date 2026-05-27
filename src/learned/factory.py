from __future__ import annotations

from dataclasses import dataclass

from runtime.modes import normalize_runtime_mode, runtime_forbids_fallbacks

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
    runtime_mode: str = "trainable_stub"
    patch_checkpoint_path: str = ""
    dynamics_checkpoint_path: str = ""
    temporal_checkpoint_path: str = ""
    patch_strict_mode: bool = False
    patch_strict_checkpoint: bool = True
    dynamics_strict_checkpoint: bool = False
    temporal_strict_checkpoint: bool = False


@dataclass(slots=True)
class BackendBundle:
    text_encoder: TextEncoder
    graph_encoder: GraphEncoder
    identity_encoder: IdentityAppearanceEncoder
    dynamics_backend: DynamicsTransitionModel
    patch_backend: PatchSynthesisModel
    temporal_backend: TemporalConsistencyModel
    backend_names: dict[str, object]


class LearnedBackendFactory:
    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig()

    def build(self) -> BackendBundle:
        mode = normalize_runtime_mode(self.config.runtime_mode)
        self._runtime_mode = mode
        resolved_names = {
            "text_encoder": self._resolve_alias("text_encoder", self.config.text_encoder),
            "graph_encoder": self._resolve_alias("graph_encoder", self.config.graph_encoder),
            "identity_encoder": self._resolve_alias("identity_encoder", self.config.identity_encoder),
            "dynamics_backend": self._resolve_alias("dynamics_backend", self.config.dynamics_backend),
            "patch_backend": self._resolve_alias("patch_backend", self.config.patch_backend),
            "temporal_backend": self._resolve_alias("temporal_backend", self.config.temporal_backend),
        }
        self._validate_runtime_backend_policy(resolved_names)
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
                "patch_checkpoint_requested": bool(self.config.patch_checkpoint_path),
                "patch_checkpoint_path": self.config.patch_checkpoint_path,
                "patch_strict_checkpoint": self.config.patch_strict_checkpoint,
                "patch_strict_mode": self.config.patch_strict_mode,
                "runtime_mode": mode,
                "fallback_forbidden": runtime_forbids_fallbacks(self._runtime_mode),
                "dynamics_checkpoint_path": self.config.dynamics_checkpoint_path,
                "temporal_checkpoint_path": self.config.temporal_checkpoint_path,
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
            return LearnedDynamicsTransitionModel(runtime_mode=self.config.runtime_mode, checkpoint_path=self.config.dynamics_checkpoint_path, strict_checkpoint=self.config.dynamics_strict_checkpoint or runtime_forbids_fallbacks(self._runtime_mode))
        if name in {"legacy_heuristic", "legacy"}:
            return LegacyHeuristicDynamicsTransitionModel()
        raise ValueError(f"Unknown dynamics backend: {name}")

    def _build_patch(self, name: str) -> PatchSynthesisModel:
        if name in {"trainable_local", "learned_primary"}:
            if runtime_forbids_fallbacks(self._runtime_mode) and not self.config.patch_checkpoint_path:
                raise RuntimeError("strict learned runtime requires patch checkpoint")
            if self.config.patch_checkpoint_path:
                return TrainablePatchSynthesisModel.from_checkpoint(
                    self.config.patch_checkpoint_path,
                    strict_mode=self.config.patch_strict_mode,
                    strict_checkpoint=self.config.patch_strict_checkpoint or runtime_forbids_fallbacks(self._runtime_mode),
                )
            return TrainablePatchSynthesisModel(strict_mode=self.config.patch_strict_mode or runtime_forbids_fallbacks(self._runtime_mode))
        if name in {"legacy", "legacy_deterministic"}:
            return LegacyDeterministicPatchSynthesisModel()
        raise ValueError(f"Unknown patch backend: {name}")

    def _build_temporal(self, name: str) -> TemporalConsistencyModel:
        if name in {"trainable_temporal", "learned_primary"}:
            return TrainableTemporalConsistencyBackend.from_checkpoint_policy(checkpoint_path=self.config.temporal_checkpoint_path, strict_checkpoint=self.config.temporal_strict_checkpoint or runtime_forbids_fallbacks(self._runtime_mode), strict_mode=runtime_forbids_fallbacks(self._runtime_mode))
        if name in {"legacy", "legacy_baseline"}:
            return LegacyBaselineTemporalConsistencyModel()
        raise ValueError(f"Unknown temporal backend: {name}")


    def _validate_runtime_backend_policy(self, resolved_names: dict[str, str]) -> None:
        if not runtime_forbids_fallbacks(self._runtime_mode):
            return
        if resolved_names.get("dynamics_backend") in {"legacy_heuristic", "legacy"}:
            raise RuntimeError("strict runtime forbids legacy dynamics backend")
        if resolved_names.get("dynamics_backend") in {"learned_graph_delta", "learned_primary"} and not str(self.config.dynamics_checkpoint_path or "").strip():
            raise RuntimeError("strict learned runtime requires dynamics checkpoint")
        if resolved_names.get("patch_backend") in {"legacy_deterministic", "legacy"}:
            raise RuntimeError("strict runtime forbids legacy patch backend")
        if resolved_names.get("temporal_backend") in {"legacy_baseline", "legacy"}:
            raise RuntimeError("strict runtime forbids legacy temporal backend")
