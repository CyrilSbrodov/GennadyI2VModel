from __future__ import annotations

import json
from pathlib import Path
from rendering.patch_conditioning_contract import GLOBAL_COND_DIM
from rendering.trainable_patch_renderer import TrainableLocalPatchModel

RENDERER_CHECKPOINT_CONTRACT_VERSION = "renderer_checkpoint.v1"


def load_renderer_model_from_checkpoint(checkpoint_path: str) -> tuple[TrainableLocalPatchModel | object, str, dict[str, object]]:
    """Load a renderer model using the backend recorded in checkpoint metadata.

    The torch backend import is intentionally deferred until checkpoint metadata
    requests it, so numpy checkpoints do not require torch at runtime.
    """

    checkpoint = Path(checkpoint_path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    metadata = payload.get("renderer_model_metadata", {}) if isinstance(payload.get("renderer_model_metadata", {}), dict) else {}
    metadata_contract = metadata.get("checkpoint_contract_version")
    if metadata_contract is not None and str(metadata_contract) != RENDERER_CHECKPOINT_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported renderer checkpoint metadata contract version: "
            f"{metadata_contract!r}; expected {RENDERER_CHECKPOINT_CONTRACT_VERSION!r}"
        )
    if metadata.get("runtime_loadable") is False or payload.get("runtime_loadable") is False:
        raise ValueError("Renderer patch checkpoint metadata marks runtime_loadable=False")
    if "global_cond_dim" in metadata and metadata.get("global_cond_dim") is not None:
        try:
            checkpoint_global_cond_dim = int(metadata.get("global_cond_dim"))
        except (TypeError, ValueError) as err:
            raise ValueError("Renderer patch checkpoint metadata global_cond_dim must be an integer") from err
        if checkpoint_global_cond_dim != GLOBAL_COND_DIM:
            raise ValueError(
                "Renderer patch checkpoint global_cond_dim mismatch: "
                f"checkpoint={checkpoint_global_cond_dim}, runtime={GLOBAL_COND_DIM}"
            )
    backend = str(metadata.get("renderer_backend", payload.get("renderer_backend", "numpy_local")))
    model_path_value = str(payload.get("model_path", ""))
    if not model_path_value:
        raise ValueError("Renderer patch checkpoint missing model_path")

    model_path = Path(model_path_value)
    if not model_path.is_absolute():
        model_path = checkpoint.parent / model_path

    if backend == "torch_local":
        from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

        return TorchLocalPatchGenerator.load(str(model_path)), backend, dict(metadata)
    if backend in {"numpy_local", "legacy_local_renderer", "temporal_local_renderer"} or not backend:
        return TrainableLocalPatchModel.load(str(model_path)), "numpy_local", dict(metadata)
    raise ValueError(f"Unsupported renderer backend in patch checkpoint metadata: {backend}")
