from __future__ import annotations

import json
from pathlib import Path
from rendering.trainable_patch_renderer import TrainableLocalPatchModel


def load_renderer_model_from_checkpoint(checkpoint_path: str) -> tuple[TrainableLocalPatchModel | object, str, dict[str, object]]:
    """Load a renderer model using the backend recorded in checkpoint metadata.

    The torch backend import is intentionally deferred until checkpoint metadata
    requests it, so numpy checkpoints do not require torch at runtime.
    """

    checkpoint = Path(checkpoint_path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    metadata = payload.get("renderer_model_metadata", {}) if isinstance(payload.get("renderer_model_metadata", {}), dict) else {}
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
