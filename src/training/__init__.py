from training.types import StageResult, TrainingConfig


def train_stage(*args, **kwargs):
    from training.orchestrator import train_stage as _train_stage

    return _train_stage(*args, **kwargs)


def train_pipeline(*args, **kwargs):
    from training.orchestrator import train_pipeline as _train_pipeline

    return _train_pipeline(*args, **kwargs)


__all__ = ["TrainingConfig", "StageResult", "train_stage", "train_pipeline"]
