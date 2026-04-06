from core.input_layer import InputAssetLayer
from perception.detector import BackendConfig, YoloPersonDetectorAdapter
from perception.pipeline import PerceptionPipeline


def test_perception_smoke_on_realistic_inputs() -> None:
    pipe = PerceptionPipeline()
    outs = pipe.analyze_video(["video.mp4#t=0.0", "video.mp4#t=0.1", "video.mp4#t=0.2"], batch_size=2)

    assert len(outs) == 3
    assert all(o.persons for o in outs)
    assert all(o.persons[0].track_id for o in outs)
    assert len({o.persons[0].track_id for o in outs}) == 1
    assert all(len(o.persons[0].face_landmarks) >= 68 for o in outs)
    assert all("left" in o.persons[0].hand_landmarks for o in outs)


def test_checkpoint_validation_error_is_clear() -> None:
    detector = YoloPersonDetectorAdapter(
        BackendConfig(backend="torch", checkpoint="missing/path/model.pt")
    )
    try:
        detector.detect("frame://x")
        assert False, "must raise"
    except FileNotFoundError as exc:
        assert "checkpoint is missing" in str(exc)


def test_input_layer_metadata_for_video() -> None:
    layer = InputAssetLayer()
    req = layer.build_request(images=[], video="clip.mp4", text="test", fps=10, duration=2.4)

    assert req.input_type == "video"
    assert req.frame_count >= 1
    assert req.orig_size is not None
    assert req.normalized_size is not None
    assert req.timestamps and req.timestamps[0] == 0.0
    assert req.reference_set
