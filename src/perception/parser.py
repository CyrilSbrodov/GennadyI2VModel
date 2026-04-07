from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.input_layer import AssetFrame
from perception.backend import frame_to_features
from perception.detector import BackendConfig, PersonDetection
from perception.image_ops import crop_rgb, frame_to_numpy_rgb
from perception.mask_store import DEFAULT_MASK_STORE


@dataclass(slots=True)
class ParserBackendConfig:
    backend: str = "builtin"
    model_id: str = ""
    device: str = "cpu"
    confidence_threshold: float = 0.25


@dataclass(slots=True)
class ParserStackConfig:
    person_segmentation: ParserBackendConfig = field(default_factory=ParserBackendConfig)
    body_parts: ParserBackendConfig = field(default_factory=ParserBackendConfig)
    garments: ParserBackendConfig = field(default_factory=ParserBackendConfig)
    face_regions: ParserBackendConfig = field(default_factory=ParserBackendConfig)

    def is_builtin(self) -> bool:
        return all(
            cfg.backend == "builtin"
            for cfg in (self.person_segmentation, self.body_parts, self.garments, self.face_regions)
        )

    @classmethod
    def from_backend_config(cls, config: BackendConfig) -> "ParserStackConfig":
        # Если передали legacy-config, применяем один backend ко всем подпарсерам.
        mapped = ParserBackendConfig(
            backend=config.backend,
            model_id=config.checkpoint,
            device=config.device,
            confidence_threshold=config.confidence_threshold,
        )
        return cls(person_segmentation=mapped, body_parts=mapped, garments=mapped, face_regions=mapped)


@dataclass(slots=True)
class GarmentPrediction:
    garment_type: str
    state: str
    confidence: float
    source: str


@dataclass(slots=True)
class BodyPartMaskPrediction:
    part_type: str
    mask_ref: str | None
    confidence: float
    visibility: str
    source: str


@dataclass(slots=True)
class FaceRegionPrediction:
    region_type: str
    mask_ref: str | None
    confidence: float
    source: str


@dataclass(slots=True)
class ParsingPrediction:
    mask_ref: str | None
    mask_confidence: float
    source: str
    garments: list[GarmentPrediction] = field(default_factory=list)
    occlusion_hints: list[str] = field(default_factory=list)
    body_parts: list[BodyPartMaskPrediction] = field(default_factory=list)
    face_regions: list[FaceRegionPrediction] = field(default_factory=list)


class HumanParser(Protocol):
    def parse(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        ...


def _mask_to_ref(mask: "object", confidence: float, source: str, prefix: str) -> str | None:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(mask)
        if arr.size == 0:
            return None
        return DEFAULT_MASK_STORE.put(arr.astype("uint8"), confidence=confidence, source=source, prefix=prefix)
    except Exception:
        return None


def _extract_binary_mask(segment: dict) -> "object | None":
    mask = segment.get("mask")
    if mask is None:
        return None
    import numpy as np  # type: ignore

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype("uint8")


def _canonical_garment_label(model_label: str) -> str | None:
    label = model_label.lower().strip()
    mapping = {
        "coat": "coat",
        "jacket": "jacket",
        "hoodie": "hoodie",
        "sweater": "sweater",
        "shirt": "shirt",
        "blouse": "top",
        "top": "top",
        "inner": "inner_upper",
        "pants": "pants",
        "trousers": "pants",
        "jeans": "pants",
        "skirt": "skirt",
        "dress": "dress",
        "shoes": "shoes",
        "shoe": "shoes",
        "scarf": "scarf",
        "hat": "hat",
        "cap": "hat",
        "hood": "hood",
    }
    return mapping.get(label)


def _canonical_body_part_label(model_label: str) -> str | None:
    mapping = {
        "head": "head",
        "hair": "hair",
        "face": "face",
        "neck": "neck",
        "torso": "torso",
        "upper_clothes": "torso",
        "left_arm": "left_upper_arm",
        "right_arm": "right_upper_arm",
        "left_lower_arm": "left_lower_arm",
        "right_lower_arm": "right_lower_arm",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "hip": "pelvis",
        "pelvis": "pelvis",
        "left_upper_leg": "left_upper_leg",
        "right_upper_leg": "right_upper_leg",
        "left_lower_leg": "left_lower_leg",
        "right_lower_leg": "right_lower_leg",
        "left_foot": "left_foot",
        "right_foot": "right_foot",
    }
    return mapping.get(model_label.lower().strip())


class _HFSegmentationRunner:
    def __init__(self, config: ParserBackendConfig) -> None:
        self.config = config
        self._pipe = None

    def run(self, image: "object") -> list[dict]:
        if self._pipe is None:
            try:
                from transformers import pipeline  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("transformers is not installed") from exc
            if not self.config.model_id:
                raise RuntimeError("human parsing model_id is required for hf backend")
            self._pipe = pipeline("image-segmentation", model=self.config.model_id)
        output = self._pipe(image)
        return output if isinstance(output, list) else []


class SegFormerHumanParserAdapter:
    """Композиция подпарсеров: person/body/garments/face + fusion."""

    source_name = "parser:human-stack"

    def __init__(self, config: ParserStackConfig | BackendConfig | None = None) -> None:
        if isinstance(config, BackendConfig):
            self.config = ParserStackConfig.from_backend_config(config)
        else:
            self.config = config or ParserStackConfig()
        self._body_runner = _HFSegmentationRunner(self.config.body_parts)
        self._garment_runner = _HFSegmentationRunner(self.config.garments)

    def is_builtin_backend(self) -> bool:
        return self.config.is_builtin()

    def _builtin_parse(self, frame: AssetFrame | list[list[list[float]]] | str, person: PersonDetection) -> ParsingPrediction:
        feats = frame_to_features(frame)
        coat_conf = min(0.95, max(0.1, feats[0]))
        shirt_conf = min(0.95, max(0.1, feats[1]))
        return ParsingPrediction(
            mask_ref=f"mask::builtin::{person.detection_id}",
            mask_confidence=min(0.99, max(0.2, feats[2])),
            source="parser:human-stack:builtin",
            garments=[
                GarmentPrediction("coat", "worn" if coat_conf > 0.5 else "removed", coat_conf, "parser:human-stack:builtin"),
                GarmentPrediction("shirt", "visible" if shirt_conf > 0.4 else "covered", shirt_conf, "parser:human-stack:builtin"),
            ],
            occlusion_hints=["torso_partial" if feats[3] > 0.6 else "torso_visible"],
        )

    def _person_mask(self, patch: "object") -> tuple["object | None", float, str]:
        cfg = self.config.person_segmentation
        if cfg.backend == "builtin":
            return None, 0.0, "parser:person:builtin"
        if cfg.backend == "mediapipe":
            try:
                import mediapipe as mp  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("mediapipe is not installed") from exc
            with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as model:
                result = model.process(patch)
                if result.segmentation_mask is None:
                    return None, 0.0, "parser:person:mediapipe"
                import numpy as np  # type: ignore

                mask = (np.asarray(result.segmentation_mask) > 0.45).astype("uint8")
                conf = float(mask.mean())
                return mask, conf, "parser:person:mediapipe"
        raise RuntimeError(f"unsupported person segmentation backend: {cfg.backend}")

    def _body_parts(self, patch: "object") -> list[BodyPartMaskPrediction]:
        cfg = self.config.body_parts
        if cfg.backend == "builtin":
            return []
        if cfg.backend == "hf":
            parts: list[BodyPartMaskPrediction] = []
            for seg in self._body_runner.run(patch):
                part = _canonical_body_part_label(str(seg.get("label", "")))
                if part is None:
                    continue
                conf = float(seg.get("score", 0.0))
                if conf < cfg.confidence_threshold:
                    continue
                mask = _extract_binary_mask(seg)
                mask_ref = _mask_to_ref(mask, conf, "parser:body:hf", "body") if mask is not None else None
                visibility = "visible" if conf >= 0.65 else "partially_visible"
                parts.append(BodyPartMaskPrediction(part, mask_ref, conf, visibility, "parser:body:hf"))
            return parts
        raise RuntimeError(f"unsupported body parser backend: {cfg.backend}")

    def _garments(self, patch: "object") -> list[GarmentPrediction]:
        cfg = self.config.garments
        if cfg.backend == "builtin":
            return []
        if cfg.backend == "hf":
            garments: list[GarmentPrediction] = []
            for seg in self._garment_runner.run(patch):
                raw_label = str(seg.get("label", ""))
                garment = _canonical_garment_label(raw_label)
                if garment is None:
                    continue
                conf = float(seg.get("score", 0.0))
                if conf < cfg.confidence_threshold:
                    continue
                garments.append(GarmentPrediction(garment, "visible", conf, "parser:garment:hf"))
            return garments
        raise RuntimeError(f"unsupported garment parser backend: {cfg.backend}")

    def _face_regions(self, patch: "object") -> list[FaceRegionPrediction]:
        cfg = self.config.face_regions
        if cfg.backend == "builtin":
            return []
        if cfg.backend == "mediapipe":
            try:
                import mediapipe as mp  # type: ignore
                import numpy as np  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("mediapipe is not installed") from exc

            regions: list[FaceRegionPrediction] = []
            h, w = patch.shape[:2]
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
                result = mesh.process(patch)
                if not result.multi_face_landmarks:
                    return []
                lms = result.multi_face_landmarks[0].landmark

                def _region(idx: list[int], name: str) -> FaceRegionPrediction:
                    mask = np.zeros((h, w), dtype="uint8")
                    points = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in idx], dtype="int32")
                    if len(points) >= 3:
                        try:
                            from PIL import Image, ImageDraw  # type: ignore

                            img = Image.fromarray(mask)
                            draw = ImageDraw.Draw(img)
                            draw.polygon([(int(p[0]), int(p[1])) for p in points], fill=1)
                            mask = np.asarray(img, dtype="uint8")
                        except Exception:
                            x1, y1 = points[:, 0].min(), points[:, 1].min()
                            x2, y2 = points[:, 0].max(), points[:, 1].max()
                            mask[max(0, y1) : min(h, y2 + 1), max(0, x1) : min(w, x2 + 1)] = 1
                    conf = float(mask.mean())
                    ref = _mask_to_ref(mask, conf, "parser:face:mediapipe", "face")
                    return FaceRegionPrediction(name, ref, conf, "parser:face:mediapipe")

                regions.append(_region([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361], "face"))
                regions.append(_region([33, 133, 159, 145, 362, 263, 386, 374], "eyes"))
                regions.append(_region([61, 291, 13, 14, 78, 308], "mouth"))
            return regions
        raise RuntimeError(f"unsupported face parser backend: {cfg.backend}")

    def _occlusion_hints(self, person_mask: "object | None", parts: list[BodyPartMaskPrediction], garments: list[GarmentPrediction]) -> list[str]:
        hints: list[str] = []
        if person_mask is not None:
            import numpy as np  # type: ignore

            density = float(np.asarray(person_mask).mean())
            if density < 0.12:
                hints.append("person_mask_fragmented")
            elif density < 0.2:
                hints.append("person_mask_uncertain")
        part_map = {p.part_type: p for p in parts}
        torso = part_map.get("torso")
        if torso is None:
            hints.append("torso_hidden")
        elif torso.confidence < 0.5:
            hints.append("torso_partially_occluded")
        else:
            hints.append("torso_visible")

        arm_parts = [part_map.get("left_upper_arm"), part_map.get("right_upper_arm")]
        arm_visible = [p for p in arm_parts if p and p.confidence >= 0.5]
        hints.append("arms_visible" if len(arm_visible) == 2 else "arms_partially_occluded")

        lower_parts = [part_map.get("left_upper_leg"), part_map.get("right_upper_leg"), part_map.get("pelvis")]
        lower_visible = [p for p in lower_parts if p and p.confidence >= 0.5]
        hints.append("lower_body_visible" if len(lower_visible) >= 2 else "lower_body_partially_occluded")

        garment_types = {g.garment_type for g in garments}
        if garment_types & {"coat", "jacket", "hoodie"}:
            hints.append("outer_garment_covers_torso")
            if arm_visible:
                hints.append("outer_garment_covers_arms")
        return hints

    def parse(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        result: dict[str, ParsingPrediction] = {}
        use_real_path = not self.config.is_builtin()
        rgb = frame_to_numpy_rgb(frame).rgb if use_real_path else None
        for person in persons:
            if self.config.is_builtin():
                result[person.detection_id] = self._builtin_parse(frame, person)
                continue

            if rgb is None:
                raise RuntimeError("real parser backend requires numpy image conversion")
            patch = crop_rgb(rgb, person.bbox)
            person_mask, person_conf, person_source = self._person_mask(patch)
            parts = self._body_parts(patch)
            garments = self._garments(patch)
            face_regions = self._face_regions(patch)

            mask_ref = _mask_to_ref(person_mask, person_conf, person_source, "person") if person_mask is not None else None
            combined_conf = max(
                person_conf,
                max([p.confidence for p in parts], default=0.0),
                max([g.confidence for g in garments], default=0.0),
                max([f.confidence for f in face_regions], default=0.0),
            )
            occlusion_hints = self._occlusion_hints(person_mask, parts, garments)
            result[person.detection_id] = ParsingPrediction(
                mask_ref=mask_ref,
                mask_confidence=combined_conf,
                source="parser:human-stack:real",
                garments=garments,
                occlusion_hints=occlusion_hints,
                body_parts=parts,
                face_regions=face_regions,
            )
        return result
