from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from core.input_layer import AssetFrame
from perception.backend import frame_to_features
from perception.detector import BackendConfig, PersonDetection
from perception.frame_context import FrameLike, PerceptionFrameContext
from perception.image_ops import crop_rgb, frame_to_numpy_rgb
from perception.mask_store import DEFAULT_MASK_STORE


@dataclass(slots=True)
class ParserBackendConfig:
    backend: str = "builtin"
    variant: str = ""
    device: str = "cpu"
    confidence_threshold: float = 0.25


@dataclass(slots=True)
class ParserStackConfig:
    # Явный и семантически честный stack без generic model_id-магии.
    primary_human_parser: ParserBackendConfig = field(default_factory=lambda: ParserBackendConfig(backend="builtin"))
    structural_body_parser: ParserBackendConfig = field(default_factory=lambda: ParserBackendConfig(backend="builtin"))
    garment_refinement_parser: ParserBackendConfig = field(default_factory=lambda: ParserBackendConfig(backend="builtin"))
    face_parser: ParserBackendConfig = field(default_factory=lambda: ParserBackendConfig(backend="builtin"))

    def is_builtin(self) -> bool:
        return all(
            cfg.backend == "builtin"
            for cfg in (
                self.primary_human_parser,
                self.structural_body_parser,
                self.garment_refinement_parser,
                self.face_parser,
            )
        )

    @classmethod
    def from_backend_config(cls, config: BackendConfig) -> "ParserStackConfig":
        # Legacy-режим: один backend размножаем по всем parser-компонентам.
        mapped = ParserBackendConfig(
            backend=("fashn" if config.backend == "hf" else config.backend),
            variant=config.checkpoint,
            device=config.device,
            confidence_threshold=config.confidence_threshold,
        )
        return cls(
            primary_human_parser=mapped,
            structural_body_parser=mapped,
            garment_refinement_parser=mapped,
            face_parser=mapped,
        )


@dataclass(slots=True)
class GarmentPrediction:
    garment_type: str
    state: str
    confidence: float
    source: str
    mask_ref: str | None = None
    coverage_targets: list[str] = field(default_factory=list)
    attachment_targets: list[str] = field(default_factory=list)
    layer_hint: str = "unknown"


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
class EnrichedParsingPayload:
    person_mask_ref: str | None = None
    body_part_masks: dict[str, str] = field(default_factory=dict)
    garment_masks: dict[str, str] = field(default_factory=dict)
    face_region_masks: dict[str, str] = field(default_factory=dict)
    accessory_masks: dict[str, str] = field(default_factory=dict)
    coverage_hints: dict[str, list[str]] = field(default_factory=dict)
    visibility_hints: dict[str, str] = field(default_factory=dict)
    provenance_by_region: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ParsingPrediction:
    mask_ref: str | None
    mask_confidence: float
    source: str
    garments: list[GarmentPrediction] = field(default_factory=list)
    occlusion_hints: list[str] = field(default_factory=list)
    body_parts: list[BodyPartMaskPrediction] = field(default_factory=list)
    face_regions: list[FaceRegionPrediction] = field(default_factory=list)
    enriched: EnrichedParsingPayload = field(default_factory=EnrichedParsingPayload)


class HumanParser(Protocol):
    def parse(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        ...


@dataclass(slots=True)
class AdapterSegmentationOutput:
    masks: dict[str, "object"] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)
    runtime_format: str = "unknown"


class ParserAdapter(Protocol):
    source_name: str

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        ...


def _safe_array(mask: "object"):
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if not isinstance(mask, list):
        return [[0]]
    if mask and isinstance(mask[0], list) and mask[0] and isinstance(mask[0][0], list):
        return [[int(px[0] > 0) for px in row] for row in mask]
    if mask and isinstance(mask[0], list):
        return [[int(v > 0) for v in row] for row in mask]
    return [list(int(v > 0) for v in mask)]


def _label_map_to_masks(label_map: "object", class_map: dict[int, str]) -> dict[str, "object"]:
    arr = label_map.tolist() if hasattr(label_map, "tolist") else label_map
    if not isinstance(arr, list):
        return {}
    out: dict[str, object] = {}
    for lid, name in class_map.items():
        if lid == 0:
            continue
        out[name] = [[1 if int(px) == lid else 0 for px in row] for row in arr]
    return out


def _mask_conf(mask: "object") -> float:
    arr = _safe_array(mask)
    total = sum(len(row) for row in arr)
    if total == 0:
        return 0.0
    ones = sum(sum(int(v > 0) for v in row) for row in arr)
    return float(max(0.0, min(1.0, ones / total)))


def _store_mask(
    mask: "object",
    confidence: float,
    source: str,
    prefix: str,
    kind: str,
    backend: str,
    *,
    roi_bbox: tuple[float, float, float, float] | None = None,
    frame_size: tuple[int, int] | None = None,
    person_id: str | None = None,
) -> str | None:
    try:
        arr = _safe_array(mask)
        if not arr or max((max(row) if row else 0 for row in arr), default=0) <= 0:
            return None
        return DEFAULT_MASK_STORE.put(
            arr,
            confidence=confidence,
            source=source,
            prefix=prefix,
            mask_kind=kind,
            backend=backend,
            roi_bbox=roi_bbox,
            frame_size=frame_size,
            tags=[f"person:{person_id}"] if person_id else [],
        )
    except Exception:
        return None


class _HFSegmentationBackend:
    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self._pipe = None
        self._id2label: dict[int, str] = {}

    def _to_pil_image(self, patch: object) -> Image.Image:
        if isinstance(patch, Image.Image):
            return patch

        arr = patch
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        if arr.size == 0:
            raise ValueError("empty patch passed to HF segmentation backend")

        if arr.dtype != np.uint8:
            if arr.dtype.kind in {"f"}:
                arr = np.clip(arr, 0.0, 255.0)
                if float(arr.max()) <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return Image.fromarray(arr[:, :, 0], mode="L")
            if arr.shape[2] >= 3:
                return Image.fromarray(arr[:, :, :3], mode="RGB")

        raise ValueError(f"unsupported patch shape for HF segmentation backend: {arr.shape}")

    def infer(self, patch: "object") -> object:
        if self._pipe is None:
            try:
                from transformers import pipeline  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("transformers is not installed for parser backend") from exc
            dev = -1 if self.device in {"cpu", "-1"} else 0
            self._pipe = pipeline("image-segmentation", model=self.model_id, device=dev)
            model = getattr(self._pipe, "model", None)
            config = getattr(model, "config", None)
            id2label = getattr(config, "id2label", None)
            if isinstance(id2label, dict):
                self._id2label = {int(k): str(v) for k, v in id2label.items()}

        pil_image = self._to_pil_image(patch)
        out = self._pipe(pil_image)
        return out

    @property
    def id2label(self) -> dict[int, str]:
        return dict(self._id2label)


class FashnHumanParserAdapter:
    source_name = "parser:fashn"
    CLASS_MAP = {
        0: "background",
        1: "face",
        2: "hair",
        3: "top",
        4: "dress",
        5: "skirt",
        6: "pants",
        7: "belt",
        8: "bag",
        9: "hat",
        10: "scarf",
        11: "glasses",
        12: "arms",
        13: "hands",
        14: "legs",
        15: "feet",
        16: "torso",
        17: "jewelry",
    }

    def __init__(self, config: ParserBackendConfig, infer_fn: Callable[["object"], "object"] | None = None) -> None:
        self.config = config
        self._infer_fn = infer_fn
        self._runtime = None

    def _infer(self, patch: "object"):
        if self._infer_fn is not None:
            return self._infer_fn(patch), {}, "test_override"
        if self.config.backend == "builtin":
            return None, {}, "builtin"
        model_id = self.config.variant or "fashn-ai/fashn-human-parser"
        if self._runtime is None:
            self._runtime = _HFSegmentationBackend(model_id=model_id, device=self.config.device)
        return self._runtime.infer(patch), self._runtime.id2label, "hf_image_segmentation"

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput(runtime_format="builtin")

        raw, id2label, runtime_format = self._infer(patch)
        if isinstance(raw, dict) and "label_map" in raw:
            class_map = {int(k): v.lower().strip() for k, v in id2label.items()} if id2label else self.CLASS_MAP
            masks = _label_map_to_masks(raw["label_map"], class_map)
            return AdapterSegmentationOutput(
                masks=masks,
                confidences={k: _mask_conf(v) for k, v in masks.items()},
                runtime_format=runtime_format + ":label_map",
            )

        masks: dict[str, object] = {}
        if isinstance(raw, list):
            for seg in raw:
                if not isinstance(seg, dict):
                    continue
                label = str(seg.get("label", "")).lower().strip()
                mask = seg.get("mask")
                if mask is not None and label and label != "background":
                    masks[label] = _safe_array(mask)

        return AdapterSegmentationOutput(
            masks=masks,
            confidences={k: _mask_conf(v) for k, v in masks.items()},
            runtime_format=runtime_format + ":segments_list",
        )


class SCHPPascalPartParserAdapter:
    source_name = "parser:schp_pascal"
    LABEL_TO_CANONICAL = {
        "head": "head",
        "torso": "torso",
        "upper arms": "upper_arm",
        "lower arms": "lower_arm",
        "upper legs": "upper_leg",
        "lower legs": "lower_leg",
    }

    def __init__(self, config: ParserBackendConfig, infer_fn: Callable[["object"], dict[str, "object"]] | None = None) -> None:
        self.config = config
        self._infer_fn = infer_fn
        self._runtime = None

    def _infer(self, patch: "object") -> tuple[object, str]:
        if self._infer_fn is not None:
            return self._infer_fn(patch), "test_override"
        if self.config.backend == "builtin":
            return {}, "builtin"
        if self.config.backend == "generic_hf_structural":
            if self._runtime is None:
                model_id = self.config.variant
                if not model_id:
                    raise RuntimeError("generic_hf_structural requires checkpoint/model id in ParserBackendConfig.variant")
                self._runtime = _HFSegmentationBackend(model_id=model_id, device=self.config.device)
            return self._runtime.infer(patch), "hf_image_segmentation"
        if self.config.backend == "schp_pascal":
            raise RuntimeError(
                "schp_pascal backend requires SCHP-specific runtime integration; "
                "use backend='generic_hf_structural' for generic HF segmentation models"
            )
        raise RuntimeError(f"Unsupported SCHP Pascal backend: {self.config.backend}")

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput(runtime_format="builtin")
        raw, runtime_format = self._infer(patch)
        raw_masks: dict[str, object] = {}
        if isinstance(raw, list):
            for seg in raw:
                if isinstance(seg, dict) and seg.get("mask") is not None:
                    raw_masks[str(seg.get("label", ""))] = seg["mask"]
        elif isinstance(raw, dict):
            raw_masks = {str(k): v for k, v in raw.items() if k != "label_map"}
        masks: dict[str, object] = {}
        for src_label, mask in raw_masks.items():
            canonical = self.LABEL_TO_CANONICAL.get(src_label.lower().strip())
            if canonical:
                masks[canonical] = _safe_array(mask)
        confs = {name: _mask_conf(mask) for name, mask in masks.items()}
        return AdapterSegmentationOutput(masks=masks, confidences=confs, runtime_format=runtime_format)


class SCHPATRParserAdapter:
    source_name = "parser:schp_atr"
    LABEL_TO_CANONICAL = {
        "upper-clothes": "upper_clothes",
        "skirt": "skirt",
        "pants": "pants",
        "dress": "dress",
        "belt": "belt",
        "bag": "bag",
        "scarf": "scarf",
        "face": "face",
        "left-arm": "arm",
        "right-arm": "arm",
        "left-leg": "leg",
        "right-leg": "leg",
    }

    def __init__(self, config: ParserBackendConfig, infer_fn: Callable[["object"], dict[str, "object"]] | None = None) -> None:
        self.config = config
        self._infer_fn = infer_fn
        self._runtime = None

    def _infer(self, patch: "object") -> tuple[object, str]:
        if self._infer_fn is not None:
            return self._infer_fn(patch), "test_override"
        if self.config.backend == "builtin":
            return {}, "builtin"
        if self.config.backend == "generic_hf_garment":
            if self._runtime is None:
                model_id = self.config.variant
                if not model_id:
                    raise RuntimeError("generic_hf_garment requires checkpoint/model id in ParserBackendConfig.variant")
                self._runtime = _HFSegmentationBackend(model_id=model_id, device=self.config.device)
            return self._runtime.infer(patch), "hf_image_segmentation"
        if self.config.backend == "schp_atr":
            raise RuntimeError(
                "schp_atr backend requires SCHP-specific runtime integration; "
                "use backend='generic_hf_garment' for generic HF segmentation models"
            )
        raise RuntimeError(f"Unsupported SCHP ATR backend: {self.config.backend}")

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput(runtime_format="builtin")
        raw, runtime_format = self._infer(patch)
        raw_masks: dict[str, object] = {}
        if isinstance(raw, list):
            for seg in raw:
                if isinstance(seg, dict) and seg.get("mask") is not None:
                    raw_masks[str(seg.get("label", ""))] = seg["mask"]
        elif isinstance(raw, dict):
            raw_masks = {str(k): v for k, v in raw.items() if k != "label_map"}
        masks: dict[str, object] = {}
        for src_label, mask in raw_masks.items():
            canonical = self.LABEL_TO_CANONICAL.get(src_label.lower().strip())
            if canonical:
                out_name = canonical if canonical not in {"arm", "leg"} else f"{canonical}_region"
                masks[out_name] = _safe_array(mask)
        confs = {name: _mask_conf(mask) for name, mask in masks.items()}
        return AdapterSegmentationOutput(masks=masks, confidences=confs, runtime_format=runtime_format)


class FacerFaceParserAdapter:
    source_name = "parser:facer"
    LABEL_TO_CANONICAL = {
        "face": "face_skin",
        "skin": "face_skin",
        "hair": "hairline_or_hair_face_boundary",
        "eyes": "eyes",
        "left-eye": "eyes",
        "right-eye": "eyes",
        "brow": "brows",
        "left-brow": "brows",
        "right-brow": "brows",
        "nose": "nose",
        "mouth": "mouth",
        "upper-lip": "upper_lip",
        "lower-lip": "lower_lip",
    }

    def __init__(self, config: ParserBackendConfig, infer_fn: Callable[["object"], dict[str, "object"]] | None = None) -> None:
        self.config = config
        self._infer_fn = infer_fn
        self._runtime = None

    def _infer(self, patch: "object") -> tuple[dict[str, object], str]:
        if self._infer_fn is not None:
            return self._infer_fn(patch), "test_override"
        if self.config.backend == "builtin":
            return {}, "builtin"
        try:
            import facer  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("FACER backend requires facer+torch packages") from exc
        if self._runtime is None:
            parser_variant = self.config.variant or "farl/lapa/448"
            self._runtime = (
                facer.face_detector("retinaface/mobilenet", device=self.config.device),
                facer.face_parser(parser_variant, device=self.config.device),
                torch,
            )
        detector, parser, torch_mod = self._runtime
        img = patch
        if hasattr(img, "shape"):
            shape = getattr(img, "shape", ())
            if len(shape) == 3:
                img = [img]
        tensor = torch_mod.tensor(img, device=self.config.device)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
        faces = detector(tensor)
        if int(getattr(faces.get("scores", []), "numel", lambda: 0)()) == 0:
            return {}, "facer:no_faces"
        faces = parser(tensor, faces)
        seg_logits = faces.get("seg", {}).get("logits")
        label_names = faces.get("seg", {}).get("label_names", [])
        if seg_logits is None:
            return {}, "facer:no_seg"
        seg_map = seg_logits.argmax(dim=1)[0].detach().cpu().numpy()
        out: dict[str, object] = {}
        for idx, name in enumerate(label_names):
            if idx == 0:
                continue
            out[str(name)] = [[1 if int(px) == idx else 0 for px in row] for row in seg_map.tolist()]
        return out, "facer:runtime"

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput(runtime_format="builtin")
        raw_masks, runtime_format = self._infer(patch)
        masks: dict[str, object] = {}
        for src_label, mask in raw_masks.items():
            canonical = self.LABEL_TO_CANONICAL.get(src_label.lower().strip())
            if canonical:
                masks[canonical] = _safe_array(mask)
        confs = {name: _mask_conf(mask) for name, mask in masks.items()}
        return AdapterSegmentationOutput(masks=masks, confidences=confs, runtime_format=runtime_format)


class ParserFusionEngine:
    GARMENT_PRIORITY = ("fashn", "schp_atr", "builtin")

    def fuse(
        self,
        person: PersonDetection,
        fashn: AdapterSegmentationOutput,
        pascal: AdapterSegmentationOutput,
        atr: AdapterSegmentationOutput,
        facer: AdapterSegmentationOutput,
        frame_size: tuple[int, int] | None = None,
    ) -> ParsingPrediction:
        garments: list[GarmentPrediction] = []
        body_parts: list[BodyPartMaskPrediction] = []
        face_regions: list[FaceRegionPrediction] = []
        enriched = EnrichedParsingPayload()

        person_union_parts = [mask for _, mask in fashn.masks.items()]
        if not person_union_parts:
            person_union_parts = list(pascal.masks.values()) + list(atr.masks.values())
        if person_union_parts:
            matrices = [_safe_array(x) for x in person_union_parts]
            h = max((len(m) for m in matrices), default=0)
            w = max((len(m[0]) if m else 0 for m in matrices), default=0)
            union_mask = [[0 for _ in range(w)] for _ in range(h)]
            for mat in matrices:
                for y, row in enumerate(mat):
                    for x, v in enumerate(row):
                        if v > 0:
                            union_mask[y][x] = 1
            pref = _store_mask(
                union_mask,
                _mask_conf(union_mask),
                "parser:fusion",
                "person",
                "person_mask",
                "fusion",
                roi_bbox=(person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h),
                frame_size=frame_size,
                person_id=person.detection_id,
            )
            enriched.person_mask_ref = pref

        topology = dict(fashn.masks)
        topology.update(pascal.masks)
        for part, mask in topology.items():
            if part in {"top", "dress", "skirt", "pants", "belt", "bag", "hat", "scarf", "glasses", "jewelry"}:
                continue
            conf = pascal.confidences.get(part, fashn.confidences.get(part, 0.0))
            source = "parser:schp_pascal" if part in pascal.masks else "parser:fashn"
            ref = _store_mask(
                mask,
                conf,
                source,
                "body",
                "body_part_mask",
                source.split(":")[-1],
                roi_bbox=(person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h),
                frame_size=frame_size,
                person_id=person.detection_id,
            )
            if ref is None:
                continue
            body_parts.append(BodyPartMaskPrediction(part, ref, conf, "visible" if conf >= 0.5 else "partially_visible", source))
            enriched.body_part_masks[part] = ref
            enriched.visibility_hints[f"body:{part}"] = "visible" if conf >= 0.5 else "partially_visible"
            enriched.provenance_by_region[f"body:{part}"] = source

        garment_candidates = {
            "top": "upper_torso",
            "upper_clothes": "upper_torso",
            "dress": "torso_pelvis_upper_legs",
            "skirt": "pelvis_upper_legs",
            "pants": "pelvis_legs",
            "belt": "pelvis_waist",
            "bag": "torso_side",
            "scarf": "neck_upper_torso",
            "hat": "head",
        }
        seen: set[str] = set()
        for source_name, out in (("fashn", fashn), ("schp_atr", atr)):
            for g, target in garment_candidates.items():
                if g not in out.masks or g in seen:
                    continue
                seen.add(g)
                conf = out.confidences.get(g, 0.0)
                src = f"parser:{source_name}"
                ref = _store_mask(
                    out.masks[g],
                    conf,
                    src,
                    "garment",
                    "garment_mask",
                    source_name,
                    roi_bbox=(person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h),
                    frame_size=frame_size,
                    person_id=person.detection_id,
                )
                garments.append(
                    GarmentPrediction(
                        garment_type=("top" if g == "upper_clothes" else g),
                        state="visible",
                        confidence=conf,
                        source=src,
                        mask_ref=ref,
                        coverage_targets=[target],
                        attachment_targets=["torso" if g != "hat" else "head"],
                        layer_hint=("outerwear" if g in {"coat", "jacket", "scarf"} else "innerwear"),
                    )
                )
                if ref:
                    enriched.garment_masks[g] = ref
                    enriched.coverage_hints[g] = [target]
                    enriched.visibility_hints[f"garment:{g}"] = "visible" if conf >= 0.5 else "partially_visible"
                    enriched.provenance_by_region[f"garment:{g}"] = src

        for acc in ("bag", "hat", "glasses", "jewelry", "scarf", "belt"):
            if acc in fashn.masks:
                conf = fashn.confidences.get(acc, 0.0)
                ref = _store_mask(
                    fashn.masks[acc],
                    conf,
                    "parser:fashn",
                    "accessory",
                    "accessory_mask",
                    "fashn",
                    roi_bbox=(person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h),
                    frame_size=frame_size,
                    person_id=person.detection_id,
                )
                if ref:
                    enriched.accessory_masks[acc] = ref
                    enriched.visibility_hints[f"accessory:{acc}"] = "visible" if conf >= 0.5 else "partially_visible"
                    enriched.provenance_by_region[f"accessory:{acc}"] = "parser:fashn"

        face_source = facer if facer.masks else AdapterSegmentationOutput(
            masks={k: v for k, v in fashn.masks.items() if k in {"face", "hair"}},
            confidences={k: fashn.confidences.get(k, 0.0) for k in ("face", "hair")},
        )
        for region, mask in face_source.masks.items():
            canonical = "face_skin" if region == "face" else ("hairline_or_hair_face_boundary" if region == "hair" else region)
            conf = face_source.confidences.get(region, 0.0)
            src = "parser:facer" if facer.masks else "parser:fashn"
            ref = _store_mask(
                mask,
                conf,
                src,
                "face",
                "face_region_mask",
                src.split(":")[-1],
                roi_bbox=(person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h),
                frame_size=frame_size,
                person_id=person.detection_id,
            )
            if ref is None:
                continue
            face_regions.append(FaceRegionPrediction(canonical, ref, conf, src))
            enriched.face_region_masks[canonical] = ref
            enriched.visibility_hints[f"face:{canonical}"] = "visible" if conf >= 0.5 else "partially_visible"
            enriched.provenance_by_region[f"face:{canonical}"] = src

        occlusion_hints = ["torso_visible" if "torso" in topology else "torso_hidden"]
        for part in ("upper_arm", "arms"):
            if part in topology:
                occlusion_hints.append("arms_visible")
                break
        else:
            occlusion_hints.append("arms_partially_occluded")

        max_conf = max(
            [0.0]
            + list(fashn.confidences.values())
            + list(pascal.confidences.values())
            + list(atr.confidences.values())
            + list(facer.confidences.values())
        )
        return ParsingPrediction(
            mask_ref=enriched.person_mask_ref,
            mask_confidence=max_conf,
            source="parser:human-stack:fused",
            garments=garments,
            occlusion_hints=occlusion_hints,
            body_parts=body_parts,
            face_regions=face_regions,
            enriched=enriched,
        )


class SegFormerHumanParserAdapter:
    """Production-oriented parser stack: FASHN + SCHP Pascal + SCHP ATR + FACER + fusion."""

    source_name = "parser:human-stack"

    def __init__(
        self,
        config: ParserStackConfig | BackendConfig | None = None,
        *,
        fashn_adapter: FashnHumanParserAdapter | None = None,
        schp_pascal_adapter: SCHPPascalPartParserAdapter | None = None,
        schp_atr_adapter: SCHPATRParserAdapter | None = None,
        facer_adapter: FacerFaceParserAdapter | None = None,
    ) -> None:
        if isinstance(config, BackendConfig):
            self.config = ParserStackConfig.from_backend_config(config)
        else:
            self.config = config or ParserStackConfig()
        self.fashn = fashn_adapter or FashnHumanParserAdapter(self.config.primary_human_parser)
        self.schp_pascal = schp_pascal_adapter or SCHPPascalPartParserAdapter(self.config.structural_body_parser)
        self.schp_atr = schp_atr_adapter or SCHPATRParserAdapter(self.config.garment_refinement_parser)
        self.facer = facer_adapter or FacerFaceParserAdapter(self.config.face_parser)
        self.fusion = ParserFusionEngine()
        self.last_runtime_formats: dict[str, dict[str, str]] = {}

    def is_builtin_backend(self) -> bool:
        return self.config.is_builtin()

    def _builtin_parse(self, frame: FrameLike, person: PersonDetection) -> ParsingPrediction:
        feats = frame_to_features(frame)
        return ParsingPrediction(
            mask_ref=f"mask::builtin::{person.detection_id}",
            mask_confidence=min(0.99, max(0.2, feats[2])),
            source="parser:human-stack:builtin",
            garments=[
                GarmentPrediction("top", "visible", min(0.95, max(0.1, feats[0])), "parser:human-stack:builtin"),
            ],
            occlusion_hints=["torso_visible" if feats[3] <= 0.6 else "torso_partial"],
        )

    def parse(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        result: dict[str, ParsingPrediction] = {}
        self.last_runtime_formats = {}
        profiler = None
        if isinstance(frame, PerceptionFrameContext):
            profiler = frame.get("profiler")
        if self.config.is_builtin():
            for person in persons:
                result[person.detection_id] = self._builtin_parse(frame, person)
                self.last_runtime_formats[person.detection_id] = {
                    "fashn": "builtin",
                    "schp_pascal": "builtin",
                    "schp_atr": "builtin",
                    "facer": "builtin",
                }
            return result

        frame_img = frame_to_numpy_rgb(frame)

        rgb = frame_img.rgb
        frame_size = (frame_img.width, frame_img.height)
        for person in persons:
            if profiler:
                with profiler.track("roi_crop"):
                    patch = crop_rgb(rgb, person.bbox)
            else:
                patch = crop_rgb(rgb, person.bbox)
            if profiler:
                with profiler.track("parser_inference"):
                    fashn = self.fashn.parse_patch(patch)
                    pascal = self.schp_pascal.parse_patch(patch)
                    atr = self.schp_atr.parse_patch(patch)
                    facer = self.facer.parse_patch(patch)
            else:
                fashn = self.fashn.parse_patch(patch)
                pascal = self.schp_pascal.parse_patch(patch)
                atr = self.schp_atr.parse_patch(patch)
                facer = self.facer.parse_patch(patch)

            if profiler:
                with profiler.track("fusion"):
                    fused = self.fusion.fuse(person, fashn, pascal, atr, facer, frame_size=frame_size)
            else:
                fused = self.fusion.fuse(person, fashn, pascal, atr, facer, frame_size=frame_size)

            self.last_runtime_formats[person.detection_id] = {
                "fashn": fashn.runtime_format,
                "schp_pascal": pascal.runtime_format,
                "schp_atr": atr.runtime_format,
                "facer": facer.runtime_format,
            }

            result[person.detection_id] = fused
        return result
