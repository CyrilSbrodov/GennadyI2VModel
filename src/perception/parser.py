from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from core.input_layer import AssetFrame
from perception.backend import frame_to_features
from perception.detector import BackendConfig, PersonDetection
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
    def parse(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        ...


@dataclass(slots=True)
class AdapterSegmentationOutput:
    masks: dict[str, "object"] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)


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


def _store_mask(mask: "object", confidence: float, source: str, prefix: str, kind: str, backend: str) -> str | None:
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
        )
    except Exception:
        return None


class FashnHumanParserAdapter:
    source_name = "parser:fashn"
    # Контракт по классам FASHN (18 useful classes + background=0)
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

    def _hf_infer(self, patch: "object"):
        if self._infer_fn is not None:
            return self._infer_fn(patch)
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers is not installed for fashn parser") from exc
        model_id = self.config.variant or "fashn-ai/fashn-human-parser"
        pipe = pipeline("image-segmentation", model=model_id, device=self.config.device)
        return pipe(patch)

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput()
        raw = self._hf_infer(patch)
        if isinstance(raw, dict) and "label_map" in raw:
            masks = _label_map_to_masks(raw["label_map"], self.CLASS_MAP)
        else:
            # Для production-контракта принимаем только label_map path; list fallback лишь для тестов.
            masks = {}
            if isinstance(raw, list):
                for seg in raw:
                    label = str(seg.get("label", "")).lower().strip()
                    mask = seg.get("mask")
                    if mask is not None and label in self.CLASS_MAP.values() and label != "background":
                        masks[label] = _safe_array(mask)
        confs = {name: _mask_conf(mask) for name, mask in masks.items()}
        return AdapterSegmentationOutput(masks=masks, confidences=confs)


class SCHPPascalPartParserAdapter:
    source_name = "parser:schp_pascal"
    # Явный mapping Pascal-Person-Part -> canonical body topology.
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

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput()
        if self._infer_fn is None:
            raise RuntimeError("SCHP Pascal parser requires infer_fn or backend-specific integration")
        raw_masks = self._infer_fn(patch)
        masks: dict[str, object] = {}
        for src_label, mask in raw_masks.items():
            canonical = self.LABEL_TO_CANONICAL.get(src_label.lower().strip())
            if canonical:
                masks[canonical] = _safe_array(mask)
        confs = {name: _mask_conf(mask) for name, mask in masks.items()}
        return AdapterSegmentationOutput(masks=masks, confidences=confs)


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

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput()
        if self._infer_fn is None:
            raise RuntimeError("SCHP ATR parser requires infer_fn or backend-specific integration")
        raw_masks = self._infer_fn(patch)
        masks: dict[str, object] = {}
        for src_label, mask in raw_masks.items():
            canonical = self.LABEL_TO_CANONICAL.get(src_label.lower().strip())
            if canonical:
                out_name = canonical if canonical not in {"arm", "leg"} else f"{canonical}_region"
                masks[out_name] = _safe_array(mask)
        confs = {name: _mask_conf(mask) for name, mask in masks.items()}
        return AdapterSegmentationOutput(masks=masks, confidences=confs)


class FacerFaceParserAdapter:
    source_name = "parser:facer"
    # Явный mapping для face parsing outputs (варианты lapa/celebm допускаются).
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

    def parse_patch(self, patch: "object") -> AdapterSegmentationOutput:
        if self.config.backend == "builtin":
            return AdapterSegmentationOutput()
        if self._infer_fn is None:
            raise RuntimeError("FACER parser requires infer_fn or backend-specific integration")
        raw_masks = self._infer_fn(patch)
        masks: dict[str, object] = {}
        for src_label, mask in raw_masks.items():
            canonical = self.LABEL_TO_CANONICAL.get(src_label.lower().strip())
            if canonical:
                masks[canonical] = _safe_array(mask)
        confs = {name: _mask_conf(mask) for name, mask in masks.items()}
        return AdapterSegmentationOutput(masks=masks, confidences=confs)


class ParserFusionEngine:
    # Приоритеты по ТЗ.
    GARMENT_PRIORITY = ("fashn", "schp_atr", "builtin")

    def fuse(
        self,
        person: PersonDetection,
        fashn: AdapterSegmentationOutput,
        pascal: AdapterSegmentationOutput,
        atr: AdapterSegmentationOutput,
        facer: AdapterSegmentationOutput,
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
            pref = _store_mask(union_mask, _mask_conf(union_mask), "parser:fusion", "person", "person_mask", "fusion")
            enriched.person_mask_ref = pref

        # Body topology: Pascal > FASHN coarse.
        topology = dict(fashn.masks)
        topology.update(pascal.masks)
        for part, mask in topology.items():
            if part in {"top", "dress", "skirt", "pants", "belt", "bag", "hat", "scarf", "glasses", "jewelry"}:
                continue
            conf = pascal.confidences.get(part, fashn.confidences.get(part, 0.0))
            source = "parser:schp_pascal" if part in pascal.masks else "parser:fashn"
            ref = _store_mask(mask, conf, source, "body", "body_part_mask", source.split(":")[-1])
            if ref is None:
                continue
            body_parts.append(BodyPartMaskPrediction(part, ref, conf, "visible" if conf >= 0.5 else "partially_visible", source))
            enriched.body_part_masks[part] = ref
            enriched.provenance_by_region[f"body:{part}"] = source

        # Garments: FASHN primary, ATR fallback/enrichment.
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
                ref = _store_mask(out.masks[g], conf, src, "garment", "garment_mask", source_name)
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
                    enriched.provenance_by_region[f"garment:{g}"] = src

        for acc in ("bag", "hat", "glasses", "jewelry", "scarf", "belt"):
            if acc in fashn.masks:
                conf = fashn.confidences.get(acc, 0.0)
                ref = _store_mask(fashn.masks[acc], conf, "parser:fashn", "accessory", "accessory_mask", "fashn")
                if ref:
                    enriched.accessory_masks[acc] = ref
                    enriched.provenance_by_region[f"accessory:{acc}"] = "parser:fashn"

        # Face regions: FACER authoritative > FASHN coarse face/hair.
        face_source = facer if facer.masks else AdapterSegmentationOutput(
            masks={k: v for k, v in fashn.masks.items() if k in {"face", "hair"}},
            confidences={k: fashn.confidences.get(k, 0.0) for k in ("face", "hair")},
        )
        for region, mask in face_source.masks.items():
            canonical = "face_skin" if region == "face" else ("hairline_or_hair_face_boundary" if region == "hair" else region)
            conf = face_source.confidences.get(region, 0.0)
            src = "parser:facer" if facer.masks else "parser:fashn"
            ref = _store_mask(mask, conf, src, "face", "face_region_mask", src.split(":")[-1])
            if ref is None:
                continue
            face_regions.append(FaceRegionPrediction(canonical, ref, conf, src))
            enriched.face_region_masks[canonical] = ref
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

    def is_builtin_backend(self) -> bool:
        return self.config.is_builtin()

    def _builtin_parse(self, frame: AssetFrame | list[list[list[float]]] | str, person: PersonDetection) -> ParsingPrediction:
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

    def parse(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        result: dict[str, ParsingPrediction] = {}
        if self.config.is_builtin():
            for person in persons:
                result[person.detection_id] = self._builtin_parse(frame, person)
            return result

        rgb = frame_to_numpy_rgb(frame).rgb
        for person in persons:
            patch = crop_rgb(rgb, person.bbox)
            fashn = self.fashn.parse_patch(patch)
            pascal = self.schp_pascal.parse_patch(patch)
            atr = self.schp_atr.parse_patch(patch)
            facer = self.facer.parse_patch(patch)
            result[person.detection_id] = self.fusion.fuse(person, fashn, pascal, atr, facer)
        return result
