from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class BenchmarkRecord:
    record_id: str
    asset_id: str
    asset_path: str
    scenario_id: str
    canonical_prompt: str
    action_family: str
    transition_family: str
    expected_region_families: list[str]
    expected_runtime_conditions: dict[str, object]
    tags: list[str]
    notes: str
    weak_expectations: dict[str, object]


@dataclass(slots=True)
class BenchmarkDataset:
    dataset_id: str
    name: str
    version: str
    manifest_path: str
    records: list[BenchmarkRecord]


@dataclass(slots=True)
class BenchmarkDatasetDiagnostics:
    total_records: int
    valid_records: int
    invalid_records: int
    missing_assets: int
    invalid_record_ids: list[str]
    missing_asset_record_ids: list[str]


def default_benchmark_manifest_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "benchmark_assets" / "single_image_curated" / "manifest.json"


def _to_list_str(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return []


def load_benchmark_dataset(manifest_path: str | Path | None = None) -> tuple[BenchmarkDataset, BenchmarkDatasetDiagnostics]:
    manifest = Path(manifest_path) if manifest_path is not None else default_benchmark_manifest_path()
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    dataset_id = str(payload.get("dataset_id", "single_image_curated"))
    name = str(payload.get("name", "single_image_curated"))
    version = str(payload.get("version", "0"))
    records_payload = payload.get("records", [])
    if not isinstance(records_payload, list):
        raise ValueError("Benchmark dataset manifest must contain list field 'records'.")

    invalid_record_ids: list[str] = []
    missing_asset_record_ids: list[str] = []
    records: list[BenchmarkRecord] = []

    for index, raw in enumerate(records_payload):
        record_raw = raw if isinstance(raw, dict) else {}
        record_id = str(record_raw.get("record_id", f"record_{index}"))

        required = [
            record_raw.get("asset_id"),
            record_raw.get("asset_path"),
            record_raw.get("scenario_id"),
            record_raw.get("canonical_prompt"),
            record_raw.get("action_family"),
            record_raw.get("transition_family"),
        ]
        if any((not isinstance(v, str)) or (not v.strip()) for v in required):
            invalid_record_ids.append(record_id)
            continue

        expected_regions = _to_list_str(record_raw.get("expected_region_families", []))
        if not expected_regions:
            invalid_record_ids.append(record_id)
            continue

        runtime = record_raw.get("expected_runtime_conditions", {})
        if not isinstance(runtime, dict):
            invalid_record_ids.append(record_id)
            continue

        asset_path = (manifest.parent / str(record_raw["asset_path"])).resolve()
        if not asset_path.exists():
            missing_asset_record_ids.append(record_id)
            continue

        records.append(
            BenchmarkRecord(
                record_id=record_id,
                asset_id=str(record_raw["asset_id"]),
                asset_path=str(asset_path),
                scenario_id=str(record_raw["scenario_id"]),
                canonical_prompt=str(record_raw["canonical_prompt"]),
                action_family=str(record_raw["action_family"]),
                transition_family=str(record_raw["transition_family"]),
                expected_region_families=expected_regions,
                expected_runtime_conditions=runtime,
                tags=_to_list_str(record_raw.get("tags", [])),
                notes=str(record_raw.get("notes", "")),
                weak_expectations=record_raw.get("weak_expectations", {}) if isinstance(record_raw.get("weak_expectations", {}), dict) else {},
            )
        )

    diagnostics = BenchmarkDatasetDiagnostics(
        total_records=len(records_payload),
        valid_records=len(records),
        invalid_records=len(invalid_record_ids),
        missing_assets=len(missing_asset_record_ids),
        invalid_record_ids=invalid_record_ids,
        missing_asset_record_ids=missing_asset_record_ids,
    )

    if not records:
        raise ValueError(
            f"No valid benchmark records loaded from manifest {manifest}. "
            f"invalid={diagnostics.invalid_records}, missing_assets={diagnostics.missing_assets}"
        )

    dataset = BenchmarkDataset(dataset_id=dataset_id, name=name, version=version, manifest_path=str(manifest), records=records)
    return dataset, diagnostics


def dataset_diagnostics_to_dict(diagnostics: BenchmarkDatasetDiagnostics) -> dict[str, object]:
    return asdict(diagnostics)
