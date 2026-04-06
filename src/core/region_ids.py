from __future__ import annotations


def make_region_id(entity_id: str, region_type: str) -> str:
    entity = (entity_id or "scene").strip() or "scene"
    region = region_type.strip().replace(":", "_")
    return f"{entity}:{region}"


def parse_region_id(region_id: str) -> tuple[str, str]:
    if ":" not in region_id:
        return "scene", region_id
    entity_id, region_type = region_id.split(":", 1)
    return (entity_id or "scene"), (region_type or "unknown")


def is_canonical_region_id(region_id: str) -> bool:
    entity_id, region_type = parse_region_id(region_id)
    return bool(entity_id) and bool(region_type) and ":" in region_id
