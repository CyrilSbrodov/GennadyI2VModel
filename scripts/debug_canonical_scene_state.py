from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perception.pipeline import PerceptionPipeline
from representation.canonical_human_state import CanonicalHumanSceneProcessor, canonical_state_to_dict
from representation.graph_builder import SceneGraphBuilder


def _canonical_summary(canonical_people: list[dict[str, object]]) -> dict[str, object]:
    people_summary: list[dict[str, object]] = []
    for person in canonical_people:
        regions = person.get("regions", {})
        rels = person.get("relations", [])
        if not isinstance(regions, dict):
            regions = {}
        visible = {
            name: {
                "visibility": data.get("visibility_state"),
                "confidence": data.get("confidence"),
                "provenance": data.get("provenance"),
            }
            for name, data in regions.items()
            if isinstance(data, dict)
        }
        people_summary.append(
            {
                "person_id": person.get("person_id"),
                "regions": visible,
                "relation_count": len(rels) if isinstance(rels, list) else 0,
                "relations": rels,
            }
        )
    return {"persons": people_summary}


def _scene_graph_to_jsonable(graph) -> dict[str, object]:
    return {
        "frame_index": graph.frame_index,
        "global_context": asdict(graph.global_context),
        "persons": [asdict(p) for p in graph.persons],
        "objects": [asdict(o) for o in graph.objects],
        "relations": [asdict(r) for r in graph.relations],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Perception -> canonical human state -> relation/visibility reasoning -> scene graph debug")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="artifacts/canonical_scene_debug", help="Output directory")
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PerceptionPipeline()
    perception = pipeline.analyze(str(args.image))

    processor = CanonicalHumanSceneProcessor()
    canonical_people = [
        canonical_state_to_dict(processor.process(person, person_id=f"person_{idx}", frame_size=perception.frame_size, objects=perception.objects))
        for idx, person in enumerate(perception.persons, start=1)
    ]

    builder = SceneGraphBuilder()
    graph = builder.build(perception, frame_index=args.frame_index)

    (out_dir / "perception_output.json").write_text(json.dumps(asdict(perception), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "canonical_state.json").write_text(json.dumps({"persons": canonical_people}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "canonical_summary.json").write_text(json.dumps(_canonical_summary(canonical_people), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "scene_graph.json").write_text(json.dumps(_scene_graph_to_jsonable(graph), indent=2, ensure_ascii=False), encoding="utf-8")


    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "persons": len(canonical_people),
                "graph_relations": len(graph.relations),
                "artifacts": [
                    "perception_output.json",
                    "canonical_state.json",
                    "canonical_summary.json",
                    "scene_graph.json",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
