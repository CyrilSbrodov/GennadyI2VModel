from __future__ import annotations

import json

from runtime.orchestrator import GennadyEngine


if __name__ == "__main__":
    engine = GennadyEngine()
    result = engine.run(["ref_0001.png"], "Снимает пальто и садится на стул. Улыбается.")
    print(json.dumps({"frames": len(result.frames), "state_steps": len(result.state_plan.steps)}, ensure_ascii=False))
