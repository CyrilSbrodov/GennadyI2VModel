import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text.learned_bridge import BaselineTextEncoderAdapter

adapter = BaselineTextEncoderAdapter()
out = adapter.encode("Поднимает руку и аккуратно поворачивает голову")

print(out.structured_action_tokens)
print(out.target_hints)
print(out.temporal_hints)
print(out.constraints)
print(out.confidence)
print(out.diagnostics)