# Auto-discovery for unimol2_mtr model modules (mirrors unimol2.models/__init__.py).
from pathlib import Path
import importlib

for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("unimol2_mtr.models." + file.name[:-3])