# Auto-discovery for unimol2_mtr submodules (mirrors unimol2.tasks/__init__.py).
from pathlib import Path
import importlib

for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("unimol2_mtr.tasks." + file.name[:-3])