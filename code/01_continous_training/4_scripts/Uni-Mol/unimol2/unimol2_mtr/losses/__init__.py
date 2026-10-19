# Auto-discovery for unimol2_mtr loss modules (mirrors unimol2.losses/__init__.py).
from pathlib import Path
import importlib

for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("unimol2_mtr.losses." + file.name[:-3])