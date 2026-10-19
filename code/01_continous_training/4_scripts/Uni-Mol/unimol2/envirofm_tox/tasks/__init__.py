# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
"""Task-level fine-tuning extensions for EnviroFM-v2.

Auto-discovery mirrors ``unimol2.tasks/__init__.py``: every non-underscore module in this
directory is imported, so dropping a new task file in registers it with Uni-Core.
"""
from pathlib import Path
import importlib

for _file in sorted(Path(__file__).parent.glob("*.py")):
    if not _file.name.startswith("_"):
        importlib.import_module("envirofm_tox.tasks." + _file.name[:-3])
