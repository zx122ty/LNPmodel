# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
"""Optimizer extensions for EnviroFM-v2.

Auto-discovery mirrors ``unicore.optim/__init__.py``: every non-underscore module in this
directory is imported, so a new optimizer dropped in registers itself with Uni-Core and
its ``add_args`` becomes reachable from the CLI (``parse_args_and_arch`` loops over
``REGISTRIES`` after the task's args are added).
"""
from pathlib import Path
import importlib

for _file in sorted(Path(__file__).parent.glob("*.py")):
    if not _file.name.startswith("_"):
        importlib.import_module("envirofm_tox.optim." + _file.name[:-3])
