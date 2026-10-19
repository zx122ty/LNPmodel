# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
# EnviroFM-v2 Option A (dual-stream hybrid) pretraining extension package.
#
# Sibling of the vendored ``unimol2`` package, same pattern as ``envirofm_tox`` and
# ``unimol2_mtr``: registers the task/loss/model on top of the stock implementation
# without patching the vendored tree, so an upstream re-pull cannot silently drop our
# changes. Imports (and therefore registers) the stock unimol2 package first so a run
# pointed at this user-dir has both the Option A arch and the stock arch available.
import unimol2  # noqa: F401 — registers base task/loss/arch
import unimol2_mtr  # noqa: F401 — MTRHead reuse + its registry
import unimol2_dual.tasks  # noqa: F401
import unimol2_dual.losses  # noqa: F401
import unimol2_dual.models  # noqa: F401

__all__ = ["tasks", "losses", "models"]
