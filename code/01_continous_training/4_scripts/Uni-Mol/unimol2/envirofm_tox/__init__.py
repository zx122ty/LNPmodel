# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
# EnviroFM-v2 fine-tuning extension package.
#
# This package is a *sibling* of the vendored ``unimol2`` package rather than a patch to
# it. It registers the task and optimizer the EnviroFM-v2 protocol needs (plan §7.1) on
# top of the stock ``unimol2`` implementation and imports everything else from it, so the
# vendored tree stays byte-identical to upstream and an upstream re-pull cannot silently
# drop our changes. ``unimol2_mtr`` is the existing precedent for this pattern.
#
# Neither the losses nor the model need extending: plan §11's A0 baseline is explicitly
# "Standard MSE / Cross-Entropy" over the stock heads, and the two things the stock code
# cannot do — target standardisation on a non-QM9 endpoint, and layer-wise LR decay — are
# a task concern and an optimizer concern respectively.
import importlib

import envirofm_tox.tasks
import envirofm_tox.optim
from envirofm_tox import param_grouping

__all__ = ["tasks", "optim", "param_grouping"]
