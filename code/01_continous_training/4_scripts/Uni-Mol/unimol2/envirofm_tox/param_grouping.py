# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
"""Parameter-name registry and depth classification for layer-wise LR decay.

Layer-wise learning rate decay (plan §7.1) needs to know, for each ``Parameter`` object,
which block of the network it belongs to. Uni-Core's optimiser construction path hands the
optimiser a *list of parameters* — ``Trainer._build_optimizer`` builds
``[(name, param), ...]`` and then calls ``separate_decay_params``, which strips the names
before any optimiser class sees them. By the time a registered optimiser's ``__init__``
runs it therefore has parameters but no names, and depth is not recoverable from a tensor.

Rather than monkey-patching the vendored ``unicore.optim`` plumbing, the task (which is
the one place that *builds* the model, and therefore the only place that legitimately has
it) publishes the name map here, and the LLRD optimiser reads it back. The contract is
one-way and explicit: :func:`register_param_names` is called from
:meth:`envirofm_tox.tasks.tox_finetune.ToxFinetuneTask.build_model`.

Keys are ``id(param)`` rather than the parameter itself on purpose: ``torch.Tensor``
defines ``__eq__`` elementwise, so a dict keyed by tensors can raise
"Boolean value of Tensor with more than one element is ambiguous" on a hash collision.
``id`` is stable for the lifetime of the model: ``nn.Module.to``/``half``/``bfloat16``
mutate each parameter's ``.data`` in place rather than replacing the ``Parameter`` object,
and ``DistributedUnicoreModel`` delegates ``named_parameters`` to the wrapped module.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("envirofm_tox.param_grouping")

# ``encoder.layers.<i>.<rest>`` — the transformer blocks.
_LAYER_RE = re.compile(r"^encoder\.layers\.(\d+)\.")
# ``classification_heads.<head_name>.<rest>`` — the task head.
_HEAD_RE = re.compile(r"^classification_heads\.")

_PARAM_NAMES: dict[int, str] = {}
_UNNAMED_WARNED = False


def register_param_names(model) -> int:
    """Publish ``model``'s parameter names for :func:`name_of`. Returns the count."""
    global _UNNAMED_WARNED
    _PARAM_NAMES.clear()
    for name, param in model.named_parameters():
        _PARAM_NAMES[id(param)] = name
    _UNNAMED_WARNED = False
    log.info("param_grouping: registered %d parameter names", len(_PARAM_NAMES))
    return len(_PARAM_NAMES)


def name_of(param) -> str | None:
    """Return the registered name for ``param``, or ``None`` if it was never registered."""
    return _PARAM_NAMES.get(id(param))


def n_layers(names) -> int:
    """Number of transformer blocks implied by ``names``."""
    depths = [int(m.group(1)) for n in names if (m := _LAYER_RE.match(n))]
    return max(depths) + 1 if depths else 0


def classify(name: str | None) -> tuple[str, int]:
    """Map a parameter name to ``(role, depth)``.

    ``role`` is one of:

    * ``"head"`` — the task head (``classification_heads.*``). Always full LR.
    * ``"layer"`` — a transformer block, with ``depth`` = block index (0 = first block,
      i.e. furthest from the output; ``n_layers - 1`` = the top block, nearest the head).
    * ``"stem"`` — embeddings, atom/edge feature encoders, the SE(3) invariant kernel, the
      coordinate-refinement head, and the encoder layer-norms. Treated as sitting one
      level *below* block 0, the standard BERT-style convention.

    An unregistered name is reported as ``"stem"`` and warned about once: it means a
    parameter escaped registration (e.g. the model was built without the EnviroFM task),
    and silently giving it backbone LR would be a quiet protocol deviation.
    """
    global _UNNAMED_WARNED
    if name is None:
        if not _UNNAMED_WARNED:
            log.warning(
                "param_grouping: encountered unregistered parameter(s); treating them as "
                "backbone 'stem'. This means the model was not built by "
                "ToxFinetuneTask.build_model, so LLRD depths fall back to the stem level."
            )
            _UNNAMED_WARNED = True
        return "stem", -1

    if _HEAD_RE.match(name):
        return "head", -1
    m = _LAYER_RE.match(name)
    if m:
        return "layer", int(m.group(1))
    return "stem", -1


def lr_scale(name: str | None, *, n_layer_blocks: int, factor: float,
             backbone_ratio: float) -> float:
    """Multiplier to apply to the schedule's current (head) LR for ``name``.

    * head: ``1.0``
    * transformer block ``i`` of ``n``: ``backbone_ratio * factor ** (n - 1 - i)``
    * stem: ``backbone_ratio * factor ** n``

    With ``backbone_ratio = lr_backbone / lr_head`` the two LRs the plan fixes (head
    ``2e-4``, top backbone block ``2e-5``) come out exactly right by construction, and the
    decay is applied geometrically downwards from the top block to the stem.
    """
    role, depth = classify(name)
    if role == "head":
        return 1.0
    if role == "layer":
        return backbone_ratio * factor ** (n_layer_blocks - 1 - depth)
    return backbone_ratio * factor ** n_layer_blocks
