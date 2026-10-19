#!/usr/bin/env bash
# Reproducible build of the `unimol2` conda env on this host.
#
# Why this exists instead of a bare `conda env create -f unimol2_environment.yml`:
#
#   1. `unicore==0.0.1` is not on PyPI. It is the in-repo Uni-Core source
#      (Uni-Core-main/, version.txt == 0.0.1). Its setup.py does `import torch`
#      at module level, and conda runs pip's build for local packages before
#      installing anything -- with PEP 517 build isolation that build env has no
#      torch, so an `-e ./Uni-Core-main` line in the yml can never resolve.
#      It has to be a second step with --no-build-isolation.
#
#   2. The cu130 torch wheels only exist on download.pytorch.org, which serves a
#      single connection at ~45 KB/s and stalls outright. pip has no way to
#      parallelize a wheel download, so torch (502 MB) never finishes. We pull
#      the three wheels with ./pardl.py (parallel ranged HTTP, ~6.5 MB/s) and
#      point pip at them with --find-links.
#
# Everything else comes from the Tsinghua PyPI mirror. Uni-Core's CUDA
# extensions are off by default (setup.py DISABLE_CUDA_EXTENSION = True), so no
# nvcc is needed.
#
# Usage: ./install_env.sh          (idempotent; safe to re-run after a failure)

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

ENV_NAME=unimol2
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_BASE}/envs/${ENV_NAME}}"
PY="${ENV_PREFIX}/bin/python"
TUNA=https://pypi.tuna.tsinghua.edu.cn/simple/
PT_INDEX=https://download.pytorch.org/whl/cu130
WHEELHOUSE=./wheelhouse

# --- 1. conda half: python + the conda-level deps -----------------------------
if [ ! -x "${PY}" ]; then
  echo "==> creating conda env at ${ENV_PREFIX}"
  # Only the conda section runs; the pip section of the yml is handled below.
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi

# --- 2. parallel-fetch the torch wheels ---------------------------------------
mkdir -p "${WHEELHOUSE}"
fetch() {  # fetch <wheel-filename> <parts>
  local whl="$1" parts="${2:-12}"
  [ -s "${WHEELHOUSE}/${whl}" ] && { echo "    have ${whl}"; return 0; }
  echo "==> downloading ${whl}"
  # URL-encode the '+' in the local version tag.
  python3 pardl.py "${PT_INDEX}/${whl/+/%2B}" "${WHEELHOUSE}/${whl}" --parts "${parts}"
}

fetch torch-2.13.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl 12
fetch torchvision-0.28.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl 4
fetch torchaudio-2.11.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl 4

# --- 3. pip half: everything else, torch satisfied from ./wheelhouse ----------
# Extract the pip: block from the yml (comments and the --extra-index-url line
# dropped) so the yml stays the single source of truth for versions.
sed -n '/^  - pip:/,$p' unimol2_environment.yml \
  | tail -n +2 | sed 's/^      //' \
  | grep -v '^#' | grep -v '^$' | sed 's/^- //' \
  > ./pip_reqs.txt

echo "==> pip install ($(wc -l < ./pip_reqs.txt) pins)"
"${PY}" -m pip install -U -r ./pip_reqs.txt \
  --find-links "${WHEELHOUSE}" --index-url "${TUNA}" --exists-action=b

# --- 4. Uni-Core, built against the now-installed torch -----------------------
echo "==> installing Uni-Core (no CUDA extensions)"
"${PY}" -m pip install --no-build-isolation -e ./Uni-Core-main

# --- 5. smoke test ------------------------------------------------------------
"${PY}" - <<'EOF'
import torch, rdkit, xgboost, lightgbm, numba
import unicore
from unicore_cli.train import cli_main          # the entry point the pipeline shells out to
from unicore.logging.metrics import aggregate   # train-time metric plumbing
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available(), "| devices", torch.cuda.device_count())
print("unicore", unicore.__version__, "| cli ok | rdkit", rdkit.__version__)
EOF
echo "==> done"
