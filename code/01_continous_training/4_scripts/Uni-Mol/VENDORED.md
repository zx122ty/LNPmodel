# Vendored Uni-Mol2 task code

This directory contains the `unimol2` task package used by
`4_scripts/run_lipids_continue_pretrain.py` through Uni-Core's `--user-dir`.

- Upstream: https://github.com/deepmodeling/Uni-Mol
- Commit: `90f52c41299a1a582da0f9765e9f87aa21faa16a`
- Vendored path: `unimol2/`
- License: MIT, copied to `LICENSE`

The package is vendored because the training command requires a stable
`--user-dir` and the previous local symlink pointed outside this repository.
Python bytecode and the upstream Git metadata are intentionally excluded.
