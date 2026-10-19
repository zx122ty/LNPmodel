# v5 shim user-dir: importing this module registers BOTH the stock unimol2
# task/loss/model AND the v5 MTR task/loss/model (unimol2_mtr).
# Point `unicore-train --user-dir <this directory>`; unicore imports this module
# by basename ("unimol2_v5"), whose import-time side effect registers everything.
import unimol2          # noqa: F401 — registers base task/loss/arch
import unimol2_mtr      # noqa: F401 — registers unimol2_mtr task/loss/arch