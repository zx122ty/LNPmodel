"""
全局配置：路径、阈值、SMARTS 判据。

所有数值阈值均由 LNPDB.csv 中 12,844 个真实可电离脂质的实测分布反推得到，
而非套用类药性规则（Lipinski 会删掉 >90% 的真实脂质）。
"""

import os
from pathlib import Path

# ------------------------------------------------------------------ 路径
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("LNP_RAW_DIR", PROJECT_DIR / "raw_data"))
OUT_DIR = Path(os.environ.get("LNP_STAGE0_OUT_DIR", PROJECT_DIR / "output"))

SRC_LNPDB = RAW_DIR / "LNPDB.csv"
SRC_ATLAS = RAW_DIR / "LNP_Atlas_DB_202509_v1.csv"
SRC_ABSP = RAW_DIR / "paper_ABSP.csv"
SRC_MTCS = RAW_DIR / "paper_MTCS.csv"
SRC_UMI = RAW_DIR / "paper_umi-lab.csv"

# 各文件编码互不相同，实测确认：Atlas 是 latin-1，两个 paper_* 带 UTF-8 BOM
ENCODINGS = {
    "LNPDB": "utf-8",
    "Atlas": "latin-1",
    "ABSP": "utf-8-sig",
    "MTCS": "utf-8",
    "umi-lab": "utf-8-sig",
}

# LNP_Atlas 经核实 SMILES 与化学名不一致（疑似 LLM 生成未校验），整体排除
EXCLUDE_SOURCES = {"Atlas"}

# ------------------------------------------------------------ 可电离判据
# 碱性胺：sp3 氮，排除酰胺/磺酰胺/肼/亚胺/季铵。允许芳胺（苯胺型 IL 真实存在）
SMARTS_BASIC_AMINE = (
    "[NX3;H0,H1,H2;"
    "!$(N[#6]=[O,N,S]);"   # 排除酰胺、脲、硫脲
    "!$(N[S,P]=O);"        # 排除磺酰胺、磷酰胺
    "!$(N[#7]);"           # 排除肼类
    "!$(N=*);"             # 排除亚胺
    "!$([N+])]"            # 排除已带正电
)
SMARTS_QUAT_N = "[NX4+;!$([NX4+][#6]=[O])]"          # 永久季铵，不可电离
SMARTS_PHOSPHATE = "P(=O)([O,OH,O-])[O,OH,O-]"       # 磷脂（helper lipid，非 IL）

# 阈值标定结果（详见 merge_report.md）：
#   LNPDB 真实 IL 召回 88.33% / 非脂质负样本假阳仅 3.10%
# 用「总脂肪碳数」而非「最长单链」——多臂树枝状脂质每条臂都短，但整体高度疏水
MIN_LOGP = 2.0        # 真实 IL 的 logP p1 = 1.27
MIN_SP3_CARBONS = 12  # 疏水质量下限
MIN_CHAIN_LEN = 4     # 最长非环碳链，仅用于排除无任何脂链的分子

# ------------------------------------------------------------- 性质过滤窗口
# 取自真实 IL 分布的 [p1, p99]，用于剔除极端异常值而非类药性筛选
MW_RANGE = (200.0, 3000.0)     # 真实: p1=354, p50=738, p99=2616, max=3984
LOGP_RANGE = (-4.0, 40.0)      # 真实: p1=1.3, p50=10.9, p99=36.7
HEAVY_ATOM_RANGE = (14, 300)   # 真实: p1=24, p50=52, p99=178, max=282

# Uni-Mol2 约束：conformer.py 在原子数超限时会「随机丢弃原子」而非报错，
# 导致分子被静默破坏。采用 remove_hs=True，预算按重原子计。
UNIMOL_MAX_ATOMS = 256
UNIMOL_REMOVE_HS = True

# 虚拟库目标规模。
# 关键事实：Uni-Mol2 的 ConformerSampleDataset 每个 epoch 只从每个分子中
# 随机取 1 个构象（conformer_sample_dataset.py:34），因此构象数量**不会**
# 增加样本数，只起 3D 数据增强作用。真实:虚拟的有效样本比完全由分子数决定。
# 取 10 万 → 12,381 : 100,000 ≈ 1:8，落在目标区间，且相比 100 万可节省
# 约 5 天的构象生成时间（枚举池 virtual_raw.parquet 保留，随时可扩容）。
TARGET_VIRTUAL_SIZE = 100_000

# ------------------------------------------------------------------ 其他
RANDOM_SEED = 42
N_JOBS = 20  # 机器 24 核，留 4 核
