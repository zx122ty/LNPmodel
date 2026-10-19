"""
反应模板库：可电离脂质的组合化学拼接规则。

设计依据（实证）：
  LNPDB 的 head/tail 列是**合成试剂**而非拆解片段。用正向反应模板
  可精确重建 IL_SMILES，因此枚举产物自带合成路线，可合成性接近 100%。

  在纯二组分真实样本上验证的重建率（模板覆盖范围内 86.4%）：
    n_alkylation        100.0%    acrylamide_michael  96.8%
    reductive_amination  90.9%    acrylate_michael    87.7%
    epoxide_opening      64.8%

  epoxide 未覆盖的 90 例经核查为 LNPDB 的 head 列标注错误（产物含
  两种反应物都不存在的丙基），非模板缺陷。

  真实产物 391/391 为**完全取代**（所有 N-H 位点耗尽），零例部分取代，
  故枚举只输出完全取代产物——部分取代会引入非真实分子并稀释预训练分布。
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# 亲核位点：sp3 氮且带 H，排除酰胺/磺酰胺/肼/亚胺
_N_NUC = "[N;!$(NC=O);!$(N[S,P]=O);!$(N[#7]);!$(N=*);!H0:1]"

REACTION_SMARTS = {
    # 迈克尔加成——最主流的 IL 合成路线
    "acrylate_michael": f"{_N_NUC}.[CH2:2]=[CH:3][C:4](=[O:5])[O:6]"
                        ">>[N:1][CH2:2][CH2:3][C:4](=[O:5])[O:6]",
    "acrylamide_michael": f"{_N_NUC}.[CH2:2]=[CH:3][C:4](=[O:5])[N:6]"
                          ">>[N:1][CH2:2][CH2:3][C:4](=[O:5])[N:6]",
    # 环氧开环——胺进攻位阻较小的末端碳（实测真实产物均为此区域选择性）
    "epoxide_opening": f"{_N_NUC}.[C:2]1[O:3][CH2:4]1>>[N:1][CH2:4][C:2][OH:3]",
    # N-烷基化
    "n_alkylation": f"{_N_NUC}.[C:2][Br,I,Cl]>>[N:1][C:2]",
    # 还原胺化
    "reductive_amination": f"{_N_NUC}.[CX3H1:2]=[O:3]>>[N:1][CH2:2]",
}

REACTIONS = {k: AllChem.ReactionFromSmarts(v) for k, v in REACTION_SMARTS.items()}

# 每个氮原子允许的取代次数上限。
# 还原胺化经亚胺中间体，一个 NH2 只能形成一个 C=N，还原后即为仲胺而停止——
# 实测真实产物均为仲胺（CNCCN...），继续烷基化到叔胺是错的。
# 其余反应（迈克尔加成/环氧开环/烷基化）无此限制，可耗尽全部 N-H。
MAX_SUBSTITUTIONS_PER_N = {
    "reductive_amination": 1,
}

# 各反应对应的亲电试剂特征——用于给试剂自动打 rxn_type 标签
ELECTROPHILE_PATTERNS = {
    "acrylate_michael": Chem.MolFromSmarts("[CH2]=[CH]C(=O)[OX2][#6]"),
    "acrylamide_michael": Chem.MolFromSmarts("[CH2]=[CH]C(=O)[NX3]"),
    "epoxide_opening": Chem.MolFromSmarts("[CX4]1[OX2][CH2]1"),
    "n_alkylation": Chem.MolFromSmarts("[CX4;!$(C=O)][Br,I,Cl]"),
    "reductive_amination": Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
}

PAT_N_NUCLEOPHILE = Chem.MolFromSmarts("[N;!$(NC=O);!$(N[S,P]=O);!$(N[#7]);!$(N=*);!H0]")


def count_nh_sites(mol):
    """可反应的 N-H 位点总数（一个 NH2 计 2 个位点）。"""
    if mol is None:
        return 0
    return sum(mol.GetAtomWithIdx(idx).GetTotalNumHs()
               for (idx,) in mol.GetSubstructMatches(PAT_N_NUCLEOPHILE))


def classify_electrophile(mol):
    """返回该试剂可参与的反应类型列表。"""
    if mol is None:
        return []
    return [rxn for rxn, pat in ELECTROPHILE_PATTERNS.items()
            if pat is not None and mol.HasSubstructMatch(pat)]


def enumerate_exhaustive(head_smiles, tail_smiles, rxn_name, max_steps=10, max_products=64):
    """
    穷尽取代：反复在 head 的 N-H 位点上接 tail，直到无位点可反应。

    只返回**完全取代**产物（终态），与真实数据一致。
    还原胺化受 MAX_SUBSTITUTIONS_PER_N 限制，每个氮只取代一次（产物为仲胺）。
    max_products 是防爆炸上限——多胺头基的中间态组合会指数增长。

    返回 canonical SMILES 集合。
    """
    rxn = REACTIONS.get(rxn_name)
    if rxn is None:
        return set()
    head = Chem.MolFromSmiles(head_smiles)
    tail = Chem.MolFromSmiles(tail_smiles)
    if head is None or tail is None:
        return set()

    # 取代次数上限：受限反应按初始氮原子数计，否则按可耗尽的 N-H 总数计
    per_n = MAX_SUBSTITUTIONS_PER_N.get(rxn_name)
    if per_n is not None:
        step_cap = per_n * len(head.GetSubstructMatches(PAT_N_NUCLEOPHILE))
    else:
        step_cap = max_steps
    step_cap = min(step_cap, max_steps)

    frontier = {Chem.MolToSmiles(head)}
    terminal = set()
    for _ in range(step_cap):
        nxt = set()
        for smi in frontier:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            products = rxn.RunReactants((mol, tail))
            if not products:
                terminal.add(smi)      # 无位点可反应 = 完全取代
                continue
            advanced = False
            for p in products:
                try:
                    pm = p[0]
                    Chem.SanitizeMol(pm)
                    nxt.add(Chem.MolToSmiles(pm))
                    advanced = True
                except Exception:
                    continue
            if not advanced:
                terminal.add(smi)
        if not nxt:
            break
        if len(nxt) > max_products:     # 截断中间态广度，避免组合爆炸
            nxt = set(sorted(nxt)[:max_products])
        frontier = nxt

    terminal |= frontier
    terminal.discard(Chem.MolToSmiles(head))   # 未反应的原料不是产物
    return terminal
