#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lipid_metrics.py — 可电离脂质候选的 pKa 分级 + 3D 形状指标（可复用、默认独立）
================================================================================
本轮（P1）交付：作为"可导入模块"，不自动接入枚举主流程 —— 由
ugi_template_validate.py / apply_lipid_gates.py 通过 import 调用（或作为
独立命令行对照），默认不开启，避免拉长枚举并便于单条分子快速验证。

一、pKa 化学分级（无外部数值工具时的规则近似，SEGR 方式）
----------------------------------------------------------
对"残余碱性中心"做化学类型分档（pKa 为经验范围，分类而非精确数值）：
  * 脂肪叔胺  N(CH2)(CH2)(CH2)        pKa ≈ 9.4 - 10   → "high"
  * N-哌嗪     六元环2个N(1个成环-N)    pKa ≈ 8.3        → "medium+"
  * 吗啉       N-(CH2CH2)2O             pKa ≈ 8.3        → "medium+"
  * 咪唑       芳香五元环 [nH] / [n]    pKa ≈ 6.9        → "medium"
  * 吡啶类     芳香 [n]                 pKa ≈ 5.2        → "low"
目标区间（03.md §7.2）：pKa 6.0-7.0 理想，5.8-7.2 可接受。
因此本模块给出碱度等级，供后续在"产物/头部"上过滤或打分。

用法：
    from lipid_metrics import pka_classify_smiles, shape_metrics_smiles
    pka_classify_smiles("CN(C)CCCCN")      # ("high", "aliphatic tertiary amine")
    shape_metrics_smiles("CCCCCCCCCCCCCC(=O)NCc1ccccc1")

二、3D 形状指标（ETKDG 构象 + MMFF 优化 + 主轴长宽比）
------------------------------------------------------
目标：将"锥形/脂质样"分子与"方形/块状/球状"分子分开。
指标：
  * aspect_ratio = 最大主轴 / 最小主轴（按惯性主轴上的坐标方差半径）
  * elong_score  = 长条度（长/次长）
指标不为唯一解，作为辅助过滤而非硬性判据。
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter as _EState  # noqa: F401  (keep import neutral)

# ----------------------------------------------------------------------------
# SMARTS（残余碱性中心分档）
# ----------------------------------------------------------------------------
# 脂肪叔胺：非酰胺、非芳、非杂原子取代、3 个脂肪 C
_TER_ALIPH = Chem.MolFromSmarts(
    "[NX3;H0;!$(N-C(=O));!$(N=[!#6]);!$(N-[OX2]);!$(N-[NX3]);!$([N+]);"
    "!$(N-n);!$(N-c);$(N([CX4])([CX4])[CX4])]")
# 吡啶类：芳香 [n]（吡啶/喹啉/嘧啶/咪唑等都含 [n]）
_AROM_N = Chem.MolFromSmarts("[n;+0;!$(n[H]);!$([n+]([O-]));!$(n-[#8])]")
# 咪唑：五元环含 2 个邻位 N（其中 1 个可为 [nH]）
_IMIDAZOLE = Chem.MolFromSmarts(
    "[n]1[c,n]c[n][c]1 |u:0,1,2,3,4|")
_IMIDAZOLE2 = None  # 简化：咪唑以 2 邻位 N + 五元环识别，见下
_IMIDAZOLE_FW = Chem.MolFromSmarts("[#6]1[#7][#6][#6][#7]1")  # 广义五元环含2N
# N-喹啉/氮杂环（大体并入吡啶 low）
_MORPHOLINE = Chem.MolFromSmarts("[NX3;R;!$(N-C(=O))]1[CH2][CH2][OX2][CH2][CH2]1")
_PIPERAZINE = Chem.MolFromSmarts("[NX3;R;!$(N-C(=O))]1[CH2][CH2][NX3;R][CH2][CH2]1")
_PYRIDINE = Chem.MolFromSmarts("[n;+0;H0;v3;R]")   # 未取代的吡啶氮

# 分档优先级：高碱 → 中 → 低。
def pka_classify_smiles(smiles: str):
    """返回 (等级, 标签)。等级 high/medium/mixed/low/other。"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "other", "parse_fail"
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return "other", "sanitize_fail"
    has_ter = m.HasSubstructMatch(_TER_ALIPH)       # 脂肪叔胺 →
    has_pip = m.HasSubstructMatch(_PIPERAZINE)      # 哌嗪 →
    has_mor = m.HasSubstructMatch(_MORPHOLINE)      # 吗啉 →
    has_imid = m.HasSubstructMatch(_IMIDAZOLE)
    if not has_imid and _IMIDAZOLE_FW is not None:
        # 五元环含 2 个 N（咪唑骨架），优先级略低于精确咪唑 SMARTS
        has_imid = m.HasSubstructMatch(_IMIDAZOLE_FW)
    has_arom = m.HasSubstructMatch(_AROM_N)        # 芳 N（吡啶……）

    if has_ter:
        return "high", "aliphatic tertiary amine"
    if has_pip or has_mor:
        return "medium+", ("piperazine" if has_pip else "morpholine")
    if has_imid:
        return "medium", "imidazole"
    if has_arom:
        return "low", "aromatic N (pyridine-like)"
    return "other", "no_basic_center"


# 是否含"残余可电离碱性中心"（高或中：脂肪叔胺/哌嗪/吗啉/咪唑都算；芳香吡啶按低）
def has_ionizable_center(m, require_high=True):
    """require_high=True 只算 高/中 碱性（适合作为可电离脂质头部）；
    require_high=False 则芳香吡啶也算。"""
    if m is None:
        return False
    if require_high:
        return (m.HasSubstructMatch(_TER_ALIPH) or m.HasSubstructMatch(_PIPERAZINE)
                or m.HasSubstructMatch(_MORPHOLINE) or m.HasSubstructMatch(_IMIDAZOLE)
                or m.HasSubstructMatch(_IMIDAZOLE_FW))
    return (m.HasSubstructMatch(_TER_ALIPH) or m.HasSubstructMatch(_PIPERAZINE)
            or m.HasSubstructMatch(_MORPHOLINE) or m.HasSubstructMatch(_IMIDAZOLE)
            or m.HasSubstructMatch(_IMIDAZOLE_FW) or m.HasSubstructMatch(_AROM_N))


# ----------------------------------------------------------------------------
# 3D 形状指标
# ----------------------------------------------------------------------------
def shape_metrics_smiles(smiles: str, n_conformers: int = 20,
                         force_field: str = "mmff94"):
    """计算单个 SMILES 的 3D 形状：返回 dict 或含 'error' 键。

    流程：ETKDG(v3) 生成 n_conformers → 择优 MMFF94 最小化 → 取能量最低构象
    → 计算惯量主轴 → 主轴方差 → 长宽比。
    """
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if m is None:
        return {"error": "parse"}
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    try:
        params.maxIterations = 1000
        params.SetUseRandomCoords(True)
    except Exception:
        pass  # 兼容低版本 RDKit
    ok = AllChem.EmbedMultipleConfs(m, numConfs=max(1, n_conformers), params=params)
    if ok != 0 or m.GetNumConforms() == 0:
        # 退避：ETKDG 失败改用旧 EmbedMolecule + UFF 轻优化
        from rdkit.Chem import rdDistGeom
        r = rdDistGeom.EmbedMolecule(m, useRandomCoords=True)
        if r != 0:
            return {"error": "embed_fail"}

    # 择优：MMFF94 最小化取能量最低
    import rdkit.Chem.rdForceFieldHelpers as FF
    best_e = None
    best_conf = None
    for conf in m.GetConformers():
        try:
            e = FF.MMFFGetMoleculeForceField(m, mmffVariant=force_field, confId=conf.GetId())
            if e is not None:
                e.Minimize()
                eng = e.CalcEnergy()
            else:
                eng = None
        except Exception:
            eng = None
        if eng is not None and (best_e is None or eng < best_e):
            best_e, best_conf = eng, conf.GetId()
    if best_conf is None:
        best_conf = 0

    # 转动惯量主轴
    conf = m.GetConformer(best_conf)
    coords = [list(conf.GetAtomPosition(a.GetIdx())) for a in m.GetAtoms()]
    # 去质心
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    xs = [c[0] - cx for c in coords]
    ys = [c[1] - cy for c in coords]
    zs = [c[2] - cz for c in coords]
    # 协方差矩阵（近似惯量的主方向）
    Ixx = sum(x * x for x in xs); Iyy = sum(y * y for y in ys); Izz = sum(z * z for z in zs)
    Ixy = sum(x * y for x, y in zip(xs, ys)); Ixz = sum(x * z for x, z in zip(xs, zs))
    Iyz = sum(y * z for y, z in zip(ys, zs))
    # 特征值（对称 3x3）
    eigs = _sym_eigvals3(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    eigs.sort(reverse=True)
    rad = [max(e, 1e-6) ** 0.5 for e in eigs]  # 主轴半径（大致成比例）
    aspect = rad[0] / rad[-1] if rad[-1] > 1e-6 else float("inf")
    elong = rad[0] / rad[1] if rad[1] > 1e-6 else float("inf")
    return {
        "n_confs": m.GetNumConformers(),
        "energy_kcal": best_e,
        "aspect_ratio": aspect,
        "elong_ratio": elong,
        "radius_pc": rad[0], "radius_pc_mid": rad[1], "radius_pc_min": rad[2],
        "force_field": force_field,
    }


def _sym_eigvals3(a, b, c, xy, xz, yz):
    """对称 3x3 矩阵 |a xy xz; xy b yz; xz yz c| 的特征值（解析求解）。"""
    # 用数值特征多项式，简单可靠（自带 numpy）
    import numpy as np
    mat = np.array([[a, xy, xz],
                    [xy, b, yz],
                    [xz, yz, c]])
    return np.linalg.eigvalsh(mat).tolist()


def shape_pass(m, min_aspect=1.7):
    """3D 已给出的初步判据：长宽比 ≥ min_aspect → 偏长条/脂质样；
    过低 → 球状/块状。作为可选辅助过滤。"""
    s = shape_metrics_smiles(Chem.MolToSmiles(m)) if isinstance(m, Chem.Mol) else {}
    if not s or "error" in s:
        return False, s
    return s["aspect_ratio"] >= min_aspect, s


# ----------------------------------------------------------------------------
# 可选独立 CLI（便于单条/小批量验证，不进枚举主流程）
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        smi = line.split("\t")[0]
        gr, lab = pka_classify_smiles(smi)
        try:
            sh = shape_metrics_smiles(smi, n_conformers=5)
        except Exception as e:
            sh = {"error": str(e)}
        print(f"{smi}\tpKa_class={gr}({lab})\tshape={sh}")