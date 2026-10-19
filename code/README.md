# LipidFM —— 可电离脂质预训练大模型（Ionizable-Lipid Foundation Model）

基于 [Uni-Mol2](https://github.com/dptech-corp/Uni-Mol) 对 **可电离脂质 (ionizable lipids, ILs)** 做
域自适应继续预训练（domain-adaptive continued pretraining），产出面向 LNP（脂质纳米颗粒）下游性质预测
（转染效率等）的 3D 分子骨架。

**核心流水线（三阶段）：**

1. **虚拟库构建**：多源文献可电离脂质 + 组合化学枚举，构造约 **11.2 万**个可电离脂质分子
   （真实 12,381 + 虚拟 100,000）。
2. **3D 构象生成 + 继续预训练**：对语料生成 Uni-Mol2 兼容的 3D 构象，用 `unicore-train` 对 84M
   Uni-Mol2 底座继续预训练 30k 步，得到 **LipidFM** checkpoint。
3. **下游微调与评测**：用 unimol_tools 的 `MolTrain(task='regression', …)` 在两个转染率数据集上做
   端到端微调，与**未微调的原始 84M 底座（baseline）**对比。

> 本 README 面向**复现**编写：按顺序执行即可重现从原始数据到下游指标的全流程。

---

## 0. 获取代码、数据与权重

GitHub 仓库只包含源码、配置和文档。数据、预训练底座和 LipidFM checkpoint 从 Hugging
Face 获取，并保持与本仓库相同的相对路径：

```bash
# 在仓库根目录执行；<HF_REPO_ID> 例如 your-name/LipidFM-assets
python tools/download_hf_assets.py --repo-id <HF_REPO_ID>
```

资产目录、最小资产集和发布建议见仓库根目录的 [`ASSETS.md`](../ASSETS.md)；快速入口见
[`README.md`](../README.md)。

建议从仓库根目录安装主环境：

```bash
bash install_env.sh
conda activate unimol2
python -c "import unicore, unimol_tools, torch, rdkit; print('environment OK')"
```

`install_env.sh` 默认在当前 Conda 安装目录下创建 `unimol2`；需要自定义位置时设置
`CONDA_ENV_PREFIX=/path/to/env`。

---

## 1. 环境与依赖

LipidFM 主线使用同一个 Python 环境：

| 用途 | 环境 | 关键包 |
|---|---|---|
| 阶段 0 虚拟库构建 | `conda activate unimol2` | RDKit, pandas, joblib, tqdm, pyarrow |
| 阶段 1 构象生成 + 继续预训练 | 同一 `unimol2` 环境 | unicore, torch 2.13 |
| 阶段 2 下游 MolTrain 微调 | 同一 `unimol2` 环境 | unimol_tools, torch, RDKit |

实测确认：`unimol2` conda 环境同时具备 `unicore`、`unimol_tools`、`torch`、`rdkit`，阶段 2（unicore-train）
与阶段 3（MolTrain）都在该环境下运行即可。

```bash
conda activate unimol2
# 验证关键依赖（非必需，但建议）
python -c "import unicore, unimol_tools, torch, rdkit; print('OK')"
```

> 仓库 `requirements.txt` 是**全部**阶段共用依赖的近似清单；实测环境为
> `torch 2.13.0+cu130 / rdkit 2026.03.5 / unicore / unimol_tools`。

### 原始数据（阶段 0 输入）

| 文件 | 位置 |
|---|---|
| 5 个来源原始 CSV（LNPDB / LNP_Atlas / paper_ABSP / paper_MTCS / paper_umi-lab） | `00_prepare_ILs_vitual_library/raw_data/`（可用 `LNP_RAW_DIR` 覆盖） |
| ChEMBL 37 反遗忘回放数据 | `01_continous_training/1_data/raw/CHEMBLdb/chembl_37_chemreps.txt.gz` |

### 预训练底座权重（阶段 2 起点）

| 文件 | 位置 |
|---|---|
| Uni-Mol2 84M / 1.1B 底座 | `01_continous_training/2_models/pretrained/checkpoint_84M.pt`、`checkpoint_1.1B.pt`、`dict.txt` |

> ⚠️ 底座权重与训练产出的 LipidFM checkpoint 都是 **GB 级二进制**，**不随源码分发**，需自行准备。

---

## 2. 阶段 0 —— 构建可电离脂质虚拟库

目录：`00_prepare_ILs_vitual_library/`

**运行顺序**（每个脚本独立可跑，输出落在 `output/`）：

```bash
cd 00_prepare_ILs_vitual_library
python 00_load_and_merge_raw.py    # 多源合并去重 → 真实可电离脂质语料 + Ugi 砌块库
python 01_build_reagent_library.py # 提取 head(多胺)/tail(亲电试剂) 试剂库
python 02_augment_reagents.py      # 尾巴沿链长/双键/支化规则化扩增
python 03_enumerate_library.py     # 组合化学枚举（约 196 万组合，20 核约 2 分钟）
python 04_filter_and_merge.py      # 质控 + 去泄漏 + 反应类型配额均衡 → 预训练语料
```

**产物**（`output/`，阶段 1 的输入）：

| 文件 | 内容 |
|---|---|
| `ionizable_lipids.parquet` | 真实可电离脂质（含活性标签） |
| `pretrain_corpus.parquet` | **最终预训练语料**：真实 12,381 + 虚拟 100,000（`source_type` 列区分） |
| `merge_report.md` / `qc_report.md` | 数据来源贡献、去重率、性质分布、过滤原因统计 |

**关键设计（务必了解，勿随意改动阈值）：**

- **可电离判据**在 LNPDB 的 12,844 个真实 IL 上标定：召回 88.33%、非脂质负样本假阳仅 3.10%
  （见 `config.py` 的 SMARTS 与阈值注释）。判据顺序：磷脂 → 可质子化胺 → 疏水性(logP) → 疏水质量(脂肪碳数) → 脂链长度。
- **反应类型配额均衡**（`04_filter_and_merge.py`）：枚举产物 76% 来自还原胺化，直接使用会让模型过拟合单一化学，
  故按 `RXN_QUOTA` 下采样。
- **去泄漏**：枚举产物与真实集按 InChIKey 比对剔除，防止下游微调时训练集污染测试集。
- **构象采样机制**：Uni-Mol2 的 `ConformerSampleDataset` 每个 epoch 每个分子只取 1 个构象，因此构象数
  只做 **3D 数据增强**、不增加样本数；真实:虚拟有效样本比 1:8.08 由分子数决定（见 `config.py` 说明）。

---

## 3. 阶段 1 —— 3D 构象生成 + 继续预训练（产出 LipidFM）

目录：`01_continous_training/`。所有参数集中在 `3_configs/pipeline.yaml`。

### 3.1 生成 3D 构象（Step 1）

```bash
cd 01_continous_training
python main.py --step prepare            # 全量
# 冒烟测试（小样本）：
python main.py --step prepare --set prepare.sample_size=60 prepare.chembl_max=15 \
      prepare.output_dir=1_data/processed/smoke
# 断点续跑：
python main.py --step prepare --set prepare.resume=true
```

输出到 `1_data/processed/ionizable_lipids/unimolv2_3d/`：`lipids_unimolv2_3d.sdf`（构象）、
`lipids_unimolv2_summary.json`（统计）、`lipids_unimolv2_checkpoint.json`（断点状态）、失败清单 CSV。

> **三条关键适配**（针对脂质 >60 重原子、~39 个可旋转键的柔性）：`random_coords=true`（否则长链嵌合失败）、
> `max_iterations=2000`（MMFF94s 收敛）、`max_atoms=256`。详见该脚本 docstring 与 `pipeline.yaml` 注释，
> **不可回退到源项目默认值**——它们会静默丢弃大部分脂质。

### 3.2 继续预训练（Step 2）

LMDB 在首次运行时自动构建（`build_lipids_pretrain_lmdb.py`），无需手动。

```bash
# 全量训练（84M，30k 步）
python main.py --step pretrain
# 仅打印 unicore-train 命令不启动：
python main.py --step pretrain --set pretrain.dry_run=true
# 1.1B 模型：
python main.py --step pretrain --set pretrain.model_size=1.1b \
      pretrain.pretrained_checkpoint=2_models/pretrained/checkpoint_1.1B.pt \
      pretrain.output_dir=2_models/continued_pretrain/unimol2_lipids_1.1b
```

- checkpoint 落在 `<output_dir>/checkpoints/`。**LipidFM v2（本文档主线）**：
  `2_models/continued_pretrain/unimol2_lipids_84m_v2/checkpoints/checkpoint_last.pt`
  （完整数据 133,181 分子训练，valid loss 0.0515，masked_acc 99.8%）。
- `main.py` 内置守卫：`prepare.max_atoms > pretrain.max_atoms` 时报错退出，避免
  Uni-Mol2 `CroppingDataset` 静默随机丢弃原子（否则分子被破坏且不报错）。

### 3.3 硬件

单张 RTX 4090（48 GB）。有效 batch = `batch_size × update_freq`（默认 `8 × 4 = 32`）；
1.1B 需减小 `batch_size`、增大 `update_freq` 以适配显存。

---

## 4. 阶段 2 —— 下游微调与 baseline 对比（MolTrain）

目录：`01_continous_training/4_scripts/finetune_moltrain.py`。

**两个下游数据集**（`benchmark/input_data/dataset.xlsx`）：

| 数据集 | sheet | 标签列 | 样本数 | 标签尺度 |
|---|---|---|---|---|
| lumi-lab | Sheet2 | `RLU (log2)` | 1,920 | 已是 log 尺度 |
| Ours（自有） | Sheet3 | `transfection_efficiency` | 768 | **原始 RLU**（2e3 ~ 1e8，跨 4 个量级） |

> **⚠️ 标签尺度不一致是核心对比陷阱**：Sheet2 是 log2、Sheet3 是原始 RLU。`finetune_moltrain.py`
> 对 Sheet3 **先做 log10**，使两者都在对数尺度上可比。最终指标统一按 **建模尺度** 汇报：
> Sheet2 = log2(RLU)、Sheet3 = log10(RLU)。R²/Spearman 无量纲可跨数据集比；**RMSE 的绝对量纲仍不同**，
> 跨数据集比 RMSE 无意义。

### 4.1 拆分策略（数据集划分）

`finetune_moltrain.py` 通过 `--split {random,scaffold,butina}` 选择 CV 拆分，**默认 `scaffold`**：

- **random**：KFold 随机拆分。**仅用于 baseline 对比**，不作为最终性能报告口径——随机拆分会让
  结构相似（甚至同核）分子落在 train/test 两侧，产生泄漏、高估泛化。
- **scaffold（默认，主评方式）**：基于 **Bemis–Murcko 骨架** 的化学多样性拆分（`chem_splits.py`）。
  共享同一核心骨架的分子被分在同一折，**同一骨架绝不跨 train/test**。无环脂质（骨架为空）每个分子
  单独成组，避免误合并。组按规模贪心均衡分配到各折。
- **butina**：基于 **ECFP4 指纹 + Tanimoto 相似度**（`--butina-cutoff`，默认 **0.6**）做 `Butina`
  聚类，再按簇均衡分折。用于评估模型在**化学差异簇**上的鲁棒性。实现：纯 RDKit
  （`4_scripts/chem_splits.py`，无需 DeepChem）。

**拆分产物**：每折的 train/test **索引与数据** 落在
`5_outputs/finetune_moltrain/splits/<dataset>_<split>/`（`*_train.csv` / `*_test.csv`，含
`row_index/SMILES/TARGET/group` 四列 + 摘要 JSON），可审计、可复现、可直接喂给训练。

**防泄漏 / 保多样性自检**（`4_scripts/chem_splits.py` 的 `--split-only` 模式已跑通）：
每折 train+test = 全量样本；各折 test 互斥；且**同一 scaffold/butina 簇不出现在同一折的 train 与 test**。

> **小样本注意**：Sheet3 仅 768 样本、化学簇很少（butina 0.6 下仅约 10 簇），第 5 折可能占
> 半数样本（439/768）。butina 更适合作为**每折 R² 分布**的鲁棒性诊断，而非单一 headline 数。

### 4.2 运行（微调 LipidFM v2 + baseline + 按拆分评估）

```bash
cd 01_continous_training

# ① 先只生成/检查拆分（跳过昂贵的训练）：
python 4_scripts/finetune_moltrain.py --dataset all --split scaffold --kfold 5 --split-only
python 4_scripts/finetune_moltrain.py --dataset all --split butina  --kfold 5 --split-only

# ② 用 LipidFM v2 checkpoint 微调（默认 scaffold 拆分 = 最终性能口径）
python 4_scripts/finetune_moltrain.py --dataset all --split scaffold --epochs 20 --lr 1e-4 --kfold 5

# ③ 化学簇鲁棒性评估（butina）
python 4_scripts/finetune_moltrain.py --dataset all --split butina  --epochs 20 --lr 1e-4 --kfold 5

# ④ baseline（官方 84M 底座，仅作对比，可用随机拆分）
USE_STOCK_84M=1 python 4_scripts/finetune_moltrain.py --dataset all --split random --epochs 20 --lr 1e-4 --kfold 5
```

- 结果写 `5_outputs/finetune_moltrain/moltrain_results_<时间戳>.json`，同时聚合进
  `moltrain_results.json`（按 `model:dataset` 分键，LipidFM 与 stock-84M 各一栏，不互相覆盖）。
- `USE_STOCK_84M=1` 时不安装 monkey-patch，MolTrain 走官方底座加载；`LIPIDFM_CHECKPOINT` 可覆盖
  checkpoint 路径（默认 `checkpoint_best.pt`）。
- `--butina-cutoff` 调相似度阈值；`--split-only` 只写拆分不训练（省时核验）。

**实现要点（理解后才能正确复现/判读）：**

- `MolTrain(task='regression', metrics='r2', target_normalize='standard')` 是**真正的端到端微调**
  （SMILES → 实时构象 → backbone + 回归头联合训练，5 折 CV）。
- **weight 加载**：MolTrain 默认 `load_pretrained=True` 并自动下载官方底座权重。`finetune_moltrain.py`
  用 monkey-patch 把 `UniMolV2Model.load_pretrained_weights` 替换为加载我们的 LipidFM checkpoint
  （`strict=False`，跳过回归头）。**关键修复**：该 monkey-patch 只在我们构造模型时加载 checkpoint；
  每折训练后重载 `model_{fold}.pth` 会**放行给官方加载器**。早期版本每次都回读原始 checkpoint、
  丢弃已微调的折权重，导致 R² 崩塌到 -120~-2e100（见 `#05-修复错误`）。跑 baseline 用官方底座即可。
- **反归一化**：`MolTrain.fit()` 的 `cv_pred`/`data['target']` 实际在 **标准化（z-score）尺度**。
  脚本用 `target_scaler.inverse_transform` 把两者都拉回**建模尺度**再算指标；**不要**误当 cv_pred 已在
  建模尺度而跳过反归一化（会把 Sheet3 的 z-score 当 log10 直接 `10**`，得到 -5e10 的假 R²）。
- **scaffold/butina 拆分接入**：`_install_chem_splitter()` monkey-patch 让 MolTrain 内部用
  `chem_splits` 算出的同一套折（与 `--split-only` 保存的完全一致，无漂移）。

### 复现目录一览

```
01_continous_training/
├── main.py                     # 阶段 1 流水线入口（prepare / pretrain）
├── 3_configs/pipeline.yaml     # 阶段 1 全部参数
├── 4_scripts/
│   ├── prepare_lipids_unimol_dataset.py   # Step1: 3D 构象 + ChEMBL 回放
│   ├── build_lipids_pretrain_lmdb.py      # Step2a: SDF → LMDB
│   ├── run_lipids_continue_pretrain.py    # Step2b: unicore-train 继续预训练
│   ├── chem_splits.py                     # ★ scaffold/butina 拆分（纯 RDKit）
│   ├── finetune_moltrain.py               # ★ 下游微调主线（MolTrain，含拆分）
│   └── finetune_lipids_downstream.py      # [遗留] 冻结特征 linear/MLP 探针
├── 1_data/ 2_models/ 5_outputs/
│   └── 5_outputs/finetune_moltrain/splits/   # 每折 train/test 索引 + 组标签
├── logs/ error/
└── 00_prepare_ILs_vitual_library/        # 阶段 0（见上）
```

---

## 5. 遗留 / 早期探索脚本（仅供参考，**非复现主线**）

以下脚本是开发早期探路代码，功能已被正式主线取代，**保留但勿作为复现路径**：

| 脚本 | 状态 | 说明 |
|---|---|---|
| `my_code/01_explore_data.py`、`02_train_models.py`、`04_predict.py` | 遗留 | 1D-QSAR / 2D-指纹 / UniMol2 特征 + 经典 ML（RF/XGB/LightGBM…）的早期对比；`04_predict.py` 的 `best_model.joblib` 分支为死代码（无产物） |
| `01_continous_training/4_scripts/finetune_lipids_downstream.py` | **废弃** | 其 `full_finetune` 分支**名不副实**：名义是端到端，实际是冻结 backbone + 在预提取特征上训 2 层 MLP（≈线性探针的 MLP 版），无法发挥完整数据 continue pretrain 的优势。真正的端到端微调见 `finetune_moltrain.py` |
| `01_continous_training/4_scripts/benchmark_lipids.py` | 遗留 | Ridge/RF 作用于冻结 [CLS] 表示，属**冻结特征评估**，与 MolTrain 端到端微调**不同口径**，不可直接混比 |

---

## 6. 已知指标口径与注意事项

1. **MolTrain 结果 vs 冻结特征基准**：`benchmark_lipids.py`/linear-probe 的 R² 是冻结 backbone 表示 + 简单头；
   `finetune_moltrain.py` 是 backbone 联合微调。两者**不可直接对比**，报告中须分别注明。
2. **Sheet3 需 log10**：否则 4 个量级的原始 RLU 会让 RMSE 巨大、且 R² 主要被少数超大值主导。
   主线已在 `finetune_moltrain.py` 落地 log10，且指标统一在 log10(RLU) 建模尺度上报。
3. **checkpoint 未分发**：LipidFM checkpoint（~963 MB）与底座权重均为本地二进制，复现需自备。
4. **标签冲突字段**：真实语料 `activity_fold_range > 10` 的分子标签不可靠（见 `merge_report.md`），
   若用于回归建议剔除或降权。
5. **两大已修复的脚本坑**（勿回退）：
   - **weight 加载被 monkey-patch 吞掉**：`load_pretrained_weights` 需区分「构造时加载我们的
     checkpoint」与「每折训练后重载 `model_{fold}.pth`」。若后者也被 patch 成回读原始 checkpoint，
     微调权重全部丢弃，R² 崩塌到 -120~-2e100。
   - **反归一化尺度**：`cv_pred`/`target` 在 z-score 尺度，须用 `target_scaler` 反归一化；直接把
     z-score 当 log10 做 `10**` 会得到 -5e10 的假 R²。
6. **化学拆分**：最终性能以 **scaffold 拆分**（主）与 **butina 拆分**（鲁棒性诊断）为准；random 仅
   供 baseline 对比。见 §4.1。

---

## 7. 下一步方向（next session）

**目标：构建 LipidFM v3，检验「规模定律（scaling law）」是否在 LipidFM 上成立。**

- **假设**：继续预训练语料越大（更多可电离脂质），下游微调性能越好（scaling law）。
- **路径**：在 `01_continous_training/` 用**更多可电离脂质**继续预训练 84M（或对比更大底座），产出
  **LipidFM v3**，再用 §4 的 scaffold/butina 拆分在 Sheet2/Sheet3 上端到端微调，与 v2 及 stock-84M 对比。
- 复用：阶段 0 虚拟库构建（`00_prepare_ILs_vitual_library/`，可加新文献/枚举源）、阶段 1
  `main.py --step prepare/pretrain`、阶段 2 `finetune_moltrain.py`（已含化学拆分）。
- 对比口径固定为 §4.1 的 scaffold/butina 拆分 + 建模尺度指标，避免跨模型比较时口径漂移。

---

## 8. 快速核验清单（复现自检）

- [ ] `conda activate unimol2` 下 `import unicore, unimol_tools` 均成功
- [ ] 阶段 0 产物 `pretrain_corpus.parquet` 存在且含 `source_type`（real/virtual 各约 12k / 100k）
- [ ] 阶段 1 SDF 已生成，`lipids_unimolv2_summary.json` 无大面积失败
- [ ] `2_models/pretrained/checkpoint_84M.pt` 与 LipidFM `checkpoint_last.pt` 存在
- [ ] `chem_splits.py` 自检通过（`python 4_scripts/chem_splits.py` → ALL CHECKS PASSED）
- [ ] `--split-only` 能对 scaffold/butina 生成 `5_outputs/finetune_moltrain/splits/*.csv` 且各折
      train+test=全量、test 互斥、同组不跨 train/test
- [ ] 下游微调能分别对 LipidFM v2 与 baseline 各产出一份 `moltrain_results_*.json`（含 scaffold 拆分）
