# MOSEI-ARE

**CMU-MOSEI 多模态情感建模项目**，对应题目三项主线：时序对齐（Alignment）、缺失鲁棒性（Robustness）和可解释性（Explainability）。

## 三问产出总览

| 问题 | 核心任务 | 状态 | 主要提交文件 |
|------|---------|------|-------------|
| **问题一** | 附件1 视频特征提取与时序对齐 | ✅ 已完结 | `problem1/问题一建模与结果.md` + 100 条特征 |
| **问题二** | 附件2/3 连续缺失鲁棒情感预测 | ✅ 已完结 | 三组权重 + 附件3预测 CSV |
| **问题三** | 附件2/4 可解释预测与证据溯源 | ✅ 已完结 | 附件4预测+证据CSV/JSON |

## 目录结构

| 目录 | 用途 |
|------|------|
| `problem1/` | 附件1特征提取、时序对齐（已完结） |
| `problem2/` | 问题二训练、评估、附件3推理 |
| `problem3/` | 问题三可解释预测、附件4推理 |
| `models/` | GapSlotEmo 共用模型定义 |
| `configs/` | 共享实验配置 |
| `data/` | 数据位置与版本约定 |
| `docs/` | 建模决策文档 |
| `legacy/` | 历史 SlotEmo 实验（不作为正式提交） |

## 快速复现

**前提**：Python 3.10+，PyTorch 2.x，CUDA 11.8/12.x，NVIDIA GPU（≥6 GB 显存）。`MOSEI_DATA_ROOT` 指向包含附件 1-4 的数据根目录。

### 问题一（仅 CPU 特征提取）

```powershell
cd "C:/Users/栋栋/Desktop/E题/MOSEI-ARE"
python problem1/exp3_problem1.py  # 生成 100 条特征
```

### 问题二

```powershell
$env:MOSEI_DATA_ROOT = "C:\Users\栋栋\Desktop\E题\E题数据\E题数据"

# 训练三组（Arm A / B / C）
python -m problem2.scripts.run train --arm A --seed 42 --device cuda --output problem2/outputs/arm_A_seed_42
python -m problem2.scripts.run train --arm B --seed 42 --device cuda --output problem2/outputs/arm_B_seed_42
python -m problem2.scripts.run train --arm C --seed 42 --device cuda --output problem2/outputs/arm_C_seed_42

# 验证（含 27 条件 grid）
python -m problem2.scripts.run evaluate --checkpoint problem2/outputs/arm_C_seed_42/best.pt --split valid --grid --device cuda

# 附件3推理
python -m problem2.scripts.run infer --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda
```

### 问题三

```powershell
# 全量指标（快速）
python -m problem3.scripts.run evaluate-valid --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda
python -m problem3.scripts.run evaluate-test  --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda

# 子集证据扫描
python -m problem3.scripts.run evidence-valid --checkpoint problem2/outputs/arm_C_seed_42/best.pt --sample-size 32 --device cuda
python -m problem3.scripts.run evidence-test  --checkpoint problem2/outputs/arm_C_seed_42/best.pt --sample-size 32 --device cuda

# 附件4推理（含证据）
python -m problem3.scripts.run infer --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda --window-length 6 --window-top-k 3

# 汇总报告
python problem3/scripts/summarize.py

# 单元测试
python -m unittest problem3.test_pipeline -v
```

## 提交文件清单

### 问题一
- `problem1/问题一建模与结果.md` —— 完整方法与结果报告

### 问题二（位于 `problem2/outputs/`）

| 文件 | 说明 |
|------|------|
| `arm_{A,B,C}_seed_42/best.pt` | 模型权重（约 19.6–19.9 MB） |
| `arm_{A,B,C}_seed_42/attachment3_predictions.csv` | **附件3提交文件**，30 条预测 |
| `arm_C_seed_42/valid_evaluation.json` | 728 条验证，含 27 grid |
| `arm_C_seed_42/test_evaluation.json` | 727 条测试指标 |
| `arm_{A,B,C}_seed_42/history.json` | 训练曲线 |
| `arm_{A,B,C}_seed_42/问题二建模与结果.md` | 建模与结果报告 |

### 问题三（位于 `problem2/outputs/arm_C_seed_42/`）

| 文件 | 说明 |
|------|------|
| `problem3_attachment4/attachment4_predictions.csv` | **附件4提交文件**，20 条预测+证据 |
| `problem3_attachment4/attachment4_evidence.json` | 完整证据 JSON |
| `problem3_attachment4/attachment4_provenance.json` | 溯源信息 |
| `problem3_valid_evaluation.json` | 728 条验证指标 |
| `problem3_test_evaluation.json` | 727 条测试指标 |
| `problem3_valid_evidence.json` | 32 条验证证据 |
| `problem3_test_evidence.json` | 32 条测试证据 |
| `problem3/问题三建模与结果.md` | 建模与结果报告 |

## 核心指标汇总

### 问题二（Arm C，seed=42）

| Split | Accuracy | F1-macro | MAE | Pearson |
|-------|----------|-----------|-----|---------|
| valid clean | 0.473 | 0.442 | 0.760 | 0.331 |
| valid mixed | 0.486 | 0.464 | 0.756 | 0.322 |
| test clean | 0.492 | 0.449 | 0.839 | 0.393 |
| test mixed | 0.490 | 0.446 | 0.845 | 0.368 |

### 问题三（附件4，20 条，无标签）

预测分布：负 6 / 中 4 / 正 10；强度范围 [-1.69, +0.96]；主控模态 audio/vision/text 各约 1/3。

## 数据位置

原始附件保留在库外（**不纳入 Git**）：
- `MOSEI_DATA_ROOT/附件1-数据集原始音视频/` —— 100 条原视频
- `MOSEI_DATA_ROOT/附件2-数据集特征文件/` —— `aligned_50.pkl`、`label.xlsx`、`unaligned_50.pkl`
- `MOSEI_DATA_ROOT/附件3-模态缺失特征样本/`
- `MOSEI_DATA_ROOT/附件4-可解释专用视频和特征文件/`

详见 `data/README.md`。Git 只保存代码、说明文档和聚合评估报告。

## 打包交付

```powershell
# 生成全部三问交付压缩包（不含原始视频和模型权重）
python problem1/package_results.py   # 问题一
# 问题二/三：将 outputs/ 下文件整理为交付目录后压缩
```

竞赛附件合计须低于 50 MB；`problem2/outputs/` 下的 `.pt` 权重文件约 20 MB 可酌情不打包（仅供复现），预测 CSV 和证据 JSON 均很小。
