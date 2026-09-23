# Problem 2 Robust emotion prediction with local modality gaps

## Task

Train on the prescribed Attachment 2 training split and select settings on its validation split. Freeze the method before inference on the unlabeled Attachment 3 data. Handle continuous local gaps within a modality and report how gap type, position and duration affect performance.

Report polarity Accuracy and F1; report intensity MAE and Pearson correlation. Keep validation metrics separate from Attachment 3 predictions. The unlabeled special test cannot support an accuracy claim or threshold selection.

## Folder plan

- `data/`: Attachment 2/3 release and sample-ID mapping. Data stays outside Git.
- `configs/`: model, objective, masks, seed and validation-only decisions.
- `scripts/`: training, validation, missingness comparisons and Attachment 3 inference.
- `outputs/`: logs, predictions and experiment summaries. Large features and weights are Git-ignored.
- The shared model code belongs in the repository's `models/` directory.

## Status

Implementation and structural checks are available. Formal GPU training, held-out evaluation and Attachment 3 predictions have not been run; no Problem 2 performance result is claimed yet.

赛题文献与实际数据接口审计后的建模路线见 [问题二三文献解读与建模路线](../docs/问题二三文献解读与建模路线.md)。附件3对齐版只提供 `text_bert`、语音、视觉，正式模型必须在附件2上按相同文本入口完成验证。

现有 SlotEmo 的代码审计、最小改造机制和 A/B/C 验证协议见 [基于SlotEmo的问题二建模方案](基于SlotEmo的问题二建模方案.md)。该文件为设计，不是已训练结果。

## Linux 运行

在仓库根目录安装 `problem2/requirements.txt`。设置 `MOSEI_DATA_ROOT` 为**同时包含附件2和附件3文件夹**的数据根目录，或在每个命令使用 `--data-root`。以下命令需在具备 CUDA 的 Linux 机器上执行；本仓库未在当前 Windows 机器上开展正式训练。

```bash
python -m pip install -r problem2/requirements.txt
python -m unittest problem2.test_pipeline -v
export MOSEI_DATA_ROOT=/path/to/E题数据
python -m problem2.scripts.run train --arm A --seed 42
python -m problem2.scripts.run train --arm B --seed 42
python -m problem2.scripts.run train --arm C --seed 42
python -m problem2.scripts.run evaluate --checkpoint problem2/outputs/arm_C_seed_42/best.pt --split valid --grid
python -m problem2.scripts.run evaluate --checkpoint problem2/outputs/arm_C_seed_42/best.pt --split test
python -m problem2.scripts.run infer --checkpoint problem2/outputs/arm_C_seed_42/best.pt
```

A 是修正输入/掩码后的完整输入基线；B 增加连续缺口训练；C 再增加同位跨模态补偿。训练集动态生成缺口，验证集按固定种子生成同一组混合缺口。`evaluate --grid` 报告 3 模态 × 3 位置 × 3 长度的性能表。正式比较应补 3 个种子，并只用 valid 选方案。`test` 仅对冻结后的最终模型运行一次。附件3推理会生成 30 行 CSV 和来源记录，不会伪造无标签样本的 F1。

每个训练目录保存运行配置、训练曲线、标准化参数和最佳权重。权重由 `.gitignore` 排除，约 20 MB；最终提交须实际检查包含代码、配置和**单个选定权重**的压缩包总量不超过 50 MB。若输出目录已有运行记录，训练脚本会拒绝覆盖。
