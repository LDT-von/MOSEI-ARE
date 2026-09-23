# 问题一 特征提取与时序对齐

先阅读 [问题一建模与结果](问题一建模与结果.md)。这是从附件1原视频实际运行得到的流程，包含100份特征与来源记录。79条样本通过自动对齐筛查；另外21条完整保留，默认时序文本被屏蔽，候选映射留作审计。筛查通过不等于人工确认的对齐正确率。

原始附件、逐样本输入清单、特征张量、转写与逐样本审计文件不会纳入 Git。下方文件清单描述本地完整运行产物；在新克隆中先按“从原视频复现”生成后再读取。仓库保留实现代码和聚合质量、运行及附件包清单。

## 主要文件

- `outputs/features/`：100份NPZ，包含原生及对齐后的三模态特征。
- `outputs/summary_100.csv`：全量结果、时长、维度、有效位置数和筛查状态。
- `outputs/metadata/`：100份来源JSON，包括原文、词时间、实际帧号、裁剪框和哈希。
- `outputs/example/`：典型样本时间对应图与逐位置记录。图中文字块是候选对齐。
- `outputs/alignment_review.csv`：需人工复核的词及候选时间。
- `outputs/quality_report.json`：结构、数值与重建验证结果。
- `outputs/media_audit.json`：MP4播放与编码载荷时间差异、逐声道零信号审计。
- `outputs/independent_asr_audit.json`：四条样本的独立自动语音识别抽查。
- `outputs/run_manifest.json`、`environment.json`、`processing_log.jsonl`：版本、参数、权重哈希和逐样本日志。

## 使用已有结果

```python
from pathlib import Path
import numpy as np

path = next(Path('outputs/features').glob('*.npz'))
with np.load(path, allow_pickle=False) as d:
    print(d['grid_intervals'].shape)
    print(d['text'].shape, d['audio'].shape, d['vision'].shape)
    valid_text = d['text_mask']
```

`load_features.py` 提供右侧填充的批次读取示例。维度为384/33/512，属于问题一自生成接口，不能直接输入使用附件2之768/74/35维的旧SlotEmo模型。问题二、三仍须遵守附件2与专项测试集的统一接口要求。

## 从原视频复现

默认目录是 `题目工作目录\MOSEI-ARE\problem1`；原始附件留在题目工作目录下的 `E题数据\E题数据`。输入清单相对于题目工作目录；移动附件后先传入新的题目目录重建清单。使用Python 3.10的独立环境安装 `requirements.txt`。原运行使用CUDA 12.4版PyTorch；从PyTorch官方安装源选择与硬件匹配的构建。

```text
python inventory.py --workspace <题目工作目录>
python prepare_models.py
python pipeline.py --workspace <题目工作目录>
python apply_quality_gate.py
python verify_outputs.py
python audit_media.py
python test_alignment.py
python load_features.py
python build_report.py
python package_results.py
```

`prepare_models.py` 只下载公开预训练权重，不下载额外情感数据集。语音和视觉权重保存至assets，文本模型使用Hugging Face缓存。`audit_transcripts.py` 是可选的独立语音诊断，另需本地 `openai/whisper-small`，修订为 `973afd24965f72e36ca33b3055d56a652f456b4d`；如无缓存，可用 `huggingface_hub.snapshot_download` 获取该修订后再运行。四条原运行诊断输出已包含在本地交付包中；在新副本里要重现这份诊断文件，需在打包前补跑该脚本。它不影响主提取和质量筛查。

本地工作目录现已生成完整100份特征；只需核验现有结果时运行 `python verify_outputs.py`，无需重复提取。从头复现实验请在新的项目副本中执行上面的命令，避免覆盖当前交付结果。提取中断后、且 `pipeline.py` 内容未变时，可以在同一副本追加 `--resume`；脚本会核对源码和特征哈希，源码变化时会重新提取。

原视频和标签不做回写。`--limit` 仅用于调试，样本数不足100时最终验证会拒绝通过。全部规则与参数固定在脚本和运行清单，未利用情感标签选择阈值。

只想检查原视频提取能否启动时，在 `problem1` 目录执行 `python smoke_check.py --device cpu`。它在 `qa/` 下的临时副本重新提取一条真实视频、应用质量筛查并检查三模态输出，使用已有权重缓存且保留现有100份结果不变。检查报告写入 `qa/smoke_report.json`；这不是人工对齐准确率。

## 核验与附件范围

`test_alignment.py` 检查非整倍尾段、区间权重、时间平移、文字来源、真实静音和转写匹配。`verify_outputs.py` 对100份实际结果核验来源哈希、原文和标签、数值、有效位置、词时间顺序，并从原生特征独立重建默认汇总。

打包脚本从 `problem1` 生成第一题附件包，收录代码、说明、100份特征与追溯证据，排除预训练大权重、原视频、缓存、临时日志和本机身份路径。50MB是整份竞赛附件的上限，还需给问题二、三留出空间。准确大小与SHA-256记录在仓库根目录的 `problem1/outputs/package_manifest.json`。

验证范围是时间组织与实现核验。没有人工词边界真值，不报告词边界准确率；没有运行情感预测性能评估，也不能解释为F1改进。

## 检查真实对齐效果

`evaluate_quality.py` 已在本地 `qa/` 生成分层抽取的30段视频、90个词的人工复核表。先在 `qa/manual_review_clips.csv` 核对音画与转写；再听视频，在 `qa/manual_review_words.csv` 填写词是否可听到及人工起止秒数。完成后运行 `python evaluate_quality.py score`，会生成 `qa/manual_quality_report.json`，按自动通过高分、自动通过较低分、被屏蔽三组报告词边界误差、0.2秒时间格重叠、可用词保留和不可匹配词拒绝情况，并与等分时长基线比较。未填写人工时间前，这些指标没有真实测量值。`qa/` 不进入提交包。
