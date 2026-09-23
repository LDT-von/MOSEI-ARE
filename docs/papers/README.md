# 赛题参考论文（本地阅读目录）

依据项目外层的赛题 Word《复杂场景下多模态情感识别的数学建模与算法设计》末尾参考文献整理。编号 01–10 与 Word 一致；11 是此前讨论的 SlotSPE，**不属于赛题 Word 的参考文献**。下载时间：2026-09-23。这里仅存放公开可访问的论文 PDF，不包含比赛数据集。PDF 被 `.gitignore` 排除，不会随 Git 推送。

| 编号 | 论文 | 本地 PDF | 页数 | 来源与状态 |
| --- | --- | --- | ---: | --- |
| 01 | Zhang 等，Deep learning-based multimodal emotion recognition from audio, visual, and text modalities: A systematic review of recent advancements and future prospects，*Expert Systems with Applications*，2024 | — | — | [出版社页面](https://www.sciencedirect.com/science/article/pii/S0957417423021942)，需机构访问或购买；未找到可核实的公开作者稿，未下载 |
| 02 | Qiu 等，Beyond Missing Modalities，CVPR 2026 | [PDF](02_Beyond_Missing_Modalities_CVPR2026.pdf) | 11 | [CVF 开放版](https://openaccess.thecvf.com/content/CVPR2026/papers/Qiu_Beyond_Missing_Modalities_Hypergraph_Conditioned_Diffusion_for_Uncertainty-Aware_Multimodal_Emotion_CVPR_2026_paper.pdf) |
| 03 | Yang 与 Li，Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis，标为 CVPR 2026 | — | — | [搜索结果指向的 CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Factorize_Reconstruct_Enhance_A_Unified_Framework_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf) 当前返回 404；未找到可核实的作者开放版，未下载 |
| 04 | Zhuang 等，CMAD，ICCV 2025 | [PDF](04_CMAD_ICCV2025.pdf) | 11 | [CVF 开放版](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf) |
| 05 | Zhu 等，Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data，ACL 2025 | [PDF](05_Proxy_Driven_Robust_MSA_ACL2025.pdf) | 16 | [ACL Anthology](https://aclanthology.org/2025.acl-long.1075.pdf) |
| 06 | Mai 与 Han，Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective，ACL 2026 | [PDF](06_Learning_Invariant_Modality_Representation_ACL2026.pdf) | 24 | [ACL Anthology](https://aclanthology.org/2026.acl-long.2119.pdf) |
| 07 | Fang 等，EMOE: Modality-Specific Enhanced Dynamic Emotion Experts，CVPR 2025 | [PDF](07_EMOE_CVPR2025.pdf) | 11 | [CVF 开放版](https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.pdf) |
| 08 | Wan 等，Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation，ACL 2026 | [PDF](08_Locate_and_Explain_ACL2026.pdf) | 18 | [ACL Anthology](https://aclanthology.org/2026.acl-long.2012.pdf) |
| 09 | Mai 与 Han，CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion，CVPR 2026 | [PDF](09_CaReFlow_CVPR2026.pdf) | 11 | [CVF 开放版](https://openaccess.thecvf.com/content/CVPR2026/papers/Mai_CaReFlow_Cyclic_Adaptive_Rectified_Flow_for_Multimodal_Fusion_CVPR_2026_paper.pdf) |
| 10 | 王楠等，基于知识蒸馏与动态调整机制的多模态情感分析模型，《计算机学报》，2025 | [PDF](10_AUMDF_CJC2025.pdf) | 20 | [高校图书馆提供的期刊 PDF](https://lib.zjsru.edu.cn/25-10.11-4.pdf)；[期刊原站链接](https://cjc.ict.ac.cn/online/onlinepaper/wn-2025818172921.pdf) 当时连接失败 |
| 11 | Zhang 等，Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis（SlotSPE），ICLR 2026，额外背景 | [PDF](11_SlotSPE_ICLR2026.pdf) | 37 | [arXiv 作者稿](https://arxiv.org/pdf/2512.01116) |

**书目信息核对：**赛题 Word 把 02 的副标题写作 “Hypergraph Conditioned Diffusion”，但所下载的 CVF PDF 首页写作 “Hypergraph Guided Diffusion”。引用时应以论文 PDF 首页与正式会议记录为准，避免把两个写法混用。03 的 PDF 链接虽可被搜索引擎索引，但下载验证失败；在取得原文前不要据二手摘要引用其中实验数字。

`download_manifest.json` 记录每份已下载 PDF 的原始 URL、字节数和 SHA-256。需要重新下载时，在项目根目录运行：

```powershell
python docs/papers/download_papers.py
```

问题一报告里的 PyTorch、FFmpeg、TorchVision、Hugging Face 链接是工具或模型文档，不属于上述赛题研究论文清单。
