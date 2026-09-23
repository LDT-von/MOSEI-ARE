"""
SlotEmo 配置模块
基于SlotSPE论文思想改造的多模态情感预测模型配置
"""
import os
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DATA_ROOT = os.path.abspath(os.environ.get(
    'MOSEI_DATA_ROOT', str(REPO_ROOT.parent / 'E题数据' / 'E题数据')
))
ATTACHMENT1_PATH = os.path.join(DATA_ROOT, '附件1-数据集原始多模态样本', 'MOSEI数据集部分原始视频-100条')
ATTACHMENT2_PATH = os.path.join(DATA_ROOT, '附件2-数据集特征文件')
ATTACHMENT3_PATH = os.path.join(DATA_ROOT, '附件3-模态缺失特征样本')
ATTACHMENT4_PATH = os.path.join(DATA_ROOT, '附件4-可解释专项视频样本与特征文件', '附件4-可解释专项视频样本与特征文件')

# 特征版本: 'aligned' 或 'unaligned'
FEATURE_VERSION = 'aligned'

# =====================
# 模型配置 (基于SlotSPE)
# =====================
MODEL_CONFIG = {
    # 各模态原始特征维度
    'text_dim': 768,
    'audio_dim': 74,
    'vision_dim': 35,

    # 投影维度 (与SlotSPE一致)
    'projection_dim': 128,

    # Slot Attention 超参数
    'num_slots_text': 16,      # 文本情感槽位数
    'num_slots_audio': 16,     # 语音情感槽位数
    'num_slots_vision': 16,    # 视觉情感槽位数
    'num_slots_joint': 8,      # 跨模态融合后的情感事件槽位

    'slot_iters': 5,           # Slot迭代次数 (SlotSPE用3)
    'slot_heads': 8,           # Slot注意力头数

    # MoE Decoder 超参数 (与SlotSPE一致)
    'temperature': 0.01,       # Gumbel-TopK温度
    'topk_ratio': 0.25,        # 保留前25%的slot (可解释性)

    # Transformer 跨模态融合
    'num_heads': 8,
    'cross_attn_iters': 3,     # 跨模态注意力迭代次数
    'dropout': 0.1,

    # Reconstruction Head
    'recon_lambda': 0.1,       # 重建损失权重

    # 分类头
    'num_classes': 3,          # Negative, Neutral, Positive

    # 时序对齐最大长度
    'max_seq_len': 50,
}

# =====================
# 训练配置
# =====================
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'patience': 10,
    'min_delta': 1e-4,

    # 多任务损失权重
    'cls_loss_weight': 1.0,
    'reg_loss_weight': 0.5,
    'recon_loss_weight': 0.1,
}

# =====================
# 实验配置
# =====================
EXPERIMENT_CONFIG = {
    'exp1_slot_baseline': {
        'description': '基线实验: 直接用Slot Attention + 简单融合',
        'use_recon': False,
        'use_cross_modal': False,
    },
    'exp2_with_recon': {
        'description': '重建实验: 加入Reconstruction Head处理模态缺失',
        'use_recon': True,
        'use_cross_modal': False,
    },
    'exp3_interpretable': {
        'description': '可解释性实验: 完整SlotEmo架构，含跨模态交互',
        'use_recon': True,
        'use_cross_modal': True,
    },
}

# 随机种子
SEED = 42

# 输出路径
OUTPUT_ROOT = str(PROJECT_ROOT / 'outputs')
EXPERIMENTS_ROOT = str(PROJECT_ROOT / 'experiments')
