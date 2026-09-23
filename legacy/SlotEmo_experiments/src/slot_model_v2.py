"""
SlotEmo v2: 优化版
改进:
1. 加入sparse loss让slot专门化
2. 加入dropout防止过拟合
3. 返回slot对输入的attention, 用于关键证据定位
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .slot_modules import (
    MultiHeadSlotAttention,
    MoESlotDecoder,
    IterativeCrossAttention,
    ReconstructionHead
)


class SlotEmoV2(nn.Module):
    """
    SlotEmo v2 - 加入sparse正则和更好的可解释性
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_cross_modal = config.get('use_cross_modal', False)
        self.use_recon = config.get('use_recon', False)

        d = config['projection_dim']
        dropout = config.get('dropout', 0.3)

        # ============= 模态投影 =============
        self.text_proj = nn.Sequential(
            nn.Linear(config['text_dim'], d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(config['audio_dim'], d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(config['vision_dim'], d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ============= Slot Attention =============
        self.text_slots = MultiHeadSlotAttention(
            num_slots=config['num_slots_text'],
            dim=d,
            heads=config['slot_heads'],
            iters=config['slot_iters'],
        )
        self.audio_slots = MultiHeadSlotAttention(
            num_slots=config['num_slots_audio'],
            dim=d,
            heads=config['slot_heads'],
            iters=config['slot_iters'],
        )
        self.vision_slots = MultiHeadSlotAttention(
            num_slots=config['num_slots_vision'],
            dim=d,
            heads=config['slot_heads'],
            iters=config['slot_iters'],
        )

        # ============= 跨模态Slot交互 =============
        if self.use_cross_modal:
            self.cross_ta = IterativeCrossAttention(d, num_heads=config['num_heads'], iters=config['cross_attn_iters'])
            self.cross_tv = IterativeCrossAttention(d, num_heads=config['num_heads'], iters=config['cross_attn_iters'])

        # ============= MoE Decoder =============
        self.text_decoder = MoESlotDecoder(
            dim=d,
            num_slots=config['num_slots_text'],
            num_classes=config['num_classes'],
            temperature=config['temperature'],
            topk_ratio=config['topk_ratio'],
        )
        self.audio_decoder = MoESlotDecoder(
            dim=d,
            num_slots=config['num_slots_audio'],
            num_classes=config['num_classes'],
            temperature=config['temperature'],
            topk_ratio=config['topk_ratio'],
        )
        self.vision_decoder = MoESlotDecoder(
            dim=d,
            num_slots=config['num_slots_vision'],
            num_classes=config['num_classes'],
            temperature=config['temperature'],
            topk_ratio=config['topk_ratio'],
        )

        # ============= 重建头 =============
        if self.use_recon:
            self.recon_text = ReconstructionHead(config['text_dim'], d)
            self.recon_audio = ReconstructionHead(config['audio_dim'], d)
            self.recon_vision = ReconstructionHead(config['vision_dim'], d)

        # ============= 回归头 =============
        self.regression_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

        # 全局融合
        self.fusion = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
        )

    def detect_missing(self, modality, mask=None):
        """检测模态是否整体缺失"""
        if mask is None:
            return False
        valid_ratio = mask.float().mean()
        return valid_ratio < 0.5

    def compute_sparse_loss(self, gates_list):
        """
        计算sparse损失: 鼓励slot之间的分工

        目标:
        1. 让样本间使用slots的多样性(不同样本用不同slots)
        2. 最小化选中slots之间的gate方差(让某些slot占主导)

        gates_list: 包含多个(B, S)gate张量的列表
        """
        total_loss = 0
        for gates in gates_list:
            # 1. 跨样本多样性: 不同样本使用不同slots
            # 计算每个slot在所有样本上的平均使用率
            slot_usage = gates.mean(dim=0)  # (S,)
            # 鼓励平均分布: 最大化使用率的熵
            entropy_usage = -(slot_usage * torch.log(slot_usage + 1e-8)).sum()
            total_loss = total_loss - 0.5 * entropy_usage  # 最大化熵 = 最小化负熵

            # 2. 单样本内, 鼓励选中slot之间的多样性 (非均匀)
            # 衡量选中slot之间的差异: 让一个slot主导, 其他次要
            # 目标: 让gate分布的熵更小(更尖锐)
            entropy_per_sample = -(gates * torch.log(gates + 1e-8)).sum(dim=-1).mean()
            total_loss = total_loss + 0.3 * entropy_per_sample  # 最小化熵 = 鼓励尖锐

        return total_loss / len(gates_list)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask_text: torch.Tensor = None,
        mask_audio: torch.Tensor = None,
        mask_vision: torch.Tensor = None,
        return_attention: bool = True,
    ) -> dict:
        """
        Args:
            text: (B, 50, 768)
            audio: (B, 50, 74)
            vision: (B, 50, 35)
            mask_*: (B, 50) 1=有效, 0=缺失
            return_attention: 是否返回attention用于可解释性

        Returns:
            dict with predictions, explanations, sparse_loss
        """
        B = text.size(0)
        d = self.config['projection_dim']
        device = text.device

        # 默认mask
        if mask_text is None:
            mask_text = torch.ones(B, text.size(1), device=device)
        if mask_audio is None:
            mask_audio = torch.ones(B, audio.size(1), device=device)
        if mask_vision is None:
            mask_vision = torch.ones(B, vision.size(1), device=device)

        # 模态投影
        text_h = self.text_proj(text)
        audio_h = self.audio_proj(audio)
        vision_h = self.vision_proj(vision)

        # Slot Attention (with attention for interpretability)
        if return_attention:
            text_slots, text_attn = self.text_slots(text_h, mask=mask_text, return_attention=True)
            audio_slots, audio_attn = self.audio_slots(audio_h, mask=mask_audio, return_attention=True)
            vision_slots, vision_attn = self.vision_slots(vision_h, mask=mask_vision, return_attention=True)
        else:
            text_slots = self.text_slots(text_h, mask=mask_text)
            audio_slots = self.audio_slots(audio_h, mask=mask_audio)
            vision_slots = self.vision_slots(vision_h, mask=mask_vision)
            text_attn = audio_attn = vision_attn = None

        orig_text_slots = text_slots
        orig_audio_slots = audio_slots
        orig_vision_slots = vision_slots

        # 跨模态交互
        if self.use_cross_modal:
            text_slots, audio_slots = self.cross_ta(text_slots, audio_slots)
            text_slots, _ = self.cross_tv(text_slots, vision_slots)

        # MoE Decoder
        text_out = self.text_decoder(text_slots)
        audio_out = self.audio_decoder(audio_slots)
        vision_out = self.vision_decoder(vision_slots)

        logits_text = text_out['logits']
        logits_audio = audio_out['logits']
        logits_vision = vision_out['logits']

        # 模态权重
        text_conf = logits_text.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
        audio_conf = logits_audio.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
        vision_conf = logits_vision.softmax(dim=-1).max(dim=-1, keepdim=True)[0]

        text_conf = torch.nan_to_num(text_conf, nan=0.33)
        audio_conf = torch.nan_to_num(audio_conf, nan=0.33)
        vision_conf = torch.nan_to_num(vision_conf, nan=0.33)

        modality_weights = torch.cat([text_conf, audio_conf, vision_conf], dim=-1)
        modality_weights = F.softmax(modality_weights, dim=-1)

        logits_text = torch.nan_to_num(logits_text, nan=0.0)
        logits_audio = torch.nan_to_num(logits_audio, nan=0.0)
        logits_vision = torch.nan_to_num(logits_vision, nan=0.0)

        logits = (
            modality_weights[:, 0:1] * logits_text +
            modality_weights[:, 1:2] * logits_audio +
            modality_weights[:, 2:3] * logits_vision
        )

        # 回归
        text_global = text_slots.mean(dim=1)
        audio_global = audio_slots.mean(dim=1)
        vision_global = vision_slots.mean(dim=1)
        fused = torch.cat([text_global, audio_global, vision_global], dim=-1)
        fused = self.fusion(fused)
        regression = self.regression_head(fused).squeeze(-1)

        # Sparse loss: 让slot之间有分工
        gates_list = [text_out['slot_gate'], audio_out['slot_gate'], vision_out['slot_gate']]
        sparse_loss = self.compute_sparse_loss(gates_list)

        # 解释信息
        explanations = {
            'modality_weights': modality_weights,
            'text_slot_info': {
                'slot_gate': text_out['slot_gate'],
                'keep_mask': text_out['keep_mask'],
                'slot_logits': text_out['slot_logits'],
                'keep_score': text_out['keep_score'],
            },
            'audio_slot_info': {
                'slot_gate': audio_out['slot_gate'],
                'keep_mask': audio_out['keep_mask'],
                'slot_logits': audio_out['slot_logits'],
                'keep_score': audio_out['keep_score'],
            },
            'vision_slot_info': {
                'slot_gate': vision_out['slot_gate'],
                'keep_mask': vision_out['keep_mask'],
                'slot_logits': vision_out['slot_logits'],
                'keep_score': vision_out['keep_score'],
            },
            'logits_per_modality': {
                'text': logits_text,
                'audio': logits_audio,
                'vision': logits_vision,
            },
        }

        # 关键证据定位: 每个时间步的重要性
        if return_attention:
            # 把attention和gate结合, 得到每个时间步对最终预测的贡献
            # text_attn: (B, S, N), gate: (B, S)
            # 时间步重要性 = sum_s(gate_s * attn_s)
            for mod_name, attn, slot_info in [
                ('text', text_attn, text_out),
                ('audio', audio_attn, audio_out),
                ('vision', vision_attn, vision_out)
            ]:
                if attn is not None:
                    gate = slot_info['slot_gate']  # (B, S)
                    # 时间步重要性
                    time_importance = torch.einsum('bs,bsn->bn', gate, attn)  # (B, N)
                    explanations[f'{mod_name}_time_importance'] = time_importance

        return {
            'logits': logits,
            'regression': regression,
            'explanations': explanations,
            'sparse_loss': sparse_loss,
            'slots': {
                'text': orig_text_slots,
                'audio': orig_audio_slots,
                'vision': orig_vision_slots,
            },
        }


class SlotEmoRobustV2(SlotEmoV2):
    """
    SlotEmo的鲁棒版本 v2 - 改进的模态缺失重建
    """

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask_text: torch.Tensor = None,
        mask_audio: torch.Tensor = None,
        mask_vision: torch.Tensor = None,
        return_attention: bool = True,
    ) -> dict:
        B = text.size(0)
        device = text.device

        # 检测整体缺失
        text_missing = self.detect_missing(text, mask_text)
        audio_missing = self.detect_missing(audio, mask_audio)
        vision_missing = self.detect_missing(vision, mask_vision)

        # 如果某模态缺失,用其他模态重建
        if self.use_recon and (text_missing or audio_missing or vision_missing):
            text_h = self.text_proj(text)
            audio_h = self.audio_proj(audio)
            vision_h = self.vision_proj(vision)

            text_slots = self.text_slots(text_h, mask=mask_text)
            audio_slots = self.audio_slots(audio_h, mask=mask_audio)
            vision_slots = self.vision_slots(vision_h, mask=mask_vision)

            # 收集可用slots
            available_slots = []
            if not text_missing:
                available_slots.append(text_slots)
            if not audio_missing:
                available_slots.append(audio_slots)
            if not vision_missing:
                available_slots.append(vision_slots)

            if len(available_slots) == 0:
                return {
                    'logits': torch.zeros(B, self.config['num_classes'], device=device),
                    'regression': torch.zeros(B, device=device),
                    'explanations': {},
                    'sparse_loss': torch.tensor(0.0, device=device),
                    'slots': {},
                }

            # 加权融合slots (权重=有效比例)
            weights = []
            if not text_missing:
                weights.append(mask_text.float().mean(dim=1).mean().item() if mask_text is not None else 1.0)
            if not audio_missing:
                weights.append(mask_audio.float().mean(dim=1).mean().item() if mask_audio is not None else 1.0)
            if not vision_missing:
                weights.append(mask_vision.float().mean(dim=1).mean().item() if mask_vision is not None else 1.0)
            weights = torch.tensor(weights, device=device)
            weights = F.softmax(weights, dim=0)

            joint_slots = sum(w * s for w, s in zip(weights, available_slots))

            # 重建缺失的模态
            if text_missing:
                text = self.recon_text(text, joint_slots)
            if audio_missing:
                audio = self.recon_audio(audio, joint_slots)
            if vision_missing:
                vision = self.recon_vision(vision, joint_slots)

        return super().forward(text, audio, vision, mask_text, mask_audio, mask_vision, return_attention)
