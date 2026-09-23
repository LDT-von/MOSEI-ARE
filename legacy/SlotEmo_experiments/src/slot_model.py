"""
SlotEmo: 基于SlotSPE的多模态情感预测模型
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


class SlotEmo(nn.Module):
    """
    SlotEmo: 基于Slot Attention的多模态情感预测模型

    三个模态各自拆解为情感slot,通过跨模态交互和TopK选择实现可解释性
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        d = config['projection_dim']

        # ============= 模态投影 =============
        self.text_proj = nn.Sequential(
            nn.Linear(config['text_dim'], d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(config['audio_dim'], d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(config['vision_dim'], d),
            nn.LayerNorm(d),
            nn.GELU(),
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

        # ============= 跨模态Slot交互 (可选) =============
        self.use_cross_modal = config.get('use_cross_modal', False)
        if self.use_cross_modal:
            self.cross_ta = IterativeCrossAttention(d, num_heads=config['num_heads'], iters=config['cross_attn_iters'])
            self.cross_tv = IterativeCrossAttention(d, num_heads=config['num_heads'], iters=config['cross_attn_iters'])
            self.cross_av = IterativeCrossAttention(d, num_heads=config['num_heads'], iters=config['cross_attn_iters'])

        # ============= 模态贡献度 =============
        self.modality_weight = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.ReLU(),
            nn.Linear(d, 3),
            nn.Softmax(dim=-1),
        )

        # ============= MoE Decoder =============
        # 各模态的slot decoder
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

        # ============= 重建头 (用于模态缺失) =============
        self.use_recon = config.get('use_recon', False)
        if self.use_recon:
            self.recon_text = ReconstructionHead(config['text_dim'], d)
            self.recon_audio = ReconstructionHead(config['audio_dim'], d)
            self.recon_vision = ReconstructionHead(config['vision_dim'], d)

        # ============= 回归头 =============
        self.regression_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 2, d),
        )

    def detect_missing(self, modality, mask=None):
        """检测模态是否整体缺失"""
        if mask is None:
            return False
        valid_ratio = mask.float().mean()
        return valid_ratio < 0.5  # 超过50%缺失视为整体缺失

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask_text: torch.Tensor = None,
        mask_audio: torch.Tensor = None,
        mask_vision: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            text: (B, 50, 768)
            audio: (B, 50, 74)
            vision: (B, 50, 35)
            mask_*: (B, 50) 1=有效, 0=缺失

        Returns:
            dict with predictions and explanations
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

        # ============= 模态投影 =============
        text_h = self.text_proj(text)    # (B, 50, d)
        audio_h = self.audio_proj(audio)  # (B, 50, d)
        vision_h = self.vision_proj(vision) # (B, 50, d)

        # ============= Slot Attention =============
        # 文本slots
        text_slots = self.text_slots(text_h, mask=mask_text)  # (B, K_t, d)
        # 语音slots
        audio_slots = self.audio_slots(audio_h, mask=mask_audio)  # (B, K_a, d)
        # 视觉slots
        vision_slots = self.vision_slots(vision_h, mask=mask_vision)  # (B, K_v, d)

        # 保存原始slots (用于解释)
        orig_text_slots = text_slots
        orig_audio_slots = audio_slots
        orig_vision_slots = vision_slots

        # ============= 跨模态Slot交互 =============
        if self.use_cross_modal:
            # 文本和语音
            text_slots_refined, audio_slots_refined = self.cross_ta(text_slots, audio_slots)
            # 文本和视觉
            text_slots_final, _ = self.cross_tv(text_slots_refined, vision_slots)
            # 语音和视觉
            audio_slots_final, vision_slots_final = self.cross_av(audio_slots_refined, vision_slots)
        else:
            text_slots_final = text_slots
            audio_slots_final = audio_slots
            vision_slots_final = vision_slots

        # ============= MoE Decoder =============
        text_out = self.text_decoder(text_slots_final)
        audio_out = self.audio_decoder(audio_slots_final)
        vision_out = self.vision_decoder(vision_slots_final)

        # 各模态的分类logits
        logits_text = text_out['logits']  # (B, C)
        logits_audio = audio_out['logits']
        logits_vision = vision_out['logits']

        # 模态权重: 基于各模态预测的置信度(最大类别概率)
        text_conf = logits_text.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
        audio_conf = logits_audio.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
        vision_conf = logits_vision.softmax(dim=-1).max(dim=-1, keepdim=True)[0]

        # 防止logits都为nan/inf时置信度也为nan
        text_conf = torch.nan_to_num(text_conf, nan=0.33)
        audio_conf = torch.nan_to_num(audio_conf, nan=0.33)
        vision_conf = torch.nan_to_num(vision_conf, nan=0.33)

        modality_weights = torch.cat([text_conf, audio_conf, vision_conf], dim=-1)
        modality_weights = F.softmax(modality_weights, dim=-1)

        # 加权融合 - 防止logits有nan
        logits_text = torch.nan_to_num(logits_text, nan=0.0)
        logits_audio = torch.nan_to_num(logits_audio, nan=0.0)
        logits_vision = torch.nan_to_num(logits_vision, nan=0.0)

        logits = (
            modality_weights[:, 0:1] * logits_text +
            modality_weights[:, 1:2] * logits_audio +
            modality_weights[:, 2:3] * logits_vision
        )

        # ============= 全局特征(用于回归) =============
        text_global = text_slots_final.mean(dim=1)  # (B, d)
        audio_global = audio_slots_final.mean(dim=1)
        vision_global = vision_slots_final.mean(dim=1)

        fused = torch.cat([text_global, audio_global, vision_global], dim=-1)
        fused = self.fusion(fused)  # (B, d)

        # 回归预测
        regression = self.regression_head(fused).squeeze(-1)  # (B,)

        # ============= 解释信息 =============
        explanations = {
            # 模态重要性
            'modality_weights': modality_weights,
            # 各模态的slot信息
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
            # 各模态的logits
            'logits_per_modality': {
                'text': logits_text,
                'audio': logits_audio,
                'vision': logits_vision,
            },
        }

        return {
            'logits': logits,
            'regression': regression,
            'explanations': explanations,
            'slots': {
                'text': orig_text_slots,
                'audio': orig_audio_slots,
                'vision': orig_vision_slots,
            },
        }


class SlotEmoRobust(SlotEmo):
    """
    SlotEmo的鲁棒版本 - 增加模态缺失重建能力
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask_text: torch.Tensor = None,
        mask_audio: torch.Tensor = None,
        mask_vision: torch.Tensor = None,
    ) -> dict:
        B = text.size(0)
        device = text.device

        # 检测整体缺失
        text_missing = self.detect_missing(text, mask_text)
        audio_missing = self.detect_missing(audio, mask_audio)
        vision_missing = self.detect_missing(vision, mask_vision)

        # 如果某模态缺失,用其他模态重建
        if self.use_recon:
            # 先获取所有slots
            text_h = self.text_proj(text)
            audio_h = self.audio_proj(audio)
            vision_h = self.vision_proj(vision)

            text_slots = self.text_slots(text_h, mask=mask_text)
            audio_slots = self.audio_slots(audio_h, mask=mask_audio)
            vision_slots = self.vision_slots(vision_h, mask=mask_vision)

            # 融合slots用于重建
            # 策略: 把所有可用模态的slots平均
            available_slots = []
            if not text_missing:
                available_slots.append(text_slots)
            if not audio_missing:
                available_slots.append(audio_slots)
            if not vision_missing:
                available_slots.append(vision_slots)

            if len(available_slots) == 0:
                # 所有模态都缺失 - 极端情况,返回零
                return {
                    'logits': torch.zeros(B, self.config['num_classes'], device=device),
                    'regression': torch.zeros(B, device=device),
                    'explanations': {},
                    'slots': {},
                }

            joint_slots = torch.stack(available_slots).mean(dim=0)

            # 重建缺失的模态
            if text_missing:
                recon_text = self.recon_text(text, joint_slots)
                text = recon_text
                text_missing = False  # 重建后视为不缺失
            if audio_missing:
                recon_audio = self.recon_audio(audio, joint_slots)
                audio = recon_audio
                audio_missing = False
            if vision_missing:
                recon_vision = self.recon_vision(vision, joint_slots)
                vision = recon_vision
                vision_missing = False

        # 调用父类
        return super().forward(text, audio, vision, mask_text, mask_audio, mask_vision)
