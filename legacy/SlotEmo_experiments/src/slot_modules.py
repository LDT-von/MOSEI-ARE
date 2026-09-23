"""
基于SlotSPE论文的Slot Attention模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, pack, unpack
import math


def split_heads(x, num_heads):
    """将最后一维分成多头: (B, N, H*D) -> (B, H, N, D)"""
    B, N, C = x.shape
    head_dim = C // num_heads
    return x.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def merge_heads(x):
    """合并多头: (B, H, N, D) -> (B, N, H*D)"""
    B, H, N, D = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * D)


class MultiHeadSlotAttention(nn.Module):
    """
    多头Slot Attention (改造自SlotSPE)

    输入: 时序特征 (B, N, D)
    输出: K个slots (B, K, D)

    每个slot通过迭代竞争机制"专门化"，捕捉输入的不同方面

    额外返回 attention weights 用于可解释性分析
    """

    def __init__(
        self,
        num_slots: int,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.heads = heads
        self.head_dim = (dim + heads - 1) // heads
        self.dim_inner = self.head_dim * heads
        self.scale = dim ** -0.5

        # Slot初始化: mu和logsigma (与SlotSPE一致)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        # LayerNorm
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        # Q/K/V投影
        self.to_q = nn.Linear(dim, self.dim_inner)
        self.to_k = nn.Linear(dim, self.dim_inner)
        self.to_v = nn.Linear(dim, self.dim_inner)
        self.combine_heads = nn.Linear(self.dim_inner, dim)

        # GRU更新
        self.gru = nn.GRUCell(dim, dim)

        # MLP
        hidden_dim = max(dim, hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor = None,
        num_slots: int = None,
        return_attention: bool = False
    ):
        """
        Args:
            inputs: (B, N, D) 输入特征
            mask: (B, N) 有效位置掩码, 1=有效, 0=无效
            num_slots: 动态slot数
            return_attention: 是否返回attention weights (用于可解释性)

        Returns:
            slots: (B, K, D)
            attention (可选): (B, K, N) slot对输入的attention
        """
        b, n, d = inputs.shape
        device = inputs.device
        dtype = inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        # 初始化slots
        mu = repeat(self.slots_mu, '1 1 d -> b s d', b=b, s=n_s)
        sigma = repeat(self.slots_logsigma.exp(), '1 1 d -> b s d', b=b, s=n_s)
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

        # 输入归一化
        inputs = self.norm_input(inputs)

        # 处理缺失掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            valid_sum = (inputs * mask_expanded).sum(dim=1, keepdim=True)
            valid_count = mask_expanded.sum(dim=1, keepdim=True) + 1e-8
            valid_mean = valid_sum / valid_count
            inputs = inputs * mask_expanded + valid_mean * (1 - mask_expanded)

        # K, V投影
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        k = split_heads(k, self.heads)
        v = split_heads(v, self.heads)

        # 存储最终的attention weights用于可解释性
        final_attn = None

        # 迭代更新slots
        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            q = split_heads(q, self.heads)

            dots = torch.einsum('bhsd,bhnd->bhsn', q, k) * self.scale
            attn = dots.softmax(dim=-2)
            attn = F.normalize(attn + self.eps, p=1, dim=-2)

            updates = torch.einsum('bhnd,bhsn->bhsd', v, attn)
            updates = merge_heads(updates)
            updates = self.combine_heads(updates)

            updates_flat = updates.reshape(-1, d)
            slots_prev_flat = slots_prev.reshape(-1, d)
            slots_flat = self.gru(updates_flat, slots_prev_flat)
            slots = slots_flat.view(b, n_s, d)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

            # 保留最后一次迭代的attention
            final_attn = attn  # (B, H, S, N)

        if return_attention:
            # 多头平均: (B, H, S, N) -> (B, S, N)
            final_attn = final_attn.mean(dim=1)
            return slots, final_attn

        return slots


class MoESlotDecoder(nn.Module):
    """
    MoE风格的Slot解码器 (改造自SlotSPE)

    每个slot独立预测一个logits, 用Gumbel-TopK选择重要的slots
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_classes: int = 3,
        temperature: float = 0.01,
        topk_ratio: float = 0.25,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.dim = dim
        self.temperature = temperature
        self.k = max(1, int(num_slots * topk_ratio))

        # Slot映射
        self.map = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # 每个slot的分类器
        self.decoder = nn.Linear(dim, num_classes)

        # Slot重要性评估
        self.pred_keep_slot = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        # 偏置初始化使初始时所有slot得分相近, 避免总是选前K个
        nn.init.normal_(self.pred_keep_slot[-1].weight, std=0.01)
        nn.init.zeros_(self.pred_keep_slot[-1].bias)

    def forward(self, slots: torch.Tensor) -> dict:
        slots = self.map(slots)

        # 每个slot的logits
        slot_logits = self.decoder(slots)  # (B, S, C)

        # Slot重要性分数 - 适度scale
        keep_score = self.pred_keep_slot(slots).squeeze(-1) * 10  # (B, S) - 适度放大

        # Gumbel-TopK 选择
        if self.training:
            keep_mask, _ = gumbel_topk_st(keep_score, k=self.k, temperature=max(self.temperature, 0.5))
        else:
            # 推理时直接选topk
            _, top_indices = keep_score.topk(self.k, dim=-1)
            keep_mask = torch.zeros_like(keep_score)
            keep_mask.scatter_(1, top_indices, 1.0)

        # Soft attention作为gate - 只对选中的slots做softmax, 但用温和温度
        # 数值稳定的softmax
        masked_score = keep_score + (1 - keep_mask) * (-1e4)
        max_score = masked_score.max(dim=-1, keepdim=True)[0]
        exp_score = torch.exp((masked_score - max_score) / self.temperature)
        slot_gate = exp_score * keep_mask
        slot_gate = slot_gate / (slot_gate.sum(dim=-1, keepdim=True) + 1e-8)

        # 加权求和
        logits = torch.einsum('bs,bsc->bc', slot_gate, slot_logits)

        return {
            'logits': logits,
            'slot_gate': slot_gate,
            'keep_mask': keep_mask,
            'slot_logits': slot_logits,
            'keep_score': keep_score,
        }


def gumbel_topk_st(logits, k=1, temperature=1.0):
    """Standard straight-through Gumbel-TopK"""
    noised_logits = logits + gumbel_noise(logits)
    topk_indices = noised_logits.topk(k=k, dim=-1).indices
    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)

    # 使用数值更稳定的实现
    # soft_k_hot是每个slot被选中的soft概率
    soft_k_hot = torch.softmax(noised_logits / max(temperature, 0.1), dim=-1) * k

    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return y, topk_indices


def relaxed_topk(logits, k, temperature=1.0):
    """迭代的relaxed top-k"""
    scores = logits
    soft_k_hot = torch.zeros_like(logits)
    for _ in range(k):
        probs = F.softmax(scores / temperature, dim=-1)
        soft_k_hot = soft_k_hot + probs
        scores = scores + torch.log((1.0 - probs).clamp(min=1e-20))
    return soft_k_hot


def gumbel_noise(t):
    """Gumbel(0, 1)噪声 - 数值稳定版本"""
    noise = torch.rand_like(t)
    noise = noise.clamp(min=1e-20, max=1.0 - 1e-7)
    return -torch.log(-torch.log(noise))


class IterativeCrossAttention(nn.Module):
    """
    迭代跨模态Slot Attention (改造自SlotSPE)
    """

    def __init__(self, dim, num_heads=8, iters=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = (dim + num_heads - 1) // num_heads
        self.dim_inner = self.head_dim * num_heads
        self.iters = iters
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, self.dim_inner)
        self.to_k = nn.Linear(dim, self.dim_inner)
        self.to_v = nn.Linear(dim, self.dim_inner)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(self.dim_inner, dim)
        self.proj_drop = nn.Dropout(0.1)

        self.gru1 = nn.GRUCell(dim, dim)
        self.gru2 = nn.GRUCell(dim, dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.norm1a = nn.LayerNorm(dim)
        self.norm1b = nn.LayerNorm(dim)
        self.norm2a = nn.LayerNorm(dim)
        self.norm2b = nn.LayerNorm(dim)

    def forward(self, slots1, slots2):
        for _ in range(self.iters):
            slots1_prev = slots1
            slots2_prev = slots2

            slots1_n = self.norm1a(slots1)
            slots2_n = self.norm2a(slots2)

            q1 = split_heads(self.to_q(slots1_n), self.num_heads)
            q2 = split_heads(self.to_q(slots2_n), self.num_heads)
            k1 = split_heads(self.to_k(slots1_n), self.num_heads)
            k2 = split_heads(self.to_k(slots2_n), self.num_heads)
            v1 = split_heads(self.to_v(slots1_n), self.num_heads)
            v2 = split_heads(self.to_v(slots2_n), self.num_heads)

            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale

            attn1 = self.attn_drop(F.softmax(attn1, dim=-1))
            attn2 = self.attn_drop(F.softmax(attn2, dim=-1))

            update1 = merge_heads(attn1 @ v2)
            update2 = merge_heads(attn2 @ v1)

            update1 = self.proj_drop(self.proj(update1))
            update2 = self.proj_drop(self.proj(update2))

            # GRU更新
            b1 = slots1.shape[0]
            s1 = slots1.shape[1]
            s2 = slots2.shape[1]
            d = slots1.shape[2]

            update1_flat = update1.view(-1, d)
            slots1_prev_flat = slots1_prev.view(-1, d)
            slots1_new = self.gru1(update1_flat, slots1_prev_flat).view(b1, s1, d)

            update2_flat = update2.view(-1, d)
            slots2_prev_flat = slots2_prev.view(-1, d)
            slots2_new = self.gru2(update2_flat, slots2_prev_flat).view(b1, s2, d)

            slots1 = slots1_new + self.mlp1(self.norm1b(slots1_new))
            slots2 = slots2_new + self.mlp2(self.norm2b(slots2_new))

        return slots1, slots2


class ReconstructionHead(nn.Module):
    """
    重建头 (改造自SlotSPE)
    用一个模态的slots重建另一个模态的输入特征
    """

    def __init__(self, query_dim, slot_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, slot_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=slot_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.output_proj = nn.Linear(slot_dim, query_dim)

    def forward(self, query_features, slots):
        query = self.query_proj(query_features)
        recon, _ = self.cross_attn(query, slots, slots)
        recon = recon + self.mlp(self.norm(recon))
        recon = self.output_proj(recon)
        return recon
