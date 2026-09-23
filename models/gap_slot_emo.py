"""Mask-aware SlotEmo with optional aligned, local cross-modal gap repair."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TextTokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.segment = nn.Embedding(2, dim)
        self.position = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if tokens.dtype != torch.long or segments.dtype != torch.long:
            raise TypeError("text_bert token IDs and segment IDs must be torch.long")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("text_bert token ID is outside the configured vocabulary")
        batch, steps = tokens.shape
        positions = torch.arange(steps, device=tokens.device).expand(batch, -1)
        x = self.token(tokens) + self.segment(segments) + self.position(positions)
        # Transformer attention cannot have every key masked. The temporary key is
        # removed from the output immediately afterwards.
        key_padding = ~observed.bool()
        empty = key_padding.all(dim=1)
        if empty.any():
            key_padding = key_padding.clone()
            key_padding[empty, 0] = False
        x = self.encoder(x, src_key_padding_mask=key_padding)
        return self.norm(x) * observed.unsqueeze(-1)


class MaskedSlotAttention(nn.Module):
    """Iterative slot attention that excludes absent tokens from every update."""

    def __init__(self, dim: int, slots: int, heads: int, iterations: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dimension must be divisible by slot heads")
        self.dim, self.slots, self.heads, self.iterations = dim, slots, heads, iterations
        self.initial = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, dim = x.shape
        head_dim = dim // self.heads
        valid = observed.bool()
        slots = self.initial.unsqueeze(0).expand(batch, -1, -1)
        x = self.input_norm(x)
        k = self.to_k(x).view(batch, steps, self.heads, head_dim).transpose(1, 2)
        v = self.to_v(x).view(batch, steps, self.heads, head_dim).transpose(1, 2)
        final_attention = None
        for _ in range(self.iterations):
            previous = slots
            q = self.to_q(self.slot_norm(slots)).view(batch, self.slots, self.heads, head_dim).transpose(1, 2)
            scores = torch.einsum("bhkd,bhtd->bhkt", q, k) * (head_dim ** -0.5)
            competition = scores.softmax(dim=2) * valid[:, None, None, :]
            attention = competition / competition.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bhkt,bhtd->bhkd", attention, v)
            updates = updates.transpose(1, 2).contiguous().view(batch, self.slots, dim)
            slots = self.gru(updates.reshape(-1, dim), previous.reshape(-1, dim)).view(batch, self.slots, dim)
            slots = slots + self.mlp(slots)
            final_attention = attention.mean(dim=1)
        # A branch with no evidence must contribute exactly zero to downstream heads.
        active = valid.any(dim=1)
        weights = self.score(slots).squeeze(-1).softmax(dim=1)
        summary = torch.einsum("bk,bkd->bd", weights, slots) * active.unsqueeze(-1)
        return summary, slots, final_attention


class AlignedGapRepair(nn.Module):
    """Use only other observed modalities at the same aligned time position."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.source = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
        self.source_score = nn.ModuleList(nn.Linear(dim, 1) for _ in range(3))
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(dim + 4, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1), nn.Sigmoid())
            for _ in range(3)
        )
        self.imputed_marker = nn.Parameter(torch.zeros(3, dim))

    @staticmethod
    def _geometry(observed: torch.Tensor) -> torch.Tensor:
        batch, steps = observed.shape
        pos = torch.arange(steps, device=observed.device).expand(batch, -1)
        left = torch.where(observed, pos, -1).cummax(dim=1).values
        right = steps - 1 - torch.where(observed.flip(1), pos, -1).cummax(dim=1).values.flip(1)
        nearest = torch.minimum(pos - left, right - pos).float() / steps
        span = (right - left - 1).clamp(min=0).float() / steps
        time = pos.float() / max(steps - 1, 1)
        return torch.stack((time, span, nearest), dim=-1)

    def forward(self, features: torch.Tensor, observed: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: B,3,T,D; observed: B,3,T; support: B,T.
        projected = torch.stack([layer(features[:, i]) for i, layer in enumerate(self.source)], dim=1)
        source_scores = torch.stack([layer(features[:, i]).squeeze(-1) for i, layer in enumerate(self.source_score)], dim=1)
        repaired, effective = [], []
        for target in range(3):
            sources = observed.clone()
            sources[:, target] = False
            any_source = sources.any(dim=1)
            weights = source_scores.masked_fill(~sources, -1e4).softmax(dim=1)
            weights = weights * sources
            candidate = (weights.unsqueeze(-1) * projected).sum(dim=1)
            source_fraction = sources.float().sum(dim=1, keepdim=False).unsqueeze(-1) / 2
            geometry = self._geometry(observed[:, target])
            gate = self.gates[target](torch.cat((candidate, geometry, source_fraction), dim=-1))
            missing = ~observed[:, target]
            impute = missing & support & any_source
            value = torch.where(
                observed[:, target].unsqueeze(-1), features[:, target],
                (gate * candidate + self.imputed_marker[target]) * impute.unsqueeze(-1),
            )
            repaired.append(value)
            effective.append(observed[:, target] | impute)
        return torch.stack(repaired, dim=1), torch.stack(effective, dim=1)


class GapSlotEmo(nn.Module):
    """A/B/C share all parameters except C's local gap-repair module."""

    def __init__(
        self, *, vocab_size: int = 30522, dim: int = 128, slots: int = 16,
        slot_heads: int = 4, slot_iterations: int = 3, max_len: int = 50,
        dropout: float = 0.1, use_gap_repair: bool = False,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.use_gap_repair = use_gap_repair
        self.text_encoder = TextTokenEncoder(vocab_size, dim, max_len, dropout)
        self.audio_project = nn.Sequential(nn.Linear(74, dim), nn.LayerNorm(dim), nn.GELU())
        self.vision_project = nn.Sequential(nn.Linear(35, dim), nn.LayerNorm(dim), nn.GELU())
        self.time_position = nn.Embedding(max_len, dim)
        self.modalities = nn.ModuleList(MaskedSlotAttention(dim, slots, slot_heads, slot_iterations) for _ in range(3))
        self.repair = AlignedGapRepair(dim) if use_gap_repair else None
        self.modality_score = nn.Sequential(nn.Linear(dim + 1, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.fusion = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.classifier = nn.Linear(dim, 3)
        self.regressor = nn.Linear(dim, 1)

    def forward(
        self, *, tokens: torch.Tensor, segments: torch.Tensor, audio: torch.Tensor,
        vision: torch.Tensor, support: torch.Tensor, observed: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        if tokens.shape[1] > self.max_len or observed.shape != (tokens.shape[0], 3, tokens.shape[1]):
            raise ValueError("inconsistent 50-position aligned input")
        support, observed = support.bool(), observed.bool()
        observed = observed & support.unsqueeze(1)
        text = self.text_encoder(tokens, segments, observed[:, 0])
        a = self.audio_project(audio) * observed[:, 1].unsqueeze(-1)
        v = self.vision_project(vision) * observed[:, 2].unsqueeze(-1)
        features = torch.stack((text, a, v), dim=1)
        if self.repair is not None:
            features, effective = self.repair(features, observed, support)
        else:
            effective = observed
        positions = self.time_position(torch.arange(tokens.shape[1], device=tokens.device))
        summaries, slots, attentions = [], [], []
        for modality in range(3):
            summary, slot_values, attention = self.modalities[modality](
                features[:, modality] + positions.unsqueeze(0), effective[:, modality]
            )
            summaries.append(summary)
            slots.append(slot_values)
            attentions.append(attention)
        summaries = torch.stack(summaries, dim=1)
        available = effective.any(dim=-1)
        coverage = observed.sum(dim=-1).float() / support.sum(dim=-1).clamp_min(1).unsqueeze(1)
        gate_in = torch.cat((summaries, coverage.unsqueeze(-1)), dim=-1)
        scores = self.modality_score(gate_in).squeeze(-1).masked_fill(~available, -1e4)
        modality_weights = scores.softmax(dim=-1) * available
        modality_weights = modality_weights / modality_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weighted = (modality_weights.unsqueeze(-1) * summaries).sum(dim=1)
        fused = self.fusion(torch.cat((weighted, summaries.flatten(1)), dim=-1))
        result = {
            "logits": self.classifier(fused),
            "regression": 3 * torch.tanh(self.regressor(fused).squeeze(-1)),
            "modality_weights": modality_weights,
            "effective_observed": effective,
        }
        if return_attention:
            result["slot_attention"] = torch.stack(attentions, dim=1)
            result["slots"] = torch.stack(slots, dim=1)
        return result
