"""Small candidate model: train-only text distillation and optional conditional fusion.

The shipped GapSlotEmo source is deliberately left intact so old checkpoint
source hashes and state dictionaries remain usable.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.gap_slot_emo import GapSlotEmo


class GapSlotEmoCandidate(GapSlotEmo):
    def __init__(self, *, use_cmf: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_cmf = use_cmf
        dim = self.classifier.in_features
        if use_cmf:
            self.modulation = nn.Sequential(
                nn.LayerNorm(3 * dim + 6),
                nn.Linear(3 * dim + 6, dim),
                nn.GELU(),
                nn.Linear(dim, 6 * dim),
            )
            # Identity at initialization; any change must be learned from data.
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)
        else:
            self.modulation = None
        generator = torch.Generator().manual_seed(20260925)
        projection = torch.randn(768, dim, generator=generator) / (768 ** 0.5)
        self.register_buffer("teacher_projection", projection)

    def distillation_loss(
        self, student: torch.Tensor, teacher: torch.Tensor, observed_text: torch.Tensor,
    ) -> torch.Tensor:
        """Match only visible text positions; the fixed teacher has no gradients."""
        if teacher.shape[:2] != student.shape[:2] or teacher.shape[-1] != 768:
            raise ValueError("teacher text must align with student positions and have 768 channels")
        valid = observed_text.bool() & torch.isfinite(teacher).all(dim=-1)
        if not valid.any():
            return student.sum() * 0
        target = F.normalize(teacher[valid].float() @ self.teacher_projection.float(), dim=-1)
        prediction = F.normalize(student[valid].float(), dim=-1)
        return (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()

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
            repair_support = support & (tokens != 101) & (tokens != 102)
            features, effective = self.repair(features, observed, repair_support)
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
        if self.modulation is not None:
            condition = torch.cat((summaries.flatten(1), coverage, available.float()), dim=-1)
            gamma, beta = self.modulation(condition).chunk(2, dim=-1)
            gamma = 1 + 0.5 * torch.tanh(gamma.view(-1, 3, summaries.shape[-1]))
            beta = 0.1 * torch.tanh(beta.view(-1, 3, summaries.shape[-1]))
            summaries = (gamma * summaries + beta) * available.unsqueeze(-1)
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
            "text_features": text,
        }
        if return_attention:
            result["slot_attention"] = torch.stack(attentions, dim=1)
            result["slots"] = torch.stack(slots, dim=1)
        return result
