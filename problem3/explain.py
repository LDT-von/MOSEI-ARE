"""Counterfactual occlusion and sliding-window evidence for the shared GapSlotEmo model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from models.gap_slot_emo import GapSlotEmo
from problem2.data import AlignedDataset


MODALITY_NAMES = ("text", "audio", "vision")


@dataclass
class ModalityContribution:
    """How each modality shifts the clean baseline prediction."""

    clean_logit: float
    clean_intensity: float
    clean_pred: int
    ablated_logits: dict[str, list[float]]
    ablated_intensity: dict[str, float]
    ablated_pred: dict[str, int]
    polarity_delta: dict[str, float]
    intensity_delta: dict[str, float]
    normalized_magnitude: dict[str, float]
    dominant_modality: str

    def to_dict(self) -> dict:
        return {
            "clean_logit": self.clean_logit,
            "clean_intensity": self.clean_intensity,
            "clean_pred": self.clean_pred,
            "polarity_delta": self.polarity_delta,
            "intensity_delta": self.intensity_delta,
            "normalized_magnitude": self.normalized_magnitude,
            "dominant_modality": self.dominant_modality,
        }


@dataclass
class WindowEvidence:
    """Top-k contiguous windows whose ablation moves the target logit the most."""

    modality: str
    target_class: int
    window_length: int
    windows: list[dict]

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "target_class": self.target_class,
            "window_length": self.window_length,
            "windows": self.windows,
        }


def _sample_inputs(sample: dict, device: torch.device) -> dict:
    return {
        "tokens": sample["tokens"].unsqueeze(0).to(device),
        "segments": sample["segments"].unsqueeze(0).to(device),
        "audio": sample["audio"].unsqueeze(0).to(device),
        "vision": sample["vision"].unsqueeze(0).to(device),
        "support": sample["support"].unsqueeze(0).to(device),
        "observed": sample["observed"].unsqueeze(0).to(device),
    }


def _forward(model: GapSlotEmo, inputs: dict) -> tuple[np.ndarray, float]:
    with torch.inference_mode():
        result = model(**inputs)
    logits = result["logits"][0].detach().cpu().numpy()
    intensity = float(result["regression"][0].detach().cpu().item())
    return logits, intensity


def modality_contributions(model: GapSlotEmo, sample: dict, device: torch.device) -> ModalityContribution:
    """Predict clean; ablate each modality by zeroing observed features at originally valid positions."""
    inputs = _sample_inputs(sample, device)
    clean_logits, clean_intensity = _forward(model, inputs)
    clean_pred = int(np.argmax(clean_logits))
    ablated_logits: dict[str, list[float]] = {}
    ablated_intensity: dict[str, float] = {}
    ablated_pred: dict[str, int] = {}
    polarity_delta: dict[str, float] = {}
    intensity_delta: dict[str, float] = {}
    normalized: dict[str, float] = {}
    for index, name in enumerate(MODALITY_NAMES):
        modified = {key: value.clone() for key, value in inputs.items()}
        mask = inputs["observed"][0, index] & inputs["support"][0]
        if name == "text":
            modified["tokens"] = modified["tokens"].clone()
            modified["tokens"][0, mask] = 0
        elif name == "audio":
            modified["audio"] = modified["audio"].clone()
            modified["audio"][0, mask] = 0
        else:
            modified["vision"] = modified["vision"].clone()
            modified["vision"][0, mask] = 0
        masked_obs = inputs["observed"].clone()
        masked_obs[0, index] = inputs["observed"][0, index] & ~mask
        modified["observed"] = masked_obs
        logits, intensity = _forward(model, modified)
        ablated_logits[name] = logits.tolist()
        ablated_intensity[name] = intensity
        ablated_pred[name] = int(np.argmax(logits))
        polarity_delta[name] = float(clean_logits[clean_pred] - logits[clean_pred])
        intensity_delta[name] = clean_intensity - intensity
    total = sum(abs(value) for value in polarity_delta.values()) or 1.0
    for name, value in polarity_delta.items():
        normalized[name] = float(abs(value) / total)
    dominant = max(normalized, key=normalized.get) if normalized else "text"
    return ModalityContribution(
        clean_logit=float(clean_logits[clean_pred]),
        clean_intensity=clean_intensity,
        clean_pred=clean_pred,
        ablated_logits=ablated_logits,
        ablated_intensity=ablated_intensity,
        ablated_pred=ablated_pred,
        polarity_delta=polarity_delta,
        intensity_delta=intensity_delta,
        normalized_magnitude=normalized,
        dominant_modality=dominant,
    )


def modality_weights_from_model(model: GapSlotEmo, sample: dict, device: torch.device) -> dict[str, float]:
    """Forward pass and read the model's own modality weighting as auxiliary signal only."""
    inputs = _sample_inputs(sample, device)
    with torch.inference_mode():
        result = model(**inputs)
    weights = result["modality_weights"][0].detach().cpu().numpy()
    return {name: float(weights[index]) for index, name in enumerate(MODALITY_NAMES)}


def _ablate_window(
    model: GapSlotEmo, inputs: dict, modality_index: int, positions: Iterable[int]
) -> tuple[np.ndarray, float]:
    modified = {key: value.clone() for key, value in inputs.items()}
    positions_tensor = torch.tensor(list(positions), dtype=torch.long, device=modified["tokens"].device)
    name = MODALITY_NAMES[modality_index]
    if name == "text":
        modified["tokens"][0, positions_tensor] = 0
    elif name == "audio":
        modified["audio"][0, positions_tensor] = 0
    else:
        modified["vision"][0, positions_tensor] = 0
    masked_obs = inputs["observed"].clone()
    row = inputs["observed"][0, modality_index].clone()
    for position in positions:
        row[position] = False
    masked_obs[0, modality_index] = row
    modified["observed"] = masked_obs
    return _forward(model, modified)


def window_evidence(
    model: GapSlotEmo, sample: dict, device: torch.device, *,
    modality: str, window_length: int, top_k: int = 3,
    target_class: int | None = None,
) -> WindowEvidence:
    """Slide a fixed-length window along the 50-position aligned sequence."""
    if modality not in MODALITY_NAMES:
        raise ValueError("modality must be text/audio/vision")
    if not 1 <= window_length <= 50:
        raise ValueError("window_length must be in [1, 50]")
    modality_index = MODALITY_NAMES.index(modality)
    inputs = _sample_inputs(sample, device)
    observed = inputs["observed"][0, modality_index].cpu().numpy()
    eligible = np.flatnonzero(observed)
    if modality == "text":
        content = np.flatnonzero(inputs["support"][0].cpu().numpy())
        if len(content) >= 2:
            eligible = eligible[(eligible != content[0]) & (eligible != content[-1])]
    clean_logits, _ = _forward(model, inputs)
    if target_class is None:
        target_class = int(np.argmax(clean_logits))
    clean_score = float(clean_logits[target_class])
    scored: list[tuple[float, int]] = []
    for start in range(0, 50 - window_length + 1):
        end = start + window_length
        if not np.any(observed[start:end]):
            continue
        positions = list(range(start, end))
        logits, _ = _ablate_window(model, inputs, modality_index, positions)
        score = float(clean_score - logits[target_class])
        scored.append((score, start))
    scored.sort(key=lambda item: item[0], reverse=True)
    windows = []
    for score, start in scored[:top_k]:
        windows.append({
            "start_position": int(start),
            "end_position": int(start + window_length),
            "start_seconds": round(float((start + 0.5) * 0.2), 3),
            "end_seconds": round(float((start + window_length + 0.5) * 0.2), 3),
            "logit_drop": round(score, 4),
        })
    return WindowEvidence(
        modality=modality,
        target_class=target_class,
        window_length=window_length,
        windows=windows,
    )


def collect_batch_evidence(
    model: GapSlotEmo, dataset: AlignedDataset, indices: list[int], device: torch.device, *,
    window_length: int = 6, top_k: int = 3,
) -> list[dict]:
    """Run modality contribution + window scan for each index; return combined per-sample record."""
    records: list[dict] = []
    for index in indices:
        sample = dataset[index]
        contribution = modality_contributions(model, sample, device)
        auxiliary = modality_weights_from_model(model, sample, device)
        per_modality_windows: dict[str, dict] = {}
        for modality in MODALITY_NAMES:
            evidence = window_evidence(
                model, sample, device,
                modality=modality, window_length=window_length, top_k=top_k,
                target_class=contribution.clean_pred,
            )
            per_modality_windows[modality] = evidence.to_dict()
        records.append({
            "sample_index": int(index),
            "sample_id": sample["id"],
            "modality_contribution": contribution.to_dict(),
            "modality_weights_auxiliary": auxiliary,
            "window_evidence": per_modality_windows,
        })
    return records


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def write_attachment4_csv(path: Path, rows: list[dict]) -> None:
    """Attachment 4: 20 rows of predictions, dominant modality, evidence summary, source pointers."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "sample_id", "polarity", "intensity",
            "dominant_modality",
            "top_text_window", "top_audio_window", "top_vision_window",
            "text_polarity_delta", "audio_polarity_delta", "vision_polarity_delta",
            "intensity_delta_text", "intensity_delta_audio", "intensity_delta_vision",
            "raw_text_snippet", "video_filename",
        ))
        for row in rows:
            windows = row["window_evidence"]
            text_window = _format_window(windows["text"]["windows"][0]) if windows["text"]["windows"] else ""
            audio_window = _format_window(windows["audio"]["windows"][0]) if windows["audio"]["windows"] else ""
            vision_window = _format_window(windows["vision"]["windows"][0]) if windows["vision"]["windows"] else ""
            contribution = row["modality_contribution"]
            writer.writerow((
                row["sample_id"],
                row["prediction"]["polarity"],
                f"{row['prediction']['intensity']:.6f}",
                contribution["dominant_modality"],
                text_window,
                audio_window,
                vision_window,
                f"{contribution['polarity_delta']['text']:.4f}",
                f"{contribution['polarity_delta']['audio']:.4f}",
                f"{contribution['polarity_delta']['vision']:.4f}",
                f"{contribution['intensity_delta']['text']:.4f}",
                f"{contribution['intensity_delta']['audio']:.4f}",
                f"{contribution['intensity_delta']['vision']:.4f}",
                (row.get("raw_text") or "")[:160],
                row.get("video_filename", ""),
            ))


def _format_window(window: dict) -> str:
    return f"{window['start_seconds']:.2f}-{window['end_seconds']:.2f}s (drop {window['logit_drop']:+.3f})"
