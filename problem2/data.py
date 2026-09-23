"""Consistent aligned_50 inputs for Attachment 2 and Attachment 3."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("text", "audio", "vision")


@dataclass
class SplitData:
    ids: list[str]
    tokens: np.ndarray
    segments: np.ndarray
    support: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    observed: np.ndarray
    labels: np.ndarray | None = None
    intensity: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)


def _integer_array(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError(f"{name} must contain finite integer values")
    return values.astype(np.int64, copy=False)


def convert_split(raw: dict, *, ids: list[str] | None = None, require_labels: bool = True) -> SplitData:
    required = {"text_bert", "audio", "vision"}
    if not required.issubset(raw):
        raise KeyError(f"missing fields: {sorted(required - set(raw))}")
    bert = np.asarray(raw["text_bert"])
    audio = np.asarray(raw["audio"], dtype=np.float32)
    vision = np.asarray(raw["vision"], dtype=np.float32)
    if bert.ndim != 3 or bert.shape[1:] != (3, 50):
        raise ValueError(f"text_bert must have shape (N,3,50), got {bert.shape}")
    n = bert.shape[0]
    if audio.shape != (n, 50, 74) or vision.shape != (n, 50, 35):
        raise ValueError(f"inconsistent aligned_50 shapes: {audio.shape}, {vision.shape}")
    if not np.isfinite(audio).all() or not np.isfinite(vision).all():
        raise ValueError("audio/vision contain NaN or infinity")
    tokens = _integer_array(bert[:, 0], "token IDs")
    attention = _integer_array(bert[:, 1], "attention mask")
    segments = _integer_array(bert[:, 2], "segment IDs")
    if tokens.min() < 0 or tokens.max() >= 30522:
        raise ValueError("token ID outside bert-base-uncased vocabulary [0,30522)")
    if not np.isin(attention, (0, 1)).all() or not np.isin(segments, (0, 1)).all():
        raise ValueError("attention mask and segment IDs must be binary")
    support = attention.astype(bool)
    text_observed = support & (tokens != 0)
    audio_observed = support & np.any(audio != 0, axis=-1)
    vision_observed = support & np.any(vision != 0, axis=-1)
    observed = np.stack((text_observed, audio_observed, vision_observed), axis=1)
    if ids is None:
        if "id" not in raw:
            raise KeyError("training split has no sample IDs")
        ids = [str(x) for x in raw["id"]]
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("sample IDs must be unique and match tensor count")
    labels = intensity = None
    if require_labels:
        if "classification_labels" not in raw or "regression_labels" not in raw:
            raise KeyError("labeled split is missing sentiment labels")
        labels = _integer_array(np.asarray(raw["classification_labels"]).reshape(-1), "classification labels")
        intensity = np.asarray(raw["regression_labels"], dtype=np.float32).reshape(-1)
        if len(labels) != n or len(intensity) != n or not np.isin(labels, (0, 1, 2)).all():
            raise ValueError("invalid label shape or class")
        if not np.isfinite(intensity).all() or np.any(np.abs(intensity) > 3.001):
            raise ValueError("sentiment intensity must be finite and within [-3,3]")
    return SplitData(ids, tokens, segments, support, audio, vision, observed, labels, intensity)


def load_attachment2(path: Path) -> dict[str, SplitData]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not {"train", "valid", "test"}.issubset(raw):
        raise KeyError("Attachment 2 must contain train, valid and test")
    splits = {name: convert_split(raw[name]) for name in ("train", "valid", "test")}
    if len(set().union(*(set(s.ids) for s in splits.values()))) != sum(map(len, splits.values())):
        raise ValueError("sample IDs overlap across train/valid/test")
    return splits


def load_attachment3(directory: Path) -> SplitData:
    paths = sorted(directory.glob("*.pkl"))
    if len(paths) != 30:
        raise ValueError(f"expected 30 aligned Attachment 3 files, found {len(paths)}")
    parts = []
    for path in paths:
        with path.open("rb") as handle:
            wrapper = pickle.load(handle)
        if "test" not in wrapper:
            raise KeyError(f"{path.name} has no test key")
        parts.append(convert_split(wrapper["test"], ids=[path.stem], require_labels=False))
    return SplitData(
        [p.ids[0] for p in parts],
        np.concatenate([p.tokens for p in parts]),
        np.concatenate([p.segments for p in parts]),
        np.concatenate([p.support for p in parts]),
        np.concatenate([p.audio for p in parts]),
        np.concatenate([p.vision for p in parts]),
        np.concatenate([p.observed for p in parts]),
    )


def fit_normalization(train: SplitData) -> dict[str, list[float]]:
    stats = {}
    for index, name in ((1, "audio"), (2, "vision")):
        values = getattr(train, name)[train.observed[:, index]]
        if not len(values):
            raise ValueError(f"training set has no observed {name} positions")
        stats[f"{name}_mean"] = values.mean(axis=0, dtype=np.float64).tolist()
        stats[f"{name}_std"] = np.maximum(values.std(axis=0, dtype=np.float64), 1e-5).tolist()
    return stats


def normalize(split: SplitData, stats: dict[str, list[float]]) -> SplitData:
    for index, name in ((1, "audio"), (2, "vision")):
        x = getattr(split, name)
        mean = np.asarray(stats[f"{name}_mean"], dtype=np.float32)
        std = np.asarray(stats[f"{name}_std"], dtype=np.float32)
        x = np.clip((x - mean) / std, -5, 5)
        x = np.where(split.observed[:, index, :, None], x, 0).astype(np.float32)
        setattr(split, name, x)
    return split


def save_stats(path: Path, stats: dict[str, list[float]]) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_stats(path: Path) -> dict[str, list[float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_gap(
    observed: np.ndarray, support: np.ndarray, *, modality: int,
    location: str, fraction: float,
) -> np.ndarray:
    """One contiguous position interval; only originally observed entries count as masked."""
    if modality not in (0, 1, 2) or location not in ("start", "middle", "end"):
        raise ValueError("invalid gap modality or location")
    if not 0 < fraction <= 1:
        raise ValueError("gap fraction must be in (0,1]")
    eligible = np.flatnonzero(observed[modality])
    if modality == 0:
        content = np.flatnonzero(support)
        if len(content) >= 2:
            eligible = eligible[(eligible != content[0]) & (eligible != content[-1])]
    gap = np.zeros_like(observed, dtype=bool)
    if len(eligible) == 0:
        return gap
    width = max(1, min(int(round(len(eligible) * fraction)), len(eligible)))
    lo, hi = int(eligible[0]), int(eligible[-1])
    if location == "start":
        start = lo
    elif location == "end":
        start = max(lo, hi - width + 1)
    else:
        start = max(lo, min((lo + hi - width + 1) // 2, hi - width + 1))
    # If the center lies in a pre-existing hole, use the closest window containing
    # observed data. Otherwise the purported corruption can remove nothing.
    starts = range(lo, max(lo, hi - width + 1) + 1)
    viable = [s for s in starts if observed[modality, s:s + width].any()]
    start = min(viable, key=lambda s: (abs(s - start), s))
    region = np.arange(start, min(start + width, observed.shape[1]))
    gap[modality, region] = observed[modality, region]
    return gap


class AlignedDataset(Dataset):
    def __init__(
        self, split: SplitData, *, mode: str = "clean", seed: int = 42,
        probability: float = 0.8, fixed: dict | None = None,
    ) -> None:
        if mode not in ("clean", "mixed", "fixed"):
            raise ValueError("mode must be clean, mixed or fixed")
        if not 0 <= probability <= 1 or (mode == "fixed" and fixed is None):
            raise ValueError("invalid masking probability or missing fixed-gap configuration")
        self.split, self.mode, self.seed, self.probability, self.fixed = split, mode, seed, probability, fixed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict:
        s = self.split
        observed = s.observed[index].copy()
        gap = np.zeros_like(observed, dtype=bool)
        if self.mode != "clean":
            rng = random.Random(self.seed + index * 7919 + self.epoch * 1000003)
            if self.mode == "fixed" or rng.random() < self.probability:
                if self.mode == "fixed":
                    options = [self.fixed]
                else:
                    count = 2 if rng.random() < 0.2 else 1
                    modalities = rng.sample(range(3), count)
                    options = [dict(modality=m, location=rng.choice(("start", "middle", "end")),
                                    fraction=rng.choice((0.15, 0.30, 0.45))) for m in modalities]
                for option in options:
                    gap |= make_gap(observed, s.support[index], **option)
        observed &= ~gap
        tokens = s.tokens[index].copy()
        tokens[~observed[0]] = 0
        audio = s.audio[index].copy()
        vision = s.vision[index].copy()
        audio[~observed[1]] = 0
        vision[~observed[2]] = 0
        item = {
            "id": s.ids[index], "tokens": torch.from_numpy(tokens),
            "segments": torch.from_numpy(s.segments[index]),
            "support": torch.from_numpy(s.support[index]),
            "audio": torch.from_numpy(audio), "vision": torch.from_numpy(vision),
            "observed": torch.from_numpy(observed), "artificial_gap": torch.from_numpy(gap),
        }
        if s.labels is not None:
            item["label"] = torch.tensor(s.labels[index], dtype=torch.long)
            item["intensity"] = torch.tensor(s.intensity[index], dtype=torch.float32)
        return item
