"""Aligned data access for Attachment 2 (train/valid/test) and Attachment 4."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from problem2.data import SplitData, convert_split


@dataclass
class Attachment4Sample:
    sample_id: str
    raw_text: str
    text_bert: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    text: np.ndarray | None = None


def load_attachment2_aligned(path: Path) -> dict[str, SplitData]:
    """Attachment 2: train / valid / test. Reuse problem2 loader for consistency."""
    from problem2.data import load_attachment2  # local import keeps modules light
    return load_attachment2(path)


def load_attachment4(directory: Path) -> list[Attachment4Sample]:
    """Read every aligned-version pkl. Each file holds a single sample."""
    files = sorted(directory.glob("*.pkl"))
    if not files:
        raise ValueError(f"no aligned pkl files found under {directory}")
    samples: list[Attachment4Sample] = []
    for path in files:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        text_bert = np.asarray(raw["text_bert"])
        if text_bert.dtype.kind != "i":
            text_bert = text_bert.astype(np.int64, copy=False)
        audio = np.asarray(raw["audio"], dtype=np.float32)
        vision = np.asarray(raw["vision"], dtype=np.float32)
        if text_bert.shape != (3, 50):
            raise ValueError(f"{path.name}: text_bert must be (3, 50), got {text_bert.shape}")
        if audio.shape != (50, 74) or vision.shape != (50, 35):
            raise ValueError(f"{path.name}: bad aligned shapes {audio.shape} {vision.shape}")
        samples.append(Attachment4Sample(
            sample_id=str(raw.get("id", path.stem)),
            raw_text=str(raw["raw_text"]),
            text_bert=text_bert,
            audio=audio,
            vision=vision,
            text=np.asarray(raw["text"], dtype=np.float32) if "text" in raw else None,
        ))
    if len(samples) != 20:
        raise ValueError(f"expected 20 attachment-4 samples, got {len(samples)}")
    return samples


def attachment4_to_split(samples: list[Attachment4Sample]) -> SplitData:
    """Stack 20 samples into a SplitData structure so problem2 utilities can consume them."""
    text_bert = np.stack([s.text_bert for s in samples], axis=0).astype(np.float32)
    audio = np.stack([s.audio for s in samples], axis=0).astype(np.float32)
    vision = np.stack([s.vision for s in samples], axis=0).astype(np.float32)
    return convert_split(
        {"text_bert": text_bert, "audio": audio, "vision": vision},
        ids=[s.sample_id for s in samples],
        require_labels=False,
    )


def grid_seconds(length: int = 50, hop: float = 0.2) -> np.ndarray:
    """Aligned grid centre times in seconds for the 50-position sequence."""
    return (np.arange(length) + 0.5) * hop


def save_stats(path: Path, stats: dict[str, list[float]]) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
