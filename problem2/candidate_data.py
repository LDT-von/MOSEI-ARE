"""Read the Attachment 2 training text teacher without changing the final data path."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch

from problem2.data import AlignedDataset, SplitData, convert_split


def load_candidate_splits(path: Path) -> tuple[dict[str, SplitData], np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not {"train", "valid", "test"}.issubset(raw):
        raise KeyError("Attachment 2 must contain train, valid and test")
    splits = {name: convert_split(raw[name]) for name in ("train", "valid", "test")}
    if len(set().union(*(set(split.ids) for split in splits.values()))) != sum(map(len, splits.values())):
        raise ValueError("sample IDs overlap across train/valid/test")
    if "text" not in raw["train"]:
        raise KeyError("candidate text distillation requires Attachment 2 train.text")
    teacher = np.asarray(raw["train"]["text"], dtype=np.float32)
    if teacher.shape != (len(splits["train"]), 50, 768) or not np.isfinite(teacher).all():
        raise ValueError("train.text teacher must be finite with shape (N,50,768)")
    return splits, teacher


class TeacherAlignedDataset(AlignedDataset):
    def __init__(self, split: SplitData, teacher: np.ndarray, **kwargs) -> None:
        super().__init__(split, **kwargs)
        if teacher.shape != (len(split), 50, 768):
            raise ValueError("teacher text is not aligned with training split")
        self.teacher = teacher

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        item["teacher_text"] = torch.from_numpy(self.teacher[index])
        return item
