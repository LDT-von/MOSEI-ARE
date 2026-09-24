"""Focused structural tests for Problem 3 evidence utilities."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.gap_slot_emo import GapSlotEmo
from problem2.data import AlignedDataset, convert_split
from problem3.data import attachment4_to_split, grid_seconds, load_attachment4
from problem3.explain import (
    collect_batch_evidence,
    modality_contributions,
    modality_weights_from_model,
    window_evidence,
    write_attachment4_csv,
    write_records,
)


def _fake_split():
    bert = np.zeros((2, 3, 50), dtype=np.float32)
    bert[:, 0, :8] = np.array([101, 11, 12, 13, 14, 15, 16, 102])
    bert[:, 1, :8] = 1
    audio = np.zeros((2, 50, 74), dtype=np.float32)
    vision = np.zeros((2, 50, 35), dtype=np.float32)
    audio[:, 1:7] = 1
    vision[:, 1:7] = 2
    return convert_split({"id": ["one", "two"], "text_bert": bert, "audio": audio, "vision": vision,
                          "classification_labels": [0, 2], "regression_labels": [-2.0, 1.5]})


def _fake_attachment4():
    samples = []
    for index in range(20):
        text_bert = np.zeros((3, 50), dtype=np.float32)
        text_bert[0, :8] = np.array([101, 11, 12, 13, 14, 15, 16, 102])
        text_bert[1, :8] = 1
        audio = np.zeros((50, 74), dtype=np.float32)
        audio[1:7] = 0.1
        vision = np.zeros((50, 35), dtype=np.float32)
        vision[1:7] = 0.1
        samples.append(type("Sample", (), {
            "sample_id": f"{index + 1:02d}",
            "raw_text": f"sample {index}",
            "text_bert": text_bert,
            "audio": audio,
            "vision": vision,
            "text": None,
        }))
    return samples


class Problem3ContractTests(unittest.TestCase):
    def test_grid_seconds_is_fifty_half_second_centres(self):
        seconds = grid_seconds()
        self.assertEqual(seconds.shape, (50,))
        self.assertAlmostEqual(float(seconds[0]), 0.1)
        self.assertAlmostEqual(float(seconds[-1]), 9.9)
        self.assertTrue(np.all(np.diff(seconds) > 0))

    def test_attachment4_split_round_trip(self):
        samples = _fake_attachment4()
        split = attachment4_to_split(samples)
        self.assertEqual(len(split), 20)
        self.assertEqual(split.tokens.shape, (20, 50))
        self.assertEqual(split.audio.shape, (20, 50, 74))
        self.assertEqual(split.vision.shape, (20, 50, 35))
        self.assertIsNone(split.labels)
        self.assertEqual({sample.sample_id for sample in samples}, set(split.ids))

    def test_modality_contributions_match_clean_pred(self):
        model = GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True).eval()
        sample = AlignedDataset(_fake_split())[0]
        contribution = modality_contributions(model, sample, torch.device("cpu"))
        self.assertIn(contribution.dominant_modality, {"text", "audio", "vision"})
        self.assertEqual(set(contribution.polarity_delta), {"text", "audio", "vision"})
        total = sum(contribution.normalized_magnitude.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_window_evidence_skips_fully_missing_regions(self):
        model = GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True).eval()
        sample = AlignedDataset(_fake_split())[0]
        evidence = window_evidence(model, sample, torch.device("cpu"), modality="audio", window_length=4, top_k=2)
        self.assertGreater(len(evidence.windows), 0)
        for window in evidence.windows:
            self.assertGreaterEqual(window["start_position"], 0)
            self.assertLessEqual(window["end_position"], 50)
            self.assertGreater(window["end_seconds"], window["start_seconds"])

    def test_auxiliary_modality_weights_sum_to_one_when_supported(self):
        model = GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True).eval()
        sample = AlignedDataset(_fake_split())[0]
        weights = modality_weights_from_model(model, sample, torch.device("cpu"))
        self.assertEqual(set(weights), {"text", "audio", "vision"})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

    def test_collect_batch_evidence_emits_full_record(self):
        model = GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True).eval()
        dataset = AlignedDataset(_fake_split())
        records = collect_batch_evidence(model, dataset, [0, 1], torch.device("cpu"), window_length=4, top_k=2)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertIn("modality_contribution", record)
            self.assertIn("window_evidence", record)
            self.assertEqual(set(record["window_evidence"]), {"text", "audio", "vision"})

    def test_attachment4_csv_writer_preserves_every_sample(self):
        rows = [{
            "sample_id": "01",
            "prediction": {"polarity": 2, "intensity": 1.0},
            "raw_text": "hello world",
            "video_filename": "01.mp4",
            "modality_contribution": {
                "dominant_modality": "text",
                "polarity_delta": {"text": 0.5, "audio": 0.2, "vision": 0.3},
                "intensity_delta": {"text": 0.1, "audio": 0.2, "vision": 0.3},
                "clean_logit": 0.0, "clean_intensity": 0.0, "clean_pred": 2,
                "normalized_magnitude": {"text": 0.5, "audio": 0.2, "vision": 0.3},
            },
            "window_evidence": {
                "text": {"windows": [{"start_seconds": 0.3, "end_seconds": 1.5, "logit_drop": 0.4}]},
                "audio": {"windows": []},
                "vision": {"windows": []},
            },
        }]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.csv"
            write_attachment4_csv(path, rows)
            content = path.read_text(encoding="utf-8-sig")
            self.assertIn("sample_id,polarity", content)
            self.assertIn("01", content)
            self.assertIn("hello world", content)

    def test_records_json_is_serialisable(self):
        records = collect_batch_evidence(
            GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True).eval(),
            AlignedDataset(_fake_split()), [0], torch.device("cpu"), window_length=4, top_k=2,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.json"
            write_records(path, records)
            restored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(restored), 1)


if __name__ == "__main__":
    unittest.main()
