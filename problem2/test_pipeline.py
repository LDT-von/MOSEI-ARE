"""Focused contract checks; these do not establish predictive performance."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch

from models.gap_slot_emo import GapSlotEmo
from problem2.data import AlignedDataset, convert_split, make_gap
from problem2.scripts.run import _checkpoint, _loader, _predict


def fake_split():
    bert = np.zeros((2, 3, 50), dtype=np.float32)
    bert[:, 0, :8] = np.array([101, 11, 12, 13, 14, 15, 16, 102])
    bert[:, 1, :8] = 1
    audio = np.zeros((2, 50, 74), dtype=np.float32)
    vision = np.zeros((2, 50, 35), dtype=np.float32)
    audio[:, 1:7] = 1
    vision[:, 1:7] = 2
    return convert_split({"id": ["one", "two"], "text_bert": bert, "audio": audio, "vision": vision,
                          "classification_labels": [0, 2], "regression_labels": [-2.0, 1.5]})


class Problem2ContractTests(unittest.TestCase):
    def test_gap_is_contiguous_and_only_masks_original_evidence(self):
        split = fake_split()
        gap = make_gap(split.observed[0], split.support[0], modality=0, location="middle", fraction=0.5)
        positions = np.flatnonzero(gap[0])
        self.assertTrue(len(positions) > 0)
        self.assertTrue(np.all(np.diff(positions) == 1))
        self.assertFalse(gap[0, 0] or gap[0, 7])
        self.assertTrue(np.all(gap <= split.observed[0]))

    def test_fixed_validation_mask_is_reproducible(self):
        dataset = AlignedDataset(fake_split(), mode="fixed", fixed={"modality": 1, "location": "start", "fraction": 0.5})
        first, second = dataset[0], dataset[0]
        self.assertTrue(torch.equal(first["artificial_gap"], second["artificial_gap"]))
        self.assertTrue(torch.all(first["audio"][~first["observed"][1]] == 0))

    def test_model_masks_missing_values_and_is_deterministic(self):
        model = GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True).eval()
        sample = AlignedDataset(fake_split(), mode="fixed", fixed={"modality": 1, "location": "middle", "fraction": 0.5})[0]
        args = {key: sample[key].unsqueeze(0) for key in ("tokens", "segments", "audio", "vision", "support", "observed")}
        with torch.inference_mode():
            a = model(**args, return_attention=True)
            b = model(**args)
            tampered = dict(args)
            tampered["audio"] = args["audio"].clone()
            tampered["audio"][~args["observed"][:, 1]] = 999
            c = model(**tampered)
        self.assertTrue(torch.allclose(a["logits"], b["logits"]))
        self.assertTrue(torch.allclose(a["logits"], c["logits"], atol=1e-6))
        self.assertTrue(torch.isfinite(a["regression"]).all())
        self.assertLessEqual(abs(float(a["regression"].item())), 3)
        self.assertTrue(a["effective_observed"][0, 1, sample["artificial_gap"][1]].all())
        self.assertTrue(torch.all(a["slot_attention"][0, 0, :, ~args["observed"][0, 0]] == 0))
        self.assertFalse(a["effective_observed"][0, 1:, [0, 7]].any())

    def test_middle_gap_on_sparse_observations_removes_evidence(self):
        observed = np.zeros((3, 10), dtype=bool)
        observed[1, [1, 8]] = True
        gap = make_gap(observed, np.ones(10, dtype=bool), modality=1, location="middle", fraction=0.5)
        self.assertEqual(int(gap.sum()), 1)
        self.assertTrue(np.all(gap <= observed))

    def test_ablations_share_initial_backbone_parameters(self):
        torch.manual_seed(42)
        base = GapSlotEmo(vocab_size=128, dim=32, slots=4, use_gap_repair=False)
        torch.manual_seed(42)
        repaired = GapSlotEmo(vocab_size=128, dim=32, slots=4, use_gap_repair=True)
        state = repaired.state_dict()
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_all_missing_is_finite_and_backpropagates(self):
        model = GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True)
        args = {
            "tokens": torch.zeros(2, 50, dtype=torch.long),
            "segments": torch.zeros(2, 50, dtype=torch.long),
            "audio": torch.zeros(2, 50, 74),
            "vision": torch.zeros(2, 50, 35),
            "support": torch.ones(2, 50, dtype=torch.bool),
            "observed": torch.zeros(2, 3, 50, dtype=torch.bool),
        }
        result = model(**args)
        self.assertTrue(torch.isfinite(result["logits"]).all())
        self.assertTrue(torch.isfinite(result["regression"]).all())
        self.assertTrue(torch.equal(result["modality_weights"], torch.zeros_like(result["modality_weights"])))
        (result["logits"].square().mean() + result["regression"].square().mean()).backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_unlabeled_prediction_path_preserves_sample_ids(self):
        split = fake_split()
        split.labels = None
        split.intensity = None
        model = GapSlotEmo(vocab_size=128, dim=32, slots=4, max_len=50, use_gap_repair=True)
        result = _predict(model, _loader(AlignedDataset(split), 2), torch.device("cpu"))
        self.assertEqual(result["ids"], ["one", "two"])
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(len(result["labels"]), 0)

    def test_official_inference_rejects_smoke_checkpoint(self):
        qa = Path(__file__).resolve().parent / "qa"
        qa.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=qa) as folder:
            self.assertEqual(Path(folder).resolve().parent, qa.resolve())
            checkpoint = Path(folder) / "smoke.pt"
            torch.save({"provenance": {"run_kind": "smoke_only_not_for_submission"}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "smoke checkpoint"):
                _checkpoint(checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
