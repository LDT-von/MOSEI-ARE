"""Focused tests for temporal correspondence, missing data and source preservation."""
import unittest
import numpy as np
from pipeline import make_grid, overlap_pool, transcript_units, acoustic_features
from apply_quality_gate import lexical_recall


class AlignmentTests(unittest.TestCase):
    def test_transcript_agreement_ignores_extra_leading_audio(self):
        self.assertEqual(lexical_recall('THEY FIND ANSWERS', 'EXTRA WORDS THEY FIND ANSWERS'), 1.)
        self.assertEqual(lexical_recall('TECHNICAL FIELDS INSTRUMENT', 'HAWKS HEARING EYESIGHT'), 0.)

    def test_preserves_last_fractional_interval(self):
        grid = make_grid(.45)
        np.testing.assert_allclose(grid, [[0, .2], [.2, .4], [.4, .45]])
        self.assertEqual(len(make_grid(34.567)), 173)

    def test_exact_overlap_and_missing_region(self):
        native = np.array([[0, .1], [.1, .3], [.4, .6]])
        x = np.array([[1.], [4.], [100.]])
        values, mask, weights = overlap_pool(x, native, make_grid(.6), [True, True, False])
        np.testing.assert_allclose(values[:, 0], [2.5, 4, 0], atol=1e-5)
        np.testing.assert_array_equal(mask, [True, True, False])
        np.testing.assert_allclose(weights.sum(1), [1, 1, 0])

    def test_time_shift_changes_correspondence(self):
        values, mask, _ = overlap_pool(np.array([[7.]]), np.array([[.25, .35]]), make_grid(.6))
        np.testing.assert_array_equal(mask, [False, True, False])
        np.testing.assert_allclose(values[:, 0], [0, 7, 0])

    def test_original_text_and_numbers_remain_traceable(self):
        text = "In 2008, Kentucky’s 10th show had 1,500 guests."
        words = transcript_units(text)
        self.assertTrue(all(text[w['char_start']:w['char_end']] == w['word'] for w in words))
        self.assertEqual(words[1]['normalized'], 'TWO THOUSAND EIGHT')
        self.assertEqual(words[2]['normalized'], "KENTUCKY'S")
        self.assertEqual(words[3]['normalized'], 'TENTH')
        self.assertEqual(words[6]['normalized'], 'ONE THOUSAND FIVE HUNDRED')

    def test_silence_is_valid_not_missing(self):
        feat, intervals, valid, _ = acoustic_features(np.zeros(1600, np.float32), np.ones(1600, bool))
        self.assertEqual(feat.shape[1], 33)
        self.assertTrue(valid.all()); self.assertTrue(np.isfinite(feat).all())
        self.assertTrue((feat[:, -1] == 0).all())
        self.assertAlmostEqual(float(intervals[-1, 1]), .1, places=6)


if __name__ == '__main__':
    unittest.main()
