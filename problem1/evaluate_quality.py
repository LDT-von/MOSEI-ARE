"""Prepare a stratified human-review sheet and score reviewed word alignment.

This is an audit of the saved outputs. It does not rerun extraction or train a model.
Run from problem1: python evaluate_quality.py prepare / score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
QA = BASE / "qa"
STEP = 0.2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare() -> None:
    if (QA / "manual_review_words.csv").exists():
        raise FileExistsError("Review sheet already exists; move it before preparing a new sample")
    summaries = read_csv(OUT / "summary_100.csv")
    passed = sorted((row for row in summaries if row["alignment_status"] == "passed_automatic_screen"),
                    key=lambda row: float(row["ctc_mean_score"]))
    masked = [row for row in summaries if row["alignment_status"] == "unreliable_masked"]
    assert len(summaries) == 100 and len(passed) == 79 and len(masked) == 21
    rng = random.Random(20260923)
    halfway = len(passed) // 2
    chosen = [("high_pass", row) for row in rng.sample(passed[halfway:], 12)]
    chosen += [("lower_pass", row) for row in rng.sample(passed[:halfway], 9)]
    silent_ids = {"-mJ2ud6oKI8__1", "-mJ2ud6oKI8__2"}
    forced = [row for row in masked if row["sample_id"] in silent_ids]
    chosen += [("masked", row) for row in forced]
    chosen += [("masked", row) for row in rng.sample([r for r in masked if r not in forced], 9 - len(forced))]
    QA.mkdir(exist_ok=True)
    clip_rows, word_rows = [], []
    for stratum, row in chosen:
        sid = row["sample_id"]
        meta = json.loads((OUT / "metadata" / f"{sid}.json").read_text(encoding="utf-8"))
        video = (BASE.parent.parent / meta["source_relpath"]).resolve()
        clip_rows.append({"stratum": stratum, "sample_id": sid, "video_path": str(video),
                          "duration_s": row["duration_seconds"], "ctc_mean_score": row["ctc_mean_score"],
                          "automatic_status": row["alignment_status"], "audio_transcript_match_yes_no_uncertain": "",
                          "face_crop_valid_yes_no_uncertain": "", "visible_speaker_matches_audio_yes_no_uncertain": "",
                          "notes": ""})
        words = meta["words"]
        for j in sorted({0, len(words) // 2, len(words) - 1}):
            word = words[j]
            word_rows.append({"stratum": stratum, "sample_id": sid, "video_path": str(video),
                              "word_index": j, "word_count": len(words), "word": word["word"],
                              "clip_duration_s": row["duration_seconds"],
                              "proposed_start_s": word["start"], "proposed_end_s": word["end"],
                              "ctc_score": round(word["ctc_score"], 6),
                              "default_timed_text_usable": int(word["usable_for_timed_text"]),
                              "human_word_audible_yes_no_uncertain": "", "human_start_s": "", "human_end_s": "",
                              "notes": ""})
    write_csv(QA / "manual_review_clips.csv", clip_rows)
    write_csv(QA / "manual_review_words.csv", word_rows)
    print(f"Prepared {len(clip_rows)} clips and {len(word_rows)} word checks in {QA}")


def bins(start: float, end: float, duration: float) -> set[int]:
    count = int((duration + STEP - 1e-9) / STEP)
    return {k for k in range(count) if min((k + 1) * STEP, duration, end) > max(k * STEP, start)}


def score() -> None:
    words = read_csv(QA / "manual_review_words.csv")
    clips = read_csv(QA / "manual_review_clips.csv")
    review_fields = ("audio_transcript_match_yes_no_uncertain", "face_crop_valid_yes_no_uncertain",
                     "visible_speaker_matches_audio_yes_no_uncertain")
    result: dict[str, object] = {
        "selected_clips": len(clips),
        "reviewed_clips": sum(any(c[field].strip().lower() in ("yes", "no", "uncertain")
                                  for field in review_fields) for c in clips),
        "selected_words": len(words), "strata": {},
    }
    for stratum in ("high_pass", "lower_pass", "masked"):
        selected = [w for w in words if w["stratum"] == stratum]
        audible = [w for w in selected if w["human_word_audible_yes_no_uncertain"].strip().lower() == "yes"
                   and w["human_start_s"].strip() and w["human_end_s"].strip()]
        inaudible = [w for w in selected if w["human_word_audible_yes_no_uncertain"].strip().lower() == "no"]
        errors, uniform_errors, near_01, near_02, ious = [], [], [], [], []
        for w in audible:
            start, end = float(w["human_start_s"]), float(w["human_end_s"])
            duration = float(w["clip_duration_s"])
            if not 0 <= start < end <= duration + 1e-4:
                raise ValueError(f"Invalid human time for {w['sample_id']} word {w['word']}")
            proposed_start, proposed_end = float(w["proposed_start_s"]), float(w["proposed_end_s"])
            boundary = (abs(proposed_start - start), abs(proposed_end - end))
            errors.extend(boundary)
            near_01.append(max(boundary) <= .1)
            near_02.append(max(boundary) <= .2)
            j, n = int(w["word_index"]), int(w["word_count"])
            uniform_errors.extend((abs(duration * j / n - start), abs(duration * (j + 1) / n - end)))
            predicted_bins, true_bins = bins(proposed_start, proposed_end, duration), bins(start, end, duration)
            ious.append(len(predicted_bins & true_bins) / max(1, len(predicted_bins | true_bins)))
        result["strata"][stratum] = {
            "selected_word_count": len(selected), "timed_human_truth_count": len(audible),
            "inaudible_or_mismatched_count": len(inaudible),
            "mean_absolute_boundary_error_s": statistics.mean(errors) if errors else None,
            "uniform_baseline_boundary_error_s": statistics.mean(uniform_errors) if uniform_errors else None,
            "both_boundaries_within_0_1_s": statistics.mean(near_01) if near_01 else None,
            "both_boundaries_within_0_2_s": statistics.mean(near_02) if near_02 else None,
            "mean_0_2_s_bin_iou": statistics.mean(ious) if ious else None,
            "audible_word_retention": statistics.mean(int(w["default_timed_text_usable"]) for w in audible) if audible else None,
            "mismatched_word_rejection": statistics.mean(1 - int(w["default_timed_text_usable"]) for w in inaudible) if inaudible else None,
        }
    result["clip_checks"] = {}
    for field in ("audio_transcript_match_yes_no_uncertain", "face_crop_valid_yes_no_uncertain",
                  "visible_speaker_matches_audio_yes_no_uncertain"):
        entries = [c[field].strip().lower() for c in clips]
        result["clip_checks"][field] = {label: entries.count(label) for label in ("yes", "no", "uncertain", "")}
    path = QA / "manual_quality_report.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "score": score}[args.action]()
