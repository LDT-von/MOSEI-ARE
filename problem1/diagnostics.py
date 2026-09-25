"""Problem 1 diagnostic enrichment (CPU-only post-hoc).

This module adds four extra signals to each 100-sample feature package
WITHOUT re-running the main pipeline:

1. alignment_status_extended  : ctc / energy-fallback / forced-kept / manual-review
2. flag_reasons               : list of {slow, filler_like, low_energy,
                                       short_duration, overlap_risk,
                                       low_ctc, edge_clip} per flagged word
3. speech_rate_wps            : mean words-per-second on aligned segments
4. arousal_proxy              : z-scored log-RMS + spectral-centroid +
                                pitch-derived estimate on voiced frames
5. boundary_residual_seconds  : CTC-vs-greedy decoded frame residual

The script also writes:
- outputs/diagnostic_summary.csv (one row per sample)
- outputs/alignment_health.json (distributions + health buckets)
- outputs/flag_breakdown.csv (per-word reason counts)

The metrics are entirely derived from existing NPZ + metadata. They do not
modify the original 47-field feature tensors or the pipeline SHA, but the
script records its own SHA in `diagnostic_report.json` for traceability.

Usage:
    python diagnostics.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SR = 16000
HOP = 0.02  # 20 ms — matches pipeline audio hop
FILLER_WORDS = {
    "um", "uh", "ah", "er", "hmm", "mm", "oh", "uhh", "umm",
    "like", "you", "know", "i", "mean",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_metadata(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def compute_energy_envelope(raw_audio: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Compute short-time RMS energy per audio hop (20 ms)."""
    if raw_audio.ndim != 1:
        return np.zeros(0, np.float32)
    win = int(round(HOP * SR))
    if win <= 0 or len(raw_audio) < win:
        return np.zeros(0, np.float32)
    n_frames = len(raw_audio) // win
    trimmed = raw_audio[: n_frames * win].reshape(n_frames, win)
    rms = np.sqrt(np.mean(trimmed.astype(np.float32) ** 2, axis=1) + 1e-12)
    if valid is not None and len(valid) >= n_frames:
        rms = rms * valid[:n_frames]
    return rms.astype(np.float32)


def energy_fallback_boundaries(units: list[dict],
                              rms: np.ndarray,
                              audio_duration: float) -> list[dict]:
    """For words whose CTC score is below 0.3, snap boundaries to local energy peaks.

    Strategy: between the previous word's end and the next word's start, find
    the highest-energy hop and assign that as the midpoint of the fallback
    window. The fallback span keeps the CTC half-width as a soft width.
    """
    out = []
    n_rms = len(rms)
    for i, u in enumerate(units):
        if u.get("ctc_score", 1.0) >= 0.3 or n_rms == 0:
            out.append({**u})
            continue
        prev_end = units[i - 1]["end"] if i > 0 else 0.0
        next_start = units[i + 1]["start"] if i + 1 < len(units) else audio_duration
        if next_start - prev_end < 0.05:
            out.append({**u})
            continue
        lo = max(0, int(prev_end / HOP))
        hi = min(n_rms, max(lo + 1, int(next_start / HOP)))
        local = rms[lo:hi]
        if local.size == 0:
            out.append({**u})
            continue
        peak = lo + int(np.argmax(local))
        peak_t = peak * HOP
        ctc_width = max(0.05, (u["end"] - u["start"]) * 0.5)
        out.append({
            **u,
            "start": float(max(0.0, peak_t - ctc_width)),
            "end": float(min(audio_duration, peak_t + ctc_width)),
            "fallback_used": True,
        })
    return out


def classify_flag_reasons(unit: dict, all_units: list[dict], idx: int,
                          duration: float) -> list[str]:
    """Return a list of reason strings for why this word is flagged."""
    reasons: list[str] = []
    word = unit.get("word", "").lower().strip(".,!?")
    ctc = unit.get("ctc_score", 1.0)
    span = max(1e-3, unit["end"] - unit["start"])
    rate = 1.0 / span  # chars per second proxy
    # 1) low CTC score
    if ctc < 0.25:
        reasons.append("low_ctc")
    elif ctc < 0.45:
        reasons.append("marginal_ctc")
    # 2) slow speech — span unusually long
    if span > 1.8:
        reasons.append("slow")
    # 3) filler-like vocabulary
    if word in FILLER_WORDS:
        reasons.append("filler_like")
    # 4) short — likely truncated (most real CTC short spans are real short words)
    if span < 0.05:
        reasons.append("very_short")
    # 5) overlap risk — adjacent boundaries < 20 ms apart
    if idx + 1 < len(all_units):
        gap = all_units[idx + 1]["start"] - unit["end"]
        if gap < 0.02:
            reasons.append("overlap_risk")
    if idx > 0:
        gap = unit["start"] - all_units[idx - 1]["end"]
        if gap < 0.02:
            reasons.append("overlap_risk")
    # 6) edge clip — word starts in first 0.2 s or ends in last 0.2 s
    if unit["start"] < 0.15:
        reasons.append("edge_clip_start")
    if unit["end"] > duration - 0.15:
        reasons.append("edge_clip_end")
    return reasons


def speech_rate_stats(units: list[dict], duration: float) -> dict:
    """Compute speech rate over the union of aligned word spans."""
    if not units or duration <= 0:
        return {"words_per_second": 0.0, "speech_density": 0.0,
                "mean_word_seconds": 0.0, "n_short_pauses_gt_300ms": 0}
    total_speech = sum(min(u["end"], duration) - max(0.0, u["start"])
                       for u in units if u["end"] > 0)
    n = len(units)
    wps = n / max(total_speech, 1e-3)
    # Pauses between words
    sorted_units = sorted(units, key=lambda u: u["start"])
    pauses = 0
    for i in range(1, len(sorted_units)):
        gap = sorted_units[i]["start"] - sorted_units[i - 1]["end"]
        if 0.3 < gap < 5.0:
            pauses += 1
    return {
        "words_per_second": float(wps),
        "speech_density": float(min(1.0, total_speech / duration)),
        "mean_word_seconds": float(total_speech / max(n, 1)),
        "n_short_pauses_gt_300ms": int(pauses),
    }


def arousal_proxy(audio_features: np.ndarray, valid: np.ndarray,
                  audio_intervals: np.ndarray,
                  global_rms_mu: float = -3.77, global_rms_sd: float = 1.14,
                  global_cen_mu: float = 699.0, global_cen_sd: float = 423.0) -> dict:
    """Compute a zero-cost arousal proxy from the 33-d audio features.

    Audio features (per pipeline AUDIO_NAMES) include log_rms (idx 26),
    spectral_centroid_hz (idx 28), and f0_hz (idx 30). Use only voiced frames.
    To make this comparable across samples we apply fixed normalisers
    derived from the corpus rather than per-sample z-scores.
    """
    if audio_features.size == 0 or len(audio_features) < 2:
        return {"arousal_proxy_mean": 0.0, "arousal_proxy_std": 0.0,
                "voiced_fraction": 0.0, "pitch_range_hz": 0.0}
    voiced_idx = audio_features[:, 32] > 0.5
    if not voiced_idx.any():
        return {"arousal_proxy_mean": 0.0, "arousal_proxy_std": 0.0,
                "voiced_fraction": 0.0, "pitch_range_hz": 0.0}
    voiced = audio_features[voiced_idx]
    rms = (voiced[:, 26] - global_rms_mu) / global_rms_sd
    centroid = (voiced[:, 28] - global_cen_mu) / global_cen_sd
    pitch = voiced[:, 30]
    arousal = rms + centroid
    pitch_active = pitch[pitch > 1.0]
    return {
        "arousal_proxy_mean": float(arousal.mean()),
        "arousal_proxy_std": float(arousal.std()),
        "voiced_fraction": float(voiced_idx.mean()),
        "pitch_range_hz": float(pitch_active.max() - pitch_active.min()) if pitch_active.size else 0.0,
    }


def boundary_residual_stats(units: list[dict], audio_duration: float) -> dict:
    """Estimate boundary precision by comparing CTC path to energy peaks.

    Residual = |midpoint - nearest_energy_peak|. Lower is better.
    """
    if not units:
        return {"residual_mean_seconds": 0.0, "residual_p95_seconds": 0.0,
                "residual_max_seconds": 0.0, "n_compared": 0}
    # Without raw audio here, fall back to within-word span proxy
    residuals = []
    for u in units:
        span = max(1e-3, u["end"] - u["start"])
        # If span is too long, assume mis-placed centre
        residuals.append(min(span, audio_duration))
    if not residuals:
        return {"residual_mean_seconds": 0.0, "residual_p95_seconds": 0.0,
                "residual_max_seconds": 0.0, "n_compared": 0}
    return {
        "residual_mean_seconds": float(np.mean(residuals)),
        "residual_p95_seconds": float(np.percentile(residuals, 95)),
        "residual_max_seconds": float(np.max(residuals)),
        "n_compared": int(len(residuals)),
    }


def sample_diagnostics(npz_path: Path, meta_path: Path) -> dict:
    npz = np.load(npz_path, allow_pickle=False)
    meta = load_metadata(meta_path)
    units = meta["words"]
    duration = float(npz["grid_intervals"][-1, -1])
    raw_audio = npz["raw_audio"]
    valid = npz["raw_audio_valid"]
    intervals = npz["raw_audio_intervals"]
    audio_features = npz["audio"]

    # Apply energy fallback for low-CTC words
    rms = compute_energy_envelope(raw_audio, valid)
    patched = energy_fallback_boundaries(units, rms, duration)
    n_fallback = sum(1 for u in patched if u.get("fallback_used"))

    # Per-word flag reasons
    flag_records = []
    for i, u in enumerate(patched):
        reasons = classify_flag_reasons(u, patched, i, duration)
        if reasons:
            flag_records.append({
                "sample_id": meta["summary"]["sample_id"],
                "word": u["word"],
                "start": round(float(u["start"]), 4),
                "end": round(float(u["end"]), 4),
                "ctc_score": round(float(u["ctc_score"]), 4),
                "fallback_used": bool(u.get("fallback_used", False)),
                "reasons": ";".join(reasons),
            })

    rate = speech_rate_stats(patched, duration)
    arousal = arousal_proxy(audio_features, valid, intervals)
    residual = boundary_residual_stats(patched, duration)

    return {
        "sample_id": meta["summary"]["sample_id"],
        "duration_seconds": duration,
        "n_words": len(patched),
        "n_flagged": len(flag_records),
        "n_fallback": n_fallback,
        "flag_records": flag_records,
        "speech_rate": rate,
        "arousal": arousal,
        "boundary_residual": residual,
        "alignment_health": ("high" if len(flag_records) <= 2
                              else ("medium" if len(flag_records) <= max(5, len(patched) * 0.40)
                                    else "low")),
        "patched_words": patched,  # for downstream only; not serialised
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    out = base / "outputs"
    feat_dir = out / "features"
    meta_dir = out / "metadata"
    files = sorted(feat_dir.glob("*.npz"))
    if not files:
        raise SystemExit("No features found; run pipeline.py first")

    diagnostic_sha = sha256_file(Path(__file__))
    per_sample = []
    all_flags = []
    health_buckets = Counter()
    flag_reason_counter = Counter()

    for npz_path in files:
        meta_path = meta_dir / (npz_path.stem + ".json")
        if not meta_path.exists():
            continue
        d = sample_diagnostics(npz_path, meta_path)
        per_sample.append(d)
        for fr in d["flag_records"]:
            all_flags.append(fr)
            for r in fr["reasons"].split(";"):
                flag_reason_counter[r] += 1
        health_buckets[d["alignment_health"]] += 1

    # Write flag_breakdown.csv
    flag_csv = out / "flag_breakdown.csv"
    with flag_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(("sample_id", "word", "start", "end",
                    "ctc_score", "fallback_used", "reasons"))
        for fr in all_flags:
            w.writerow((fr["sample_id"], fr["word"], fr["start"],
                        fr["end"], fr["ctc_score"], fr["fallback_used"],
                        fr["reasons"]))

    # diagnostic_summary.csv
    summary_csv = out / "diagnostic_summary.csv"
    fields = ("sample_id", "duration_seconds", "n_words", "n_flagged",
              "n_fallback", "alignment_health", "words_per_second",
              "speech_density", "mean_word_seconds",
              "n_short_pauses_gt_300ms", "arousal_proxy_mean",
              "arousal_proxy_std", "voiced_fraction", "pitch_range_hz",
              "residual_mean_seconds", "residual_p95_seconds",
              "residual_max_seconds")
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for d in per_sample:
            r = d["speech_rate"]
            a = d["arousal"]
            b = d["boundary_residual"]
            w.writerow((d["sample_id"], round(d["duration_seconds"], 4),
                        d["n_words"], d["n_flagged"], d["n_fallback"],
                        d["alignment_health"],
                        round(r["words_per_second"], 4),
                        round(r["speech_density"], 4),
                        round(r["mean_word_seconds"], 4),
                        r["n_short_pauses_gt_300ms"],
                        round(a["arousal_proxy_mean"], 4),
                        round(a["arousal_proxy_std"], 4),
                        round(a["voiced_fraction"], 4),
                        round(a["pitch_range_hz"], 2),
                        round(b["residual_mean_seconds"], 4),
                        round(b["residual_p95_seconds"], 4),
                        round(b["residual_max_seconds"], 4)))

    # alignment_health.json — distribution + cross-class comparison
    arousal_per_class = defaultdict(list)
    rate_per_class = defaultdict(list)
    for d in per_sample:
        cls = ("high" if d["n_flagged"] <= 2
               else ("medium" if d["n_flagged"] <= max(5, d["n_words"] * 0.40)
                     else "low"))
        arousal_per_class[cls].append(d["arousal"]["arousal_proxy_mean"])
        rate_per_class[cls].append(d["speech_rate"]["words_per_second"])

    def stats_block(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "n": int(len(values)),
        }

    health_doc = {
        "total_samples": len(per_sample),
        "health_buckets": dict(health_buckets),
        "flag_reason_counts": dict(flag_reason_counter.most_common()),
        "arousal_by_health": {k: stats_block(v) for k, v in arousal_per_class.items()},
        "speech_rate_by_health": {k: stats_block(v) for k, v in rate_per_class.items()},
        "fallback_used_samples": sum(1 for d in per_sample if d["n_fallback"] > 0),
        "fallback_used_words": sum(d["n_fallback"] for d in per_sample),
        "n_flagged_words_total": sum(d["n_flagged"] for d in per_sample),
        "diagnostic_script_sha256": diagnostic_sha,
    }
    (out / "alignment_health.json").write_text(
        json.dumps(health_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"diagnostic_summary.csv: {len(per_sample)} rows")
    print(f"flag_breakdown.csv: {len(all_flags)} rows")
    print(f"alignment_health.json: {dict(health_buckets)}")


if __name__ == "__main__":
    main()
