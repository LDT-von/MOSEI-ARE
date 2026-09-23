"""Problem 1: timestamp-preserving feature extraction from the 100 supplied videos.

No sentiment labels are used to fit, choose or tune any extractor.
Run from any directory: python pipeline.py --workspace <E-question directory>.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torchaudio
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.models import ResNet18_Weights, resnet18

SR = 16000
GRID_SECONDS = 0.2
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_REVISION = "86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
AUDIO_NAMES = ([f"mfcc_{k}" for k in range(13)] +
               [f"delta_mfcc_{k}" for k in range(13)] +
               ["log_rms", "zero_crossing_rate", "spectral_centroid_hz",
                "rolloff85_hz", "f0_hz", "periodicity", "voiced"])


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number_words(n):
    ones = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
    for value, name in [(1000000, "million"), (1000, "thousand"), (100, "hundred")]:
        if n >= value:
            return number_words(n // value) + " " + name + (" " + number_words(n % value) if n % value else "")
    raise ValueError(n)


def transcript_units(text):
    # Preserve source text and character offsets; normalization is a separate field.
    normalized = text.replace("’", "'")
    units = []
    for match in re.finditer(r"\d[\d,]*(?:st|nd|rd|th)?|[A-Za-z]+(?:'[A-Za-z]+)*", normalized):
        word = match.group()
        if word[0].isdigit():
            n = int(re.sub(r"\D", "", word))
            if word.lower() == "10th":
                spoken = "tenth"
            elif 1900 <= n < 2000:
                spoken = "nineteen " + number_words(n % 100)
            else:
                spoken = number_words(n)
        else:
            spoken = word
        units.append({"word": text[match.start():match.end()],
                      "char_start": match.start(), "char_end": match.end(),
                      "normalized": spoken.upper()})
    if not units:
        raise ValueError("No alignable transcript units; preserve sample and review manually")
    return units


def media_metadata(path):
    with av.open(str(path)) as c:
        origin = float(c.start_time or 0) / av.time_base
        duration = float(c.duration) / av.time_base
        streams = []
        for s in c.streams:
            streams.append({"type": s.type, "time_base": str(s.time_base),
                            "start_time_seconds": float(s.start_time or 0) * float(s.time_base),
                            "duration_seconds": float(s.duration or 0) * float(s.time_base),
                            "header_frame_count": s.frames,
                            "codec": s.codec_context.name})
    return {"duration": duration, "origin_seconds": origin, "streams": streams}


def decode_audio(path, metadata):
    # Resampled samples are placed by presentation timestamps, not by concatenation.
    parts = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SR)
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise ValueError("No audio stream")
        for frame in c.decode(audio=0):
            for out in resampler.resample(frame):
                parts.append((float(out.time) - metadata["origin_seconds"], out.to_ndarray().reshape(-1)))
        for out in resampler.resample(None):
            parts.append((float(out.time) - metadata["origin_seconds"], out.to_ndarray().reshape(-1)))
    n = int(math.ceil(metadata["duration"] * SR))
    signal, available = np.zeros(n, np.float32), np.zeros(n, bool)
    for t, samples in parts:
        start = int(round(t * SR))
        stop = min(n, start + len(samples))
        if stop > max(0, start):
            signal[max(start, 0):stop] = samples[max(-start, 0):stop - start]
            available[max(start, 0):stop] = True
    if not available.any():
        raise ValueError("Audio decoded to no valid samples")
    return signal, available


def acoustic_features(signal, available):
    # 40-ms Hann windows, 20-ms hop. Partial last frames are padded but recorded.
    win, hop = 640, 320
    starts = np.arange(0, len(signal), hop)
    padded = np.pad(signal, (0, win))
    frames = np.stack([padded[i:i + win] for i in starts])
    end_indices = np.minimum(starts + win, len(signal))
    fractions = np.array([available[a:b].mean() for a, b in zip(starts, end_indices)], np.float32)
    x = torch.from_numpy(frames)
    spectrum = torch.fft.rfft(x * torch.hann_window(win), n=1024).abs().square()
    banks = torchaudio.functional.melscale_fbanks(513, 20, 7600, 40, SR, norm="slaney", mel_scale="slaney")
    log_mel = (spectrum @ banks).clamp_min(1e-10).log()
    mfcc = log_mel @ torchaudio.functional.create_dct(13, 40, "ortho")
    delta = torchaudio.functional.compute_deltas(mfcc.T.unsqueeze(0), win_length=5)[0].T
    rms = (x.square().mean(1) + 1e-12).sqrt()
    zcr = ((x[:, 1:] * x[:, :-1]) < 0).float().mean(1)
    freq = torch.linspace(0, SR / 2, spectrum.shape[1])
    centroid = (spectrum * freq).sum(1) / spectrum.sum(1).clamp_min(1e-12)
    rolloff = (spectrum.cumsum(1) >= .85 * spectrum.sum(1, keepdim=True)).float().argmax(1) * (SR / 1024)
    centered = x - x.mean(1, keepdim=True)
    acf = torch.fft.irfft(torch.fft.rfft(centered, n=2048).abs().square(), n=2048)[:, :win]
    lo, hi = SR // 400, SR // 50
    lag = acf[:, lo:hi + 1].argmax(1) + lo
    periodicity = acf.gather(1, lag[:, None])[:, 0] / acf[:, 0].clamp_min(1e-12)
    voiced = (periodicity > .3) & (rms > 1e-4)
    pitch = torch.where(voiced, SR / lag.float(), 0.)
    extra = torch.stack([rms.log(), zcr, centroid, rolloff.float(), pitch,
                         periodicity, voiced.float()], 1)
    feat = torch.cat([mfcc, delta, extra], 1).numpy().astype(np.float32)
    intervals = np.stack([starts / SR, end_indices / SR], 1).astype(np.float32)
    valid = fractions > .99
    feat[~valid] = 0
    return feat, intervals, valid, fractions


def overlap_pool(features, intervals, target, valid=None):
    """Interval overlap is the sole cross-modal alignment rule; no extrapolation."""
    overlap = np.maximum(0., np.minimum(target[:, None, 1], intervals[None, :, 1]) -
                         np.maximum(target[:, None, 0], intervals[None, :, 0]))
    if valid is not None:
        overlap *= np.asarray(valid)[None, :]
    den = overlap.sum(1)
    weights = np.divide(overlap, den[:, None], out=np.zeros_like(overlap), where=den[:, None] > 0)
    return (weights @ features).astype(np.float32), den > 1e-8, weights.astype(np.float32)


class Extractors:
    def __init__(self, assets, device):
        self.device = device
        torch.hub.set_dir(str(assets))
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.ctc = self.bundle.get_model(dl_kwargs={"model_dir": str(assets), "progress": False}).to(device).eval()
        self.labels = self.bundle.get_labels()
        self.vocabulary = {c: i for i, c in enumerate(self.labels)}
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION, local_files_only=True)
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION, local_files_only=True).to(device).eval()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.vision = resnet18(weights=weights).to(device).eval()
        self.vision.fc = torch.nn.Identity()
        self.vision_transform = weights.transforms()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    @torch.inference_mode()
    def align_text(self, signal, available, text):
        units = transcript_units(text)
        transcript = "|".join(u["normalized"].replace(" ", "|") for u in units)
        targets = torch.tensor([[self.vocabulary[c] for c in transcript]], dtype=torch.int32)
        audio_end = np.flatnonzero(available)[-1] + 1
        emission, _ = self.ctc(torch.from_numpy(signal[:audio_end]).to(self.device)[None])
        emission = emission.log_softmax(-1).cpu()
        path, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        spans = torchaudio.functional.merge_tokens(path[0], scores[0].exp(), blank=0)
        if len(spans) != len(transcript):
            raise ValueError("CTC path does not match every target character")
        # Wav2Vec2 receptive field 400 samples, output stride 320 samples.
        # Use output-center +/- half stride; keep the small boundary gaps explicit.
        cursor = 0
        for i, unit in enumerate(units):
            length = len(unit["normalized"])
            selected = spans[cursor:cursor + length]
            char_only = [s for s in selected if self.labels[s.token] != "|"]
            start = (selected[0].start * 320 + 40) / SR
            end = (selected[-1].end * 320 + 40) / SR
            confidence = float(np.mean([float(s.score) for s in char_only]))
            unit.update(start=max(0., start), end=min(end, audio_end / SR),
                        ctc_score=confidence, review_required=confidence < .5)
            cursor += length + 1
        greedy = emission[0].argmax(-1).tolist()
        decoded = "".join(self.labels[c] for j, c in enumerate(greedy) if c != 0 and (j == 0 or c != greedy[j - 1])).replace("|", " ").strip()
        return units, decoded

    @torch.inference_mode()
    def text_features(self, text, units):
        # Sliding windows prevent silent truncation at the pretrained 128-token limit.
        encoded = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        sums = torch.zeros((len(ids), 384), dtype=torch.float32)
        counts = torch.zeros(len(ids))
        for start in range(0, len(ids), 96):
            segment = ids[start:start + 126]
            tokens = self.tokenizer.prepare_for_model(segment, return_tensors="pt", add_special_tokens=True)
            inp = {k: v.unsqueeze(0).to(self.device) for k, v in tokens.items()}
            out = self.text_model(**inp).last_hidden_state[0, 1:1 + len(segment)].cpu()
            sums[start:start + len(segment)] += out
            counts[start:start + len(segment)] += 1
            if start + 126 >= len(ids):
                break
        contextual = sums / counts.clamp_min(1)[:, None]
        result = []
        for unit in units:
            members = [i for i, (a, b) in enumerate(offsets) if min(b, unit["char_end"]) > max(a, unit["char_start"])]
            if not members:
                raise ValueError("Source word has no tokenizer correspondence")
            unit["subtoken_indices"] = members
            result.append(contextual[members].mean(0).numpy())
        return np.stack(result).astype(np.float32), offsets

    @torch.inference_mode()
    def visual_features(self, path, metadata, grid):
        targets = grid.mean(1)
        selected, timestamps, indices, pending = [], [], [], 0
        # Decode PTS in order; choose the nearest decoded frame to each grid center.
        previous = None
        decoded_count = 0
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                decoded_count += 1
                current = (float(frame.time) - metadata["origin_seconds"], index, frame)
                while pending < len(targets) and current[0] >= targets[pending]:
                    best = min([x for x in (previous, current) if x is not None], key=lambda x: abs(x[0] - targets[pending]))
                    selected.append(best[2].to_ndarray(format="rgb24"))
                    timestamps.append(best[0]); indices.append(best[1]); pending += 1
                previous = current
            while pending < len(targets) and previous is not None:
                selected.append(previous[2].to_ndarray(format="rgb24"))
                timestamps.append(previous[0]); indices.append(previous[1]); pending += 1
        if len(selected) != len(grid):
            raise ValueError("Video did not provide a frame for every bin")
        inputs, boxes, face_present, candidates = [], [], [], []
        for rgb in selected:
            h, w = rgb.shape[:2]
            scale = min(1., 480 / w)
            gray = cv2.cvtColor(cv2.resize(rgb, (round(w * scale), round(h * scale))), cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            candidates.append(len(faces))
            if len(faces):
                x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3]) / scale
                pad = .15 * max(bw, bh)
                x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
                x1, y1 = min(w, int(x + bw + pad)), min(h, int(y + bh + pad))
                crop = rgb[y0:y1, x0:x1]
                box = [x0, y0, x1, y1]
            else:
                crop, box = rgb, [0, 0, w, h]
            inputs.append(self.vision_transform(Image.fromarray(crop)))
            boxes.append(box); face_present.append(bool(len(faces)))
        values = []
        for start in range(0, len(inputs), 32):
            values.append(self.vision(torch.stack(inputs[start:start + 32]).to(self.device)).cpu().numpy())
        features = np.concatenate(values).astype(np.float32)
        return features, {"pts_seconds": timestamps, "frame_indices": indices,
                          "decoded_frame_count": decoded_count,
                          "crop_xyxy": boxes, "face_detected": face_present,
                          "face_candidates": candidates, "frame_sizes_hw": [list(f.shape[:2]) for f in selected]}, selected


def make_grid(duration):
    edges = np.minimum(np.arange(math.ceil(duration / GRID_SECONDS) + 1) * GRID_SECONDS, duration)
    return np.stack([edges[:-1], edges[1:]], 1).astype(np.float32)


def write_example(out_dir, sample_id, text, signal, grid, units, frames, frame_info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    example = out_dir / "example"
    example.mkdir(exist_ok=True)
    focus_end = min(float(grid[-1, 1]), 6.)
    chosen = [u for u in units if u["start"] < focus_end]
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [1.3, 1, 1.8]})
    sample_step = 80
    times = np.arange(0, len(signal), sample_step) / SR
    axes[0].plot(times, signal[::sample_step], linewidth=.6, color="#206080")
    axes[0].set_xlim(0, focus_end); axes[0].set_ylabel("Audio amplitude")
    for i, unit in enumerate(chosen):
        axes[1].broken_barh([(unit["start"], unit["end"] - unit["start"])], (i % 2, .75), facecolors="#b6d9e2")
        axes[1].text(unit["start"], i % 2 + .1, unit["word"], fontsize=8, rotation=25)
    axes[1].set_xlim(0, focus_end); axes[1].set_ylim(-.1, 2.3); axes[1].set_yticks([]); axes[1].set_xlabel("Clip time (seconds)")
    axes[2].axis("off")
    picks = np.linspace(0, min(len(frames) - 1, int(focus_end / GRID_SECONDS) - 1), 5).astype(int)
    for j, index in enumerate(picks):
        ax = axes[2].inset_axes([j / 5, .05, .195, .85])
        ax.imshow(frames[index]); ax.axis("off")
        ax.set_title(f"frame {frame_info['frame_indices'][index]} | {frame_info['pts_seconds'][index]:.3f}s", fontsize=8)
    fig.suptitle(f"{sample_id}: transcript, audio and decoded video timestamps", fontsize=12)
    fig.tight_layout(); fig.savefig(example / "alignment.png", dpi=180); plt.close(fig)
    records = []
    for i, (a, b) in enumerate(grid):
        words = [u["word"] for u in units if min(float(b), u["end"]) > max(float(a), u["start"])]
        records.append({"bin": i, "start": float(a), "end": float(b), "text": " ".join(words),
                        "video_pts": frame_info["pts_seconds"][i], "frame_index": frame_info["frame_indices"][i]})
    save_json(example / "correspondence.json", {"sample_id": sample_id, "source_text": text, "rows": records})


def run(args):
    root = args.workspace.resolve()
    base = Path(__file__).resolve().parent
    out = base / "outputs"
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)
    data = json.loads((base / "input_manifest.json").read_text(encoding="utf-8"))
    torch.manual_seed(20260923); torch.set_num_threads(4); cv2.setNumThreads(2)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = Extractors(base / "assets", device)
    summary = []
    rows = data["rows"][:args.limit] if args.limit else data["rows"]
    code_hash = sha256(__file__)
    for row in rows:
        sid = row["sample_id"]
        target = out / "features" / (sid + ".npz")
        meta_path = out / "metadata" / (sid + ".json")
        if args.resume and target.exists() and meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if (previous.get("pipeline_sha256") == code_hash and
                previous.get("source_text") == row["text"] and
                previous["summary"]["source_sha256"] == sha256(root / row["source_relpath"]) and
                previous["summary"]["feature_sha256"] == sha256(target)):
                summary.append(previous["summary"])
                print("RESUME", sid, flush=True); continue
        started = time.time()
        source = root / row["source_relpath"]
        try:
            media = media_metadata(source)
            grid = make_grid(media["duration"])
            signal, audio_available = decode_audio(source, media)
            audio, audio_intervals, audio_valid, audio_fraction = acoustic_features(signal, audio_available)
            units, greedy = models.align_text(signal, audio_available, row["text"])
            text, offsets = models.text_features(row["text"], units)
            word_intervals = np.array([[u["start"], u["end"]] for u in units], np.float32)
            aligned_text, text_mask, text_weights = overlap_pool(text, word_intervals, grid)
            aligned_audio, audio_mask, audio_weights = overlap_pool(audio, audio_intervals, grid, audio_valid)
            raw_vision, vinfo, frames = models.visual_features(source, media, grid)
            visual_mask = np.abs(np.array(vinfo["pts_seconds"]) - grid.mean(1)) <= GRID_SECONDS / 2 + 1e-3
            vision = raw_vision.copy()
            vision[~visual_mask] = 0
            scores = np.array([u["ctc_score"] for u in units], np.float32)
            score_grid = text_weights @ scores
            face_detected = np.array(vinfo["face_detected"], bool)
            features = {
                "grid_intervals": grid, "text": aligned_text.astype(np.float16),
                "audio": aligned_audio.astype(np.float32), "vision": vision.astype(np.float16),
                "text_mask": text_mask, "audio_mask": audio_mask, "vision_mask": visual_mask,
                "padding_mask": np.ones(len(grid), bool), "text_alignment_score": score_grid,
                "vision_face_detected": face_detected, "vision_scene_fallback": ~face_detected,
                "raw_text": text.astype(np.float16), "raw_text_intervals": word_intervals,
                "raw_text_scores": scores, "raw_audio": audio, "raw_audio_intervals": audio_intervals,
                "raw_audio_valid": audio_valid, "raw_audio_available_fraction": audio_fraction,
                "raw_vision": raw_vision.astype(np.float16), "raw_vision_pts": np.array(vinfo["pts_seconds"], np.float32),
                "raw_vision_frame_indices": np.array(vinfo["frame_indices"], np.int32),
            }
            assert all(np.isfinite(v).all() for v in features.values())
            np.savez_compressed(target, **features)
            summary_row = {
                "sample_id": sid, "video_id": row["video_id"], "clip_id": row["clip_id"],
                "duration_seconds": round(media["duration"], 6), "grid_seconds": GRID_SECONDS,
                "aligned_length": len(grid), "text_dim": 384, "audio_dim": 33, "vision_dim": 512,
                "word_count": len(units), "raw_audio_length": len(audio), "raw_vision_length": len(vision),
                "audio_available_ratio": round(float(audio_available.mean()), 6),
                "text_valid_bins": int(text_mask.sum()), "audio_valid_bins": int(audio_mask.sum()),
                "vision_valid_bins": int(visual_mask.sum()), "face_detected_bins": int(face_detected.sum()),
                "ctc_mean_score": round(float(scores.mean()), 6),
                "review_word_count": sum(u["review_required"] for u in units),
                "original_label": row["label"], "original_annotation": row["annotation"],
                "source_sha256": sha256(source), "feature_sha256": sha256(target),
                "feature_file": "features/" + target.name, "metadata_file": "metadata/" + meta_path.name,
                "status": "extracted",
            }
            metadata = {"summary": summary_row, "source_relpath": row["source_relpath"],
                        "pipeline_sha256": code_hash,
                        "source_text": row["text"], "media": media, "words": units,
                        "ctc_greedy_transcript_for_review": greedy, "token_offsets": offsets,
                        "video": vinfo, "runtime_seconds": round(time.time() - started, 3),
                        "alignment_rule": "interval_overlap", "padding_rule": "none_per_sample; right_zero_padding_only_when_batched",
                        "normalization_rule": "extractor_defined_only; no dataset_fitted_scaler",
                        "text_float16_max_abs_error": float(np.max(np.abs(text - text.astype(np.float16).astype(np.float32)))),
                        "vision_float16_max_abs_error": float(np.max(np.abs(raw_vision - raw_vision.astype(np.float16).astype(np.float32)))),
                        "visual_feature_meaning": "ResNet18 appearance of largest detected face; whole-frame context when no face found. Not emotion probabilities or facial action units."}
            save_json(meta_path, metadata)
            if row is rows[0]:
                write_example(out, sid, row["text"], signal, grid, units, frames, vinfo)
            summary.append(summary_row)
            print(f"OK {len(summary)}/{len(rows)} {sid} {media['duration']:.3f}s words={len(units)} ctc={scores.mean():.3f} face={face_detected.mean():.2f} runtime={time.time()-started:.1f}s", flush=True)
        except Exception as exc:
            with (out / "errors.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"sample_id": sid, "error": repr(exc)}, ensure_ascii=False) + "\n")
            raise
    with (out / "summary_100.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    versions = {p: importlib.metadata.version(p) for p in ["torch", "torchaudio", "torchvision", "transformers", "av", "numpy", "opencv-python", "Pillow"]}
    save_json(out / "run_manifest.json", {
        "sample_count": len(summary), "full_expected_count": 100, "device": device,
        "versions": versions, "ffmpeg_libraries": av.library_versions,
        "text_model": TEXT_MODEL, "text_revision": TEXT_REVISION,
        "ctc_model": "WAV2VEC2_ASR_BASE_960H", "vision_model": "ResNet18_IMAGENET1K_V1",
        "grid_seconds": GRID_SECONDS, "audio_sample_rate": SR,
        "audio_window_seconds": .04, "audio_hop_seconds": .02, "audio_names": AUDIO_NAMES,
        "text_window_tokens": 126, "text_stride_tokens": 96, "ctc_review_threshold": .5,
        "ctc_review_threshold_meaning": "predefined diagnostic only, not calibrated correctness probability",
        "source_label_sha256": data["source_label_sha256"], "pipeline_sha256": sha256(__file__),
        "pretrained_weight_sha256": {p.name: sha256(p) for p in (base / "assets").rglob("*.pth")},
        "no_sentiment_training": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
