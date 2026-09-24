"""Training, evidence evaluation and Attachment 4 inference for Problem 3.

Problem 3 reuses the GapSlotEmo checkpoint from Problem 2; the same `aligned_50`
input interface guarantees that frozen weights transfer to Attachment 4 without
re-training. The scripts compute per-sample modality contributions and the
top-k contiguous evidence windows that drop the predicted-class logit the
most when ablated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from problem2.data import AlignedDataset, load_attachment2, normalize
from problem2.scripts.run import _checkpoint, _device, _loader, _predict, REPO
from problem3.data import (
    attachment4_to_split,
    grid_seconds,
    load_attachment4,
)
from problem3.explain import (
    collect_batch_evidence,
    write_attachment4_csv,
)


ATTACHMENT2 = "附件2-数据集特征文件/aligned_50.pkl"


def _resolve_attachment4_aligned(data_root: Path) -> Path:
    """Locate the aligned-version pkl folder under Attachment 4 without hardcoding its name."""
    candidates = []
    for child in data_root.iterdir():
        if "附件4" in child.name:
            candidates.append(child)
    if not candidates:
        raise ValueError(f"no 附件4 directory under {data_root}")
    first = candidates[0]
    aligned = first / "对齐版本"
    if aligned.is_dir() and list(aligned.glob("*.pkl")):
        return aligned
    nested = first / first.name / "对齐版本"
    if nested.is_dir() and list(nested.glob("*.pkl")):
        return nested
    raise ValueError(f"could not locate aligned-version pkl folder under {first}")


def _json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for cls in (0, 1, 2):
        tp = int(((labels == cls) & (predictions == cls)).sum())
        fp = int(((labels != cls) & (predictions == cls)).sum())
        fn = int(((labels == cls) & (predictions != cls)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _evaluate_split(args: argparse.Namespace, split_name: str) -> None:
    """Compute predictions and metrics on a split; no expensive evidence scan."""
    device = _device(args.device)
    model, state = _checkpoint(args.checkpoint, device)
    splits = load_attachment2(args.data_root / ATTACHMENT2)
    split = normalize(splits[split_name], state["normalization"])
    predictions = _predict(
        model, _loader(AlignedDataset(split, mode="clean"), state["config"]["batch_size"]), device
    )
    labels = np.asarray(predictions["labels"])
    preds = np.asarray(predictions["predictions"])
    accuracy_summary = {
        "n": len(preds),
        "accuracy": float((preds == labels).mean()),
        "f1_macro": _macro_f1(labels, preds),
    }
    print(f"evaluation summary: {json.dumps(accuracy_summary)}", flush=True)
    output = args.output or args.checkpoint.parent / f"problem3_{split_name}_evaluation.json"
    _json(output, {"summary": accuracy_summary, "split": split_name})
    print(f"wrote {output}", flush=True)


def _evidence_split(args: argparse.Namespace, split_name: str) -> None:
    """Evidence scan on a manageable subset of a split."""
    device = _device(args.device)
    model, state = _checkpoint(args.checkpoint, device)
    splits = load_attachment2(args.data_root / ATTACHMENT2)
    split = normalize(splits[split_name], state["normalization"])
    dataset = AlignedDataset(split, mode="clean")
    indices = list(range(min(args.sample_size, len(dataset))))
    records = collect_batch_evidence(
        model, dataset, indices, device,
        window_length=args.window_length, top_k=args.window_top_k,
    )
    output = args.output or args.checkpoint.parent / f"problem3_{split_name}_evidence.json"
    _json(output, {"split": split_name, "records": records})
    print(f"wrote {len(records)} evidence records to {output}", flush=True)


def evaluate_valid(args: argparse.Namespace) -> None:
    _evaluate_split(args, "valid")


def evaluate_test(args: argparse.Namespace) -> None:
    _evaluate_split(args, "test")


def evidence_valid(args: argparse.Namespace) -> None:
    _evidence_split(args, "valid")


def evidence_test(args: argparse.Namespace) -> None:
    _evidence_split(args, "test")


def infer(args: argparse.Namespace) -> None:
    """Predict on Attachment 4 samples and emit a 20-row CSV + evidence JSON."""
    device = _device(args.device)
    model, state = _checkpoint(args.checkpoint, device)
    samples = load_attachment4(_resolve_attachment4_aligned(args.data_root))
    split = normalize(attachment4_to_split(samples), state["normalization"])
    dataset = AlignedDataset(split, mode="clean")
    predictions = _predict(
        model, _loader(dataset, state["config"]["batch_size"]), device
    )
    records = collect_batch_evidence(
        model, dataset, list(range(len(dataset))), device,
        window_length=args.window_length, top_k=args.window_top_k,
    )
    rows: list[dict] = []
    for sample, record, polarity, intensity in zip(samples, records, predictions["predictions"], predictions["intensity"]):
        record["prediction"] = {"polarity": int(polarity), "intensity": float(intensity)}
        record["raw_text"] = sample.raw_text
        record["video_filename"] = f"{sample.sample_id}.mp4"
        rows.append(record)
    output_dir = args.output or args.checkpoint.parent / "problem3_attachment4"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attachment4_predictions.csv"
    write_attachment4_csv(csv_path, rows)
    _json(output_dir / "attachment4_evidence.json", rows)
    _json(output_dir / "attachment4_provenance.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "attachment4_files": len(samples),
        "video_directory": str((_resolve_attachment4_aligned(args.data_root) / "videos").resolve()),
        "grid_seconds": grid_seconds().tolist(),
        "window_length_positions": args.window_length,
        "window_length_seconds": round(args.window_length * 0.2, 3),
        "feature_version": "aligned_50",
        "training_seed": state["provenance"]["seed"],
        "run_kind": state["provenance"].get("run_kind", "training"),
        "model_source_sha256": state["provenance"]["model_source_sha256"],
        "polarity_map": {"0": "negative", "1": "neutral", "2": "positive"},
        "note": "Attachment 4 has no labels; the dominant modality and top windows are prediction dependencies, not causal attributions.",
    })
    print(f"wrote {len(rows)} attachment-4 predictions to {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, default=REPO.parent / "E题数据" / "E题数据")
    common.add_argument("--device", default="cuda")
    common.add_argument("--window-length", type=int, default=6)
    common.add_argument("--window-top-k", type=int, default=3)
    common.add_argument("--sample-size", type=int, default=16)
    common.add_argument("--output", type=Path)
    sub_specs = (
        ("evaluate-valid", evaluate_valid),
        ("evaluate-test", evaluate_test),
        ("evidence-valid", evidence_valid),
        ("evidence-test", evidence_test),
        ("infer", infer),
    )
    for command, function in sub_specs:
        sub_parser = sub.add_parser(command, parents=[common])
        sub_parser.add_argument("--checkpoint", type=Path, required=True)
        sub_parser.set_defaults(func=function)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
