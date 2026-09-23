"""Train, evaluate and infer the aligned_50 local-gap sentiment model.

Run from the repository root with ``python -m problem2.scripts.run --help``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.gap_slot_emo import GapSlotEmo
from problem2.data import AlignedDataset, fit_normalization, load_attachment2, load_attachment3, normalize
from problem2.metrics import sentiment_metrics


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "problem2" / "configs" / "default.json"
DEFAULT_DATA = Path(os.environ.get("MOSEI_DATA_ROOT", REPO.parent / "E题数据" / "E题数据"))
ATTACHMENT2 = "附件2-数据集特征文件/aligned_50.pkl"
ATTACHMENT3 = "附件3-模态缺失特征样本/对齐版本"
MODEL_KEYS = ("tokens", "segments", "audio", "vision", "support", "observed")


def _json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(dataset: AlignedDataset, batch_size: int, *, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def _input(batch: dict, device: torch.device) -> dict:
    return {key: batch[key].to(device, non_blocking=True) for key in MODEL_KEYS}


def _predict(model: GapSlotEmo, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray | list[str]]:
    model.eval()
    all_ids, preds, strengths, labels, truth, counts = [], [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            result = model(**_input(batch, device))
            logits = result["logits"]
            regression = result["regression"]
            if not torch.isfinite(logits).all() or not torch.isfinite(regression).all():
                raise FloatingPointError("nonfinite prediction")
            all_ids.extend(batch["id"])
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            strengths.extend(regression.cpu().tolist())
            counts.extend(batch["observed"].sum(dim=-1).tolist())
            if "label" in batch:
                labels.extend(batch["label"].tolist())
                truth.extend(batch["intensity"].tolist())
    return {
        "ids": all_ids, "predictions": np.asarray(preds, dtype=np.int64),
        "intensity": np.asarray(strengths, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "truth": np.asarray(truth, dtype=np.float32),
        "observed_counts": counts,
    }


def _metrics(result: dict) -> dict:
    return sentiment_metrics(result["labels"], result["predictions"], result["truth"], result["intensity"])


def _make_model(config: dict, arm: str) -> GapSlotEmo:
    return GapSlotEmo(**config["model"], use_gap_repair=(arm == "C"))


def _device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def train(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    arm = args.arm
    seed = args.seed if args.seed is not None else int(config["seed"])
    _seed(seed)
    device = _device(args.device)
    data_path = args.data_root / ATTACHMENT2
    splits = load_attachment2(data_path)
    stats = fit_normalization(splits["train"])
    for split in splits.values():
        normalize(split, stats)
    train_mode = "clean" if arm == "A" else "mixed"
    train_data = AlignedDataset(splits["train"], mode=train_mode, seed=seed, probability=config["train_missing_probability"])
    valid_clean = AlignedDataset(splits["valid"], mode="clean")
    valid_mixed = AlignedDataset(splits["valid"], mode="mixed", seed=config["validation_seed"], probability=1.0)
    model = _make_model(config, arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.HuberLoss(delta=1.0)
    out = args.output or REPO / "problem2" / "outputs" / f"arm_{arm}_seed_{seed}"
    if (out / "run.json").exists() or (out / "best.pt").exists():
        raise FileExistsError(f"run directory already contains results: {out}; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "arm": arm, "seed": seed, "data_release": "aligned_50.pkl",
        "attachment2_sha256": _sha256(data_path), "device": str(device),
        "torch": str(torch.__version__), "numpy": np.__version__, "config": config,
        "model_source_sha256": _sha256(REPO / "models" / "gap_slot_emo.py"),
        "data_source_sha256": _sha256(REPO / "problem2" / "data.py"),
        "runner_source_sha256": _sha256(Path(__file__)),
        "train_n": len(splits["train"]), "valid_n": len(splits["valid"]),
    }
    _json(out / "run.json", provenance)
    _json(out / "normalization.json", stats)
    best_score, stale, history = -float("inf"), 0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_data.set_epoch(epoch)
        total_loss = 0.0
        total_n = 0
        for batch in _loader(train_data, config["batch_size"], shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            result = model(**_input(batch, device))
            label = batch["label"].to(device)
            intensity = batch["intensity"].to(device)
            loss = criterion_cls(result["logits"], label) + config["regression_weight"] * criterion_reg(result["regression"], intensity)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total_loss += loss.item() * len(label)
            total_n += len(label)
        clean = _metrics(_predict(model, _loader(valid_clean, config["batch_size"]), device))
        mixed = _metrics(_predict(model, _loader(valid_mixed, config["batch_size"]), device))
        # Fixed before training: both clean and local-gap validation matter.
        score = (clean["f1_macro"] + mixed["f1_macro"]) / 2 - 0.1 * (clean["mae"] + mixed["mae"]) / 2
        row = {"epoch": epoch, "train_loss": total_loss / total_n, "selection_score": score, "clean": clean, "mixed": mixed}
        history.append(row)
        _json(out / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score + config["min_delta"]:
            best_score, stale = score, 0
            torch.save({"model": model.state_dict(), "arm": arm, "config": config,
                        "normalization": stats, "provenance": provenance, "epoch": epoch,
                        "validation": {"clean": clean, "mixed": mixed, "selection_score": score}}, out / "best.pt")
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    checkpoint = out / "best.pt"
    print(f"best checkpoint: {checkpoint}, size={checkpoint.stat().st_size / 1e6:.2f} MB")
    if checkpoint.stat().st_size > 50_000_000:
        raise RuntimeError("checkpoint alone exceeds the 50 MB contest attachment limit")


def _checkpoint(path: Path, device: torch.device) -> tuple[GapSlotEmo, dict]:
    state = torch.load(path, map_location=device, weights_only=True)
    expected = state.get("provenance", {}).get("model_source_sha256")
    if expected and expected != _sha256(REPO / "models" / "gap_slot_emo.py"):
        raise RuntimeError("model source differs from the saved checkpoint; use the recorded model version")
    model = _make_model(state["config"], state["arm"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def evaluate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    model, state = _checkpoint(args.checkpoint, device)
    splits = load_attachment2(args.data_root / ATTACHMENT2)
    split = normalize(splits[args.split], state["normalization"])
    batch_size = state["config"]["batch_size"]
    scenarios = [("clean", AlignedDataset(split, mode="clean")),
                 ("mixed", AlignedDataset(split, mode="mixed", seed=state["config"]["validation_seed"], probability=1.0))]
    if args.grid:
        for modality in range(3):
            for location in ("start", "middle", "end"):
                for fraction in (0.15, 0.30, 0.45):
                    name = f"{('text','audio','vision')[modality]}_{location}_{fraction:.2f}"
                    scenarios.append((name, AlignedDataset(split, mode="fixed", fixed={
                        "modality": modality, "location": location, "fraction": fraction,
                    })))
    report = {name: _metrics(_predict(model, _loader(dataset, batch_size), device)) for name, dataset in scenarios}
    report["provenance"] = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256(args.checkpoint),
                            "split": args.split, "feature_version": "aligned_50"}
    output = args.output or args.checkpoint.parent / f"{args.split}_evaluation.json"
    _json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def infer(args: argparse.Namespace) -> None:
    device = _device(args.device)
    model, state = _checkpoint(args.checkpoint, device)
    samples = normalize(load_attachment3(args.data_root / ATTACHMENT3), state["normalization"])
    result = _predict(model, _loader(AlignedDataset(samples, mode="clean"), state["config"]["batch_size"]), device)
    output = args.output or args.checkpoint.parent / "attachment3_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "polarity", "intensity", "text_observed", "audio_observed", "vision_observed"))
        for sample_id, polarity, intensity, counts in zip(result["ids"], result["predictions"], result["intensity"], result["observed_counts"]):
            writer.writerow((sample_id, int(polarity), f"{float(intensity):.6f}", *counts))
    _json(output.with_suffix(".provenance.json"), {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256(args.checkpoint),
        "attachment3_files": len(samples), "feature_version": "aligned_50",
        "training_seed": state["provenance"]["seed"],
        "model_source_sha256": state["provenance"]["model_source_sha256"],
        "polarity_map": {"0": "negative", "1": "neutral", "2": "positive"},
        "note": "Attachment 3 has no labels; no Accuracy/F1/MAE/Pearson can be computed.",
    })
    print(f"wrote {len(samples)} predictions to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--output", type=Path)
    train_parser.set_defaults(func=train)
    for name, function in (("evaluate", evaluate), ("infer", infer)):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--output", type=Path)
        if name == "evaluate":
            sub.add_argument("--split", choices=("valid", "test"), default="valid")
            sub.add_argument("--grid", action="store_true", help="27 modality/location/length conditions")
        sub.set_defaults(func=function)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
