"""Render the headline figures used in problem2/3 reports.

All inputs are JSON artefacts already produced by the training/evaluation
pipeline, so this script is deterministic and CPU-only.  Output PNGs land in
problem2/outputs/_figures/ (created on demand) and are referenced from the
README + the two markdown reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "problem2" / "outputs" / "_figures"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = {
    "A (text-only)":   ("arm_A_seed_42", "Arm A"),
    "B (gap-aware)":   ("arm_B_seed_42", "Arm B"),
    "C (compensating)":("arm_C_seed_42", "Arm C*"),  # Arm C uses the best-of-2024 weights
}
SEED_COLOURS = {1: "#1f77b4", 7: "#2ca02c", 42: "#ff7f0e",
                100: "#9467bd", 2024: "#d62728"}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_test(folder: str) -> dict:
    return json.load(open(REPO / "problem2" / "outputs" / folder / "test_evaluation.json"))


def _load_valid(folder: str) -> dict:
    return json.load(open(REPO / "problem2" / "outputs" / folder / "valid_evaluation.json"))


def _load_history(folder: str) -> list[dict]:
    return json.load(open(REPO / "problem2" / "outputs" / folder / "history.json"))


# ---------------------------------------------------------------------------
# Figure 1: Arm A/B/C side-by-side test metrics.
# ---------------------------------------------------------------------------
def fig_arm_comparison() -> Path:
    metrics = ["accuracy", "f1_macro"]
    conditions = ["clean", "mixed"]
    width = 0.35
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    palette = ["#4c72b0", "#55a868", "#c44e52"]
    for ax, cond in zip(axes, conditions):
        for i, (label, (folder, _)) in enumerate(ARMS.items()):
            t = _load_test(folder)
            vals = [t[cond][m] for m in metrics]
            ax.bar(x + (i - 1) * width, vals, width, label=label,
                   color=palette[i], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(["accuracy", "f1_macro"])
        ax.set_title(f"test / {cond}")
        ax.set_ylim(0.30, 0.60)
        ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=0.8,
                   label="random baseline" if cond == "clean" else None)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("score")
    fig.suptitle("Test metrics per arm (clean vs 99 % missing-mixed)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "fig1_arm_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Arm C training curves (selection_score + loss) for the four
# candidate seeds.  Best epoch is highlighted with a star.
# ---------------------------------------------------------------------------
def fig_seed_training() -> Path:
    seeds = [1, 7, 42, 100, 2024]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for seed in seeds:
        if seed == 42:
            hist = _load_history("arm_C_seed_42_BACKUP_seed42")
            label = "seed=42 (baseline)"
        else:
            hist = _load_history(f"_scan_C_seed_{seed}")
            label = f"seed={seed}"
        epochs = [h["epoch"] for h in hist]
        scores = [h["selection_score"] for h in hist]
        losses = [h["train_loss"] for h in hist]
        best_idx = int(np.argmax(scores))
        axes[0].plot(epochs, scores, "-o", color=SEED_COLOURS[seed],
                     label=label, markersize=4)
        axes[0].plot(epochs[best_idx], scores[best_idx], "*",
                     color=SEED_COLOURS[seed], markersize=11)
        axes[1].plot(epochs, losses, "-o", color=SEED_COLOURS[seed],
                     label=label, markersize=4)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation selection_score")
    axes[0].set_title("validation score per epoch")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("training loss")
    axes[1].set_title("training loss per epoch")
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Arm C · four-seed training curves (★ = best epoch)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "fig2_arm_c_seed_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: test_clean_f1 vs test_clean_acc on a 2-D scatter for the five
# candidates.  Makes the seed=2024 trade-off obvious.
# ---------------------------------------------------------------------------
def fig_seed_scatter() -> Path:
    seeds = [1, 7, 42, 100, 2024]
    f1, acc, scores = [], [], []
    for seed in seeds:
        if seed == 42:
            t = _load_test("arm_C_seed_42_BACKUP_seed42")
            h = _load_history("arm_C_seed_42_BACKUP_seed42")
        else:
            t = _load_test(f"_scan_C_seed_{seed}")
            h = _load_history(f"_scan_C_seed_{seed}")
        f1.append(t["clean"]["f1_macro"])
        acc.append(t["clean"]["accuracy"])
        scores.append(max(h, key=lambda x: x["selection_score"])["selection_score"])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for s, x, y, sc in zip(seeds, acc, f1, scores):
        ax.scatter(x, y, s=140, color=SEED_COLOURS[s],
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(f"seed={s}\nsel={sc:.3f}",
                    (x, y), xytext=(8, 6), textcoords="offset points",
                    fontsize=8.5)
    ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=0.8, label="random F1")
    ax.set_xlabel("test clean accuracy")
    ax.set_ylabel("test clean F1-macro")
    ax.set_title("Arm C · test clean trade-off across 5 seeds")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig3_arm_c_seed_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: Arm C valid 27-cell grid heatmap (text / audio / vision
# starting offset).  Only filled cells whose key starts with one of the
# three modality prefixes.
# ---------------------------------------------------------------------------
def fig_robustness_grid() -> Path:
    v = _load_valid("arm_C_seed_42")
    grid_keys = [k for k in v if k.startswith(("text_", "audio_", "vision_"))]
    # group by modality & (position, width)
    by_mod = {"text": [], "audio": [], "vision": []}
    for k in grid_keys:
        parts = k.split("_")           # e.g. text_start_0.15
        mod = parts[0]
        pos = parts[1]                 # start / middle / end
        width = parts[2]               # 0.15 / 0.30 / 0.45
        by_mod.setdefault(mod, []).append((pos, float(width), v[k]["f1_macro"]))
    positions = ("start", "middle", "end")
    widths = sorted({w for mod in by_mod for _, w, _ in by_mod[mod]})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5),
                             sharey=True, sharex=True)
    titles = {"text": "text gap",
              "audio": "audio gap",
              "vision": "vision gap"}
    for ax, mod in zip(axes, ("text", "audio", "vision")):
        matrix = np.full((len(positions), len(widths)), np.nan)
        for pos, w, f1 in by_mod[mod]:
            i = positions.index(pos)
            j = widths.index(w)
            matrix[i, j] = f1
        im = ax.imshow(matrix, cmap="viridis", aspect="auto",
                       vmin=0.30, vmax=0.55)
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([f"{w:g}s" for w in widths])
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions)
        ax.set_title(titles[mod])
    axes[0].set_ylabel("gap position")
    fig.colorbar(im, ax=axes, label="F1-macro", shrink=0.8)
    fig.suptitle("Arm C · 27-cell robustness grid (valid set)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / "fig4_arm_c_robustness_grid.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: Attachment-3 and Attachment-4 prediction summary.
# ---------------------------------------------------------------------------
def fig_attachment_summary() -> Path:
    import csv
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, csv_path, title in [
        (axes[0], "problem2/outputs/arm_C_seed_42/attachment3_predictions.csv",
         "Attachment 3 (30 samples)"),
        (axes[1], "problem2/outputs/arm_C_seed_42/problem3_attachment4/attachment4_predictions.csv",
         "Attachment 4 (20 samples)"),
    ]:
        rows = list(csv.DictReader(open(REPO / csv_path)))
        pols = [int(r["polarity"]) for r in rows]
        ints = [float(r["intensity"]) for r in rows]
        bins = np.arange(-0.5, 3.5, 1)
        ax.hist(pols, bins=bins, color="#4c72b0", rwidth=0.85)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["neg", "neu", "pos"])
        ax.set_xlabel("predicted polarity")
        ax.set_ylabel("count")
        ax.set_title(f"{title}\nintensity [{min(ints):+.2f}, {max(ints):+.2f}] "
                     f"mean={np.mean(ints):+.2f}")
    fig.tight_layout()
    path = OUT / "fig5_attachment_predictions.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    out = []
    out.append(fig_arm_comparison())
    out.append(fig_seed_training())
    out.append(fig_seed_scatter())
    out.append(fig_robustness_grid())
    out.append(fig_attachment_summary())
    print("wrote:")
    for p in out:
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
