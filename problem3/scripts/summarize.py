"""Print the Problem 3 evidence summary."""
import json
from pathlib import Path

REPO = Path("C:/Users/栋栋/Desktop/E题/MOSEI-ARE")

# 1) Validation / Test evaluation summaries
print("=" * 80)
print("Validation / Test metrics (clean forward pass, full split)")
print("=" * 80)
for split in ("valid", "test"):
    path = REPO / "problem2" / "outputs" / "arm_C_seed_42" / f"problem3_{split}_evaluation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    print(f"{split:>5s}  n={summary['n']:>3d}  accuracy={summary['accuracy']:.4f}  f1_macro={summary['f1_macro']:.4f}")

# 2) Validation evidence subset
print()
print("=" * 80)
print("Validation evidence subset (32 samples)")
print("=" * 80)
path = REPO / "problem2" / "outputs" / "arm_C_seed_42" / "problem3_valid_evidence.json"
data = json.loads(path.read_text(encoding="utf-8"))
records = data["records"]
import collections
dominant = collections.Counter(r["modality_contribution"]["dominant_modality"] for r in records)
print(f"records={len(records)}  dominant: {dict(dominant)}")
top_window_lengths = {m: [] for m in ("text", "audio", "vision")}
for record in records:
    for modality, payload in record["window_evidence"].items():
        for window in payload["windows"]:
            top_window_lengths[modality].append((window["start_seconds"], window["end_seconds"], window["logit_drop"]))
for modality, wins in top_window_lengths.items():
    if wins:
        avg_drop = sum(w[2] for w in wins) / len(wins)
        print(f"  {modality:>6s}: {len(wins)} windows, avg logit-drop {avg_drop:+.4f}")

# 3) Attachment 4 inference
print()
print("=" * 80)
print("Attachment 4 inference (20 samples, no labels)")
print("=" * 80)
csv_path = REPO / "problem2" / "outputs" / "arm_C_seed_42" / "problem3_attachment4" / "attachment4_predictions.csv"
import csv
rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
polarity_counts = collections.Counter(int(row["polarity"]) for row in rows)
intensities = [float(row["intensity"]) for row in rows]
dominant4 = collections.Counter(row["dominant_modality"] for row in rows)
print(f"polarity: neg={polarity_counts[0]}  neu={polarity_counts[1]}  pos={polarity_counts[2]}")
print(f"intensity range: [{min(intensities):+.3f}, {max(intensities):+.3f}], mean {sum(intensities)/len(intensities):+.4f}")
print(f"dominant_modality: {dict(dominant4)}")
