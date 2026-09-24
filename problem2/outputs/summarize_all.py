"""Summary of validated test metrics, grid and attachment-3 predictions."""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

print("=" * 80)
print("Test split (clean + mixed) after early stopping")
print("=" * 80)
for arm in ("A", "B", "C"):
    test_path = REPO / "problem2" / "outputs" / f"arm_{arm}_seed_42" / "test_evaluation.json"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    clean, mixed = test["clean"], test["mixed"]
    print(f"arm {arm}")
    print(f"  clean   acc={clean['accuracy']:.4f}  f1={clean['f1_macro']:.4f}  "
          f"mae={clean['mae']:.4f}  pearson={clean['pearson']:.4f}")
    print(f"  mixed   acc={mixed['accuracy']:.4f}  f1={mixed['f1_macro']:.4f}  "
          f"mae={mixed['mae']:.4f}  pearson={mixed['pearson']:.4f}")

print()
print("=" * 80)
print("Grid matrices (validation, only perturbed subset when n_with_artificial_gap changes)")
print("=" * 80)
for arm in ("A", "B", "C"):
    valid_path = REPO / "problem2" / "outputs" / f"arm_{arm}_seed_42" / "valid_evaluation.json"
    valid = json.loads(valid_path.read_text(encoding="utf-8"))
    print(f"\narm {arm} (mean over 27 grid conditions)")
    grid_summary = {"text": [], "audio": [], "vision": []}
    for key, value in valid.items():
        if key in {"provenance", "clean", "mixed"}:
            continue
        modality = key.split("_")[0]
        grid_summary[modality].append(value["f1_macro"])
    for modality, scores in grid_summary.items():
        mean = sum(scores) / len(scores) if scores else 0.0
        print(f"  {modality:>6s}  mean_f1={mean:.4f}  n_conditions={len(scores)}")

print()
print("=" * 80)
print("Attachment 3 predictions")
print("=" * 80)
for arm in ("A", "B", "C"):
    csv_path = REPO / "problem2" / "outputs" / f"arm_{arm}_seed_42" / "attachment3_predictions.csv"
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    by_class = {0: 0, 1: 0, 2: 0}
    intensities = []
    for row in rows:
        by_class[int(row["polarity"])] += 1
        intensities.append(float(row["intensity"]))
    avg_intensity = sum(intensities) / len(intensities)
    print(f"arm {arm}  {len(rows)} samples  "
          f"neg={by_class[0]}  neu={by_class[1]}  pos={by_class[2]}  "
          f"mean_intensity={avg_intensity:+.4f}")
