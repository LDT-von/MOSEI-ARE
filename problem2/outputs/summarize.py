"""Print the best epoch summary for each arm."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

for arm in ("A", "B", "C"):
    history_path = REPO / "problem2" / "outputs" / f"arm_{arm}_seed_42" / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best = max(history, key=lambda row: row["selection_score"])
    clean = best["clean"]
    mixed = best["mixed"]
    print(
        f"arm {arm}  epoch={best['epoch']:>2}  "
        f"score={best['selection_score']:.4f}  "
        f"clean_acc={clean['accuracy']:.4f}  clean_f1={clean['f1_macro']:.4f}  "
        f"clean_mae={clean['mae']:.4f}  clean_pearson={clean['pearson']:.4f}  "
        f"mixed_acc={mixed['accuracy']:.4f}  mixed_f1={mixed['f1_macro']:.4f}  "
        f"mixed_mae={mixed['mae']:.4f}  mixed_pearson={mixed['pearson']:.4f}"
    )
