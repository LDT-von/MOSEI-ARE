"""One-shot reproduction for the contest submission.

Run with `MOSEI_DATA_ROOT` pointing at the directory that contains the four
附件 folders. The script prints the commands it would run without actually
training (training must be triggered manually because it takes ~10 minutes per
arm). Evaluation, evidence scan and Attachment-3/4 inference can be triggered
directly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent


COMMANDS = {
    "p1-feature": "python problem1/exp3_problem1.py",
    "p2-train-A": "python -m problem2.scripts.run train --arm A --seed 42 --device cuda --output problem2/outputs/arm_A_seed_42",
    "p2-train-B": "python -m problem2.scripts.run train --arm B --seed 42 --device cuda --output problem2/outputs/arm_B_seed_42",
    "p2-train-C": "python -m problem2.scripts.run train --arm C --seed 42 --device cuda --output problem2/outputs/arm_C_seed_42",
    "p2-eval": "python -m problem2.scripts.run evaluate --checkpoint problem2/outputs/arm_C_seed_42/best.pt --split valid --grid --device cuda",
    "p2-attachment3": "python -m problem2.scripts.run infer --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda",
    "p3-eval-valid": "python -m problem3.scripts.run evaluate-valid --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda",
    "p3-eval-test": "python -m problem3.scripts.run evaluate-test  --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda",
    "p3-evidence": "python -m problem3.scripts.run evidence-valid --checkpoint problem2/outputs/arm_C_seed_42/best.pt --sample-size 32 --device cuda",
    "p3-attachment4": "python -m problem3.scripts.run infer --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda --window-length 6 --window-top-k 3",
    "p3-summary": "python problem3/scripts/summarize.py",
    "p3-tests": "python -m unittest problem3.test_pipeline -v",
}


def _env() -> dict[str, str]:
    environment = dict(os.environ)
    data_root = environment.get("MOSEI_DATA_ROOT") or environment.get("DATA_ROOT")
    if data_root:
        environment["MOSEI_DATA_ROOT"] = data_root
    return environment


def _run(name: str) -> None:
    if name not in COMMANDS:
        raise SystemExit(f"unknown command: {name}")
    print(f"\n>>> {name}\n>>> {COMMANDS[name]}")
    subprocess.run(COMMANDS[name].split(), cwd=REPO, env=_env(), check=False)


def _print_all() -> None:
    print("Available commands:")
    for name, command in COMMANDS.items():
        print(f"  {name:>16s} : {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot reproduction driver.")
    parser.add_argument("command", nargs="?", help="command name from COMMANDS; omit to print the list")
    parser.add_argument("--list", action="store_true", help="print the available commands")
    args = parser.parse_args()
    if args.list or not args.command:
        _print_all()
        return
    _run(args.command)


if __name__ == "__main__":
    main()
