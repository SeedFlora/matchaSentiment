from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Matcha sentiment training workflow.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-transformers", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    if not args.skip_prepare:
        run_step([python, "scripts/prepare_data.py"])
    if not args.skip_classical:
        run_step([python, "scripts/train_classical.py", "--folds", "10"])
    if not args.skip_transformers:
        command = [
            python,
            "scripts/train_transformers.py",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.require_gpu:
            command.append("--require-gpu")
        run_step(command)


if __name__ == "__main__":
    main()
