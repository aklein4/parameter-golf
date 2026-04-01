#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
STEP_RE = re.compile(r"step:(\d+)/\d+.*train_loss:([0-9]*\.?[0-9]+)")


def find_log(run: str) -> Path:
    path = Path(run)
    if path.is_file():
        return path

    matches = ROOT.glob(f"logs/{run}*")
    matches = sorted(matches, key=lambda x: len(str(x)))
    if len(matches) == 0:
        raise SystemExit(f"expected 1 match for {run!r}, found {len(matches)}")
    return matches[0]


def read_losses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps = []
    losses = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    if not steps:
        raise SystemExit(f"no train_loss lines found in {path}")
    return np.arange(len(losses))+1, np.array(losses)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output", type=Path, default="loss_plot.png")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--ylim", nargs="+", type=float)
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))
    for run in args.runs:
        path = find_log(run)
        steps, losses = read_losses(path)
        if args.max_steps is not None:
            mask = steps <= args.max_steps
            steps = steps[mask]
            losses = losses[mask]
        (line,) = plt.plot(steps, losses, alpha=0.25, linewidth=1)
        plt.plot(steps, rolling_mean(losses, args.window), color=line.get_color(), linewidth=3, label=run)

    plt.xlabel("step")
    plt.ylabel("train loss")
    if args.ylim is not None:
        plt.ylim(args.ylim)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
