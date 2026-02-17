"""Example demonstrating a marimo HTML report with interactive 3D matplotlib figures.

This example runs a parameter sweep over a simple optimization problem,
then loads results and prints a summary.

Prerequisites:
    pip install numpy

Usage:
    python examples/marimo_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

import pyexp
from pyexp import Config, Runs, sweep


@pyexp.experiment(name="marimo_report")
def experiment(config: Config):
    """Simulate optimizing a 2D loss surface."""
    lr = config.learning_rate
    momentum = config.momentum
    np.random.seed(config.seed)

    def loss_fn(x, y):
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    x, y = config.x0, config.y0
    trajectory = [(x, y, loss_fn(x, y))]

    for i in range(config.steps):
        eps = 1e-5
        dx = (loss_fn(x + eps, y) - loss_fn(x - eps, y)) / (2 * eps)
        dy = (loss_fn(x, y + eps) - loss_fn(x, y - eps)) / (2 * eps)

        x = x - lr * dx + momentum * np.random.normal(0, 0.01)
        y = y - lr * dy + momentum * np.random.normal(0, 0.01)

        if i % 10 == 0:
            trajectory.append((x, y, loss_fn(x, y)))

    final_loss = loss_fn(x, y)
    print(
        f"lr={lr}, momentum={momentum}: final_loss={final_loss:.6f}, pos=({x:.3f}, {y:.3f})"
    )

    return {
        "final_loss": final_loss,
        "final_x": x,
        "final_y": y,
        "trajectory": trajectory,
    }


@experiment.configs
def configs():
    """Sweep learning rates and momentum values."""
    cfgs = [{"name": "opt", "x0": -1.0, "y0": 1.0, "steps": 200, "seed": 42}]
    cfgs = sweep(
        cfgs,
        [
            {"learning_rate": 0.0001, "name": "lr1e-4"},
            {"learning_rate": 0.001, "name": "lr1e-3"},
        ],
    )
    cfgs = sweep(
        cfgs,
        [
            {"momentum": 0.0, "name": "mom0"},
            {"momentum": 0.9, "name": "mom09"},
        ],
    )
    return cfgs


if __name__ == "__main__":
    experiment.run()

    # Print summary from results
    results = experiment.results()
    runs = []
    for exp in results:
        if exp.error:
            continue
        runs.append({
            "name": exp.name,
            "lr": exp.cfg.learning_rate,
            "momentum": exp.cfg.momentum,
            "final_loss": exp.result["final_loss"],
            "final_x": exp.result["final_x"],
            "final_y": exp.result["final_y"],
        })
    runs.sort(key=lambda r: r["final_loss"])

    print("\n=== Optimization Results ===")
    for run in runs:
        print(
            f"  {run['name']}: loss={run['final_loss']:.6f}"
            f" at ({run['final_x']:.3f}, {run['final_y']:.3f})"
        )
    print(f"\nBest: {runs[0]['name']}")
