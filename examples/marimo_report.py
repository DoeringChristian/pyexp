"""Example demonstrating a marimo HTML report with interactive 3D matplotlib figures.

This example runs a parameter sweep over a simple optimization problem,
then generates a self-contained HTML report using marimo's island generator.
The report contains interactive 3D loss landscape visualizations.

Prerequisites:
    pip install marimo matplotlib numpy

Usage:
    # Run the experiment (generates report.html)
    python examples/marimo_report.py

    # Open the report in a browser
    open <output_dir>/report/report.html
"""

import asyncio
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, "src")

import numpy as np

import pyexp
from pyexp import Config, Runs, sweep


@pyexp.experiment(name="marimo_report")
def experiment(config: Config):
    """Simulate optimizing a 2D loss surface.

    Each run uses a different learning rate and optimization trajectory,
    resulting in different final positions on the loss landscape.
    """
    lr = config.learning_rate
    momentum = config.momentum
    np.random.seed(config.seed)

    # Rosenbrock-like loss function
    def loss_fn(x, y):
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    # Simulate gradient descent trajectory
    x, y = config.x0, config.y0
    trajectory = [(x, y, loss_fn(x, y))]

    for i in range(config.steps):
        # Numerical gradient
        eps = 1e-5
        dx = (loss_fn(x + eps, y) - loss_fn(x - eps, y)) / (2 * eps)
        dy = (loss_fn(x, y + eps) - loss_fn(x, y - eps)) / (2 * eps)

        # Gradient descent with momentum (simplified)
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


@experiment.report
def report(results: Runs, report_dir: Path):
    """Generate an HTML report with interactive 3D figures using marimo."""
    import marimo

    # Collect data from results
    runs = []
    for exp in results:
        if exp.error:
            continue
        runs.append(
            {
                "name": exp.name,
                "lr": exp.cfg.learning_rate,
                "momentum": exp.cfg.momentum,
                "final_loss": exp.result["final_loss"],
                "final_x": exp.result["final_x"],
                "final_y": exp.result["final_y"],
                "trajectory": exp.result["trajectory"],
            }
        )
    runs.sort(key=lambda r: r["final_loss"])

    # Print summary to console
    print("\n=== Optimization Results ===")
    for run in runs:
        print(
            f"  {run['name']}: loss={run['final_loss']:.6f}"
            f" at ({run['final_x']:.3f}, {run['final_y']:.3f})"
        )
    print(f"\nBest: {runs[0]['name']}")

    # Build a marimo app and run it
    app = marimo.App(width="medium")

    @app.cell
    def header(mo):
        return (
            mo.md("""
            # Optimization Report

            Parameter sweep over learning rate and momentum for gradient descent
            on a Rosenbrock loss surface. The **optimal point** is at (1, 1)
            with loss = 0.

            Interact with the 3D plots below: **click and drag to rotate**,
            **scroll to zoom**.
            """),
        )

    @app.cell
    def data():
        import numpy as np_

        return (np_,)

    @app.cell
    def summary_table(mo, runs_):
        return (
            mo.ui.table(
                [
                    {
                        "Name": _r["name"],
                        "Learning Rate": _r["lr"],
                        "Momentum": _r["momentum"],
                        "Final Loss": f'{_r["final_loss"]:.6f}',
                        "Final Position": f'({_r["final_x"]:.3f}, {_r["final_y"]:.3f})',
                    }
                    for _r in runs_
                ],
                selection=None,
            ),
        )

    @app.cell
    def landscape(np_, runs_):
        import matplotlib.pyplot as _plt

        _fig = _plt.figure(figsize=(10, 8))
        _ax = _fig.add_subplot(111, projection="3d")

        _x = np_.linspace(-2, 2, 80)
        _y = np_.linspace(-1, 3, 80)
        _X, _Y = np_.meshgrid(_x, _y)
        _Z = np_.log1p((1 - _X) ** 2 + 100 * (_Y - _X**2) ** 2)

        _ax.plot_surface(_X, _Y, _Z, cmap="viridis", alpha=0.4, edgecolor="none")

        _colors = ["red", "blue", "orange", "green", "purple", "cyan"]
        for _i, _r in enumerate(runs_):
            _traj = _r["trajectory"]
            _tx = [p[0] for p in _traj]
            _ty = [p[1] for p in _traj]
            _tz = [np_.log1p(p[2]) for p in _traj]
            _c = _colors[_i % len(_colors)]
            _ax.plot(_tx, _ty, _tz, color=_c, linewidth=2, label=_r["name"])
            _ax.scatter([_tx[-1]], [_ty[-1]], [_tz[-1]], color=_c, s=80, zorder=5)

        _ax.scatter(
            [1],
            [1],
            [0],
            color="gold",
            s=200,
            marker="*",
            zorder=10,
            label="Optimum (1,1)",
        )
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        _ax.set_zlabel("log(1 + loss)")
        _ax.set_title("Loss Landscape - All Trajectories")
        _ax.legend(loc="upper left", fontsize=8)
        return (_fig,)

    @app.cell
    def convergence(np_, runs_):
        import matplotlib.pyplot as _plt

        _fig, _ax = _plt.subplots(figsize=(10, 5))
        for _r in runs_:
            _traj = _r["trajectory"]
            _losses = [p[2] for p in _traj]
            _steps = list(range(0, len(_losses) * 10, 10))
            _ax.semilogy(_steps, _losses, linewidth=2, label=_r["name"])

        _ax.set_xlabel("Step")
        _ax.set_ylabel("Loss (log scale)")
        _ax.set_title("Convergence Comparison")
        _ax.legend()
        _ax.grid(True, alpha=0.3)
        return (_fig,)

    app.run(defs={"runs_": runs})


if __name__ == "__main__":
    experiment.run()
