"""Generate demo logs for viewer testing using the experiment framework."""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import pyexp
from pyexp import Config, Logger, Runs


@pyexp.experiment(name="demo_logs")
def train(cfg: Config, out: Path):
    """Simulated training run demonstrating logger usage.

    The decorator passes cfg and out when the function has 2 parameters.
    """
    seed = cfg.seed

    # Create logger in output directory
    logger = Logger(out)

    loss = 0.0
    acc = 0.0

    for i in range(100):
        logger.set_global_it(i)

        # Simulate computation time
        time.sleep(0.005)

        # Simulated loss curve (decreasing with noise) - seed affects noise
        loss = 1.0 * math.exp(-i / (30 + seed * 5)) + 0.1 * math.sin(
            i / 5 + seed
        ) * math.exp(-i / 50)
        logger.add_scalar("loss", loss)

        # Simulated accuracy (increasing) - seed affects rate
        acc = 1.0 - (0.9 + seed * 0.05) * math.exp(-i / (25 + seed * 3))
        logger.add_scalar("accuracy", acc)

        # Learning rate schedule
        lr = 0.1 * (0.95 ** (i // 10))
        logger.add_scalar("learning_rate", lr)

        # Log text at certain checkpoints
        if i % 20 == 0:
            logger.add_text("checkpoint", f"Saved checkpoint at iteration {i}")

        # Log figures every 25 iterations
        if i % 25 == 0:
            try:
                import matplotlib

                matplotlib.use("Agg")  # Non-interactive backend for subprocess
                import matplotlib.pyplot as plt
                import numpy as np

                # 2D bar chart - static (faster rendering)
                fig, ax = plt.subplots()
                ax.bar(["loss", "acc"], [loss, acc])
                ax.set_title(f"Metrics at iteration {i}")
                logger.add_figure("metrics_bar", fig, interactive=False)
                plt.close(fig)

                # 3D surface plot (simulated loss landscape) - interactive
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")

                x = np.linspace(-2, 2, 50)
                y = np.linspace(-2, 2, 50)
                X, Y = np.meshgrid(x, y)

                # Loss landscape that changes with iteration (optimum moves toward origin)
                center_x = 1.5 * math.exp(-i / 40) + seed * 0.2
                center_y = 1.5 * math.exp(-i / 40) - seed * 0.2
                Z = (
                    (X - center_x) ** 2
                    + (Y - center_y) ** 2
                    + 0.5 * np.sin(3 * X) * np.cos(3 * Y)
                )

                ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
                ax.scatter([center_x], [center_y], [0], color="red", s=100)
                ax.set_xlabel("Weight 1")
                ax.set_ylabel("Weight 2")
                ax.set_zlabel("Loss")
                ax.set_title(f"Loss Landscape (iteration {i})")

                logger.add_figure("loss_landscape_3d", fig, interactive=True)
                plt.close(fig)

            except ImportError:
                # matplotlib/numpy not installed, skip figures
                pass

    # Flush any pending log writes
    logger.flush()

    return {"final_loss": loss, "final_accuracy": acc}


@train.configs
def configs():
    """Generate configs for two runs with different seeds."""
    return [
        {"name": "run1", "seed": 0.0},
        {"name": "run2", "seed": 1.0},
    ]


if __name__ == "__main__":
    train.run()

    # Print results summary
    from pyexp import LogReader

    results = train.results()
    print("\nResults:")
    for exp in results:
        if exp.error:
            print(f"  {exp.name}: ERROR - {exp.error}")
        else:
            log_reader = LogReader(exp.out)
            it, loss = log_reader["loss"]
            res = exp.result
            print(
                f"  {exp.name}: loss={res['final_loss']:.4f}, accuracy={res['final_accuracy']:.4f}"
            )
