"""Generate demo logs for viewer testing."""

import math
import sys

sys.path.insert(0, "src")

from pyexp import Logger


def generate_run(log_dir: str, seed: float = 0.0):
    """Generate a single run with the given seed for variation."""
    logger = Logger(log_dir)

    # Simulate a training run
    for i in range(100):
        logger.set_global_it(i)

        # Simulated loss curve (decreasing with noise) - seed affects noise
        loss = 1.0 * math.exp(-i / (30 + seed * 5)) + 0.1 * math.sin(i / 5 + seed) * math.exp(-i / 50)
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
                import matplotlib.pyplot as plt
                import numpy as np

                # 2D bar chart
                fig, ax = plt.subplots()
                ax.bar(["loss", "acc"], [loss, acc])
                ax.set_title(f"Metrics at iteration {i}")
                logger.add_figure("metrics_bar", fig)
                plt.close(fig)

                # 3D surface plot (simulated loss landscape)
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

                logger.add_figure("loss_landscape_3d", fig)
                plt.close(fig)

            except ImportError:
                # matplotlib/numpy not installed, skip figures
                pass

    logger.flush()


def main():
    # Generate two runs with different seeds
    print("Generating run1...")
    generate_run("out/demo_logs/run1", seed=0.0)

    print("Generating run2...")
    generate_run("out/demo_logs/run2", seed=1.0)

    print("Demo logs written to out/demo_logs/")
    print("Run viewer with: uv run --extra viewer python -m pyexp.viewer out/demo_logs")


if __name__ == "__main__":
    main()
