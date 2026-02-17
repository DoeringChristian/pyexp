"""Example demonstrating artifact saving with pyexp.

For experiments that need to save artifacts to the output directory,
use a function with the (cfg, out) signature.
"""

import json
from pathlib import Path
import pyexp
from pyexp import Config, Runs


@pyexp.experiment
def artifact_experiment(cfg: Config, out: Path):
    """Run experiment and save artifacts to out."""
    print(f"Running: {cfg.name}")

    # Access nested config with dot notation
    lr = cfg.optimizer.learning_rate
    batch_size = cfg.optimizer.batch_size

    # Simulate training
    accuracy = 0.85 + lr * 10
    loss = 0.5 - lr * 5

    # Save artifacts to the experiment's output directory
    with open(out / "metrics.json", "w") as f:
        json.dump({"accuracy": accuracy, "loss": loss}, f, indent=2)

    return {"accuracy": accuracy, "loss": loss}


@artifact_experiment.configs
def configs() -> list[dict]:
    """Generate configs with nested structure."""
    return [
        {
            "name": "small-lr",
            "optimizer": {"learning_rate": 0.001, "batch_size": 32},
        },
        {
            "name": "large-lr",
            "optimizer": {"learning_rate": 0.01, "batch_size": 64},
        },
    ]


@artifact_experiment.report
def report(results: Runs, report_dir: Path):
    """Print summary report."""
    print("\n=== Results ===")
    for exp in results:
        print(f"{exp.name}: acc={exp.result['accuracy']:.3f}, loss={exp.result['loss']:.3f}")

        # Load saved artifacts
        metrics_path = exp.out / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                saved = json.load(f)
            print(f"  Saved metrics: {saved}")


if __name__ == "__main__":
    artifact_experiment.run()
