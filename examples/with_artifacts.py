"""Example demonstrating artifact saving with pyexp."""

import json
import pyexp
from pyexp import Config


@pyexp.experiment
def experiment(config: Config):
    """Run experiment and save artifacts to out."""
    print(f"Running: {config.name}")

    # Access nested config with dot notation
    lr = config.optimizer.learning_rate
    batch_size = config.optimizer.batch_size

    # Simulate training
    result = {
        "accuracy": 0.85 + lr * 10,
        "loss": 0.5 - lr * 5,
    }

    # Save artifacts to the experiment's output directory
    with open(config.out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


@experiment.configs
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


@experiment.report
def report(results, report_dir):
    """Print summary report."""
    print("\n=== Results ===")
    for r in results:
        print(f"{r.name}: acc={r.result['accuracy']:.3f}, loss={r.result['loss']:.3f}")


if __name__ == "__main__":
    experiment.run()
