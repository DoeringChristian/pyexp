"""Example demonstrating the decorator API with output directory access."""

from pathlib import Path
import pyexp
from pyexp import Config, Runs


@pyexp.experiment
def exp(cfg: Config, out: Path):
    """Run the experiment with config and output directory access."""
    lr = cfg.learning_rate
    epochs = cfg.epochs

    print(f"Output dir: {out}")
    print(f"Running with lr={lr}, epochs={epochs}")

    return {"accuracy": 0.9 + lr * epochs / 100, "epochs_run": epochs}


@exp.configs
def configs():
    """Return list of configs to run."""
    return [
        {"name": "fast", "learning_rate": 0.01, "epochs": 10},
        {"name": "medium", "learning_rate": 0.001, "epochs": 20},
        {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
    ]


if __name__ == "__main__":
    exp.run()

    # Load results after run
    print("\n=== Results ===")
    print(f"{dir(exp['fast'])=}")
    for run in exp:
        print(
            f"{run.name}: accuracy={run.result['accuracy']:.4f}, epochs={run.result['epochs_run']}"
        )
        print(f"  cfg: {run.cfg}")
        print(f"  out: {run.out}")
        print(f"  error: {run.error}")
