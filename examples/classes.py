"""Example demonstrating the decorator API with report."""

from pathlib import Path
import pyexp
from pyexp import Config, Runs


@pyexp.experiment
def my_experiment(cfg: Config, out: Path):
    """Run the experiment with config and output directory access."""
    lr = cfg.learning_rate
    epochs = cfg.epochs

    print(f"Output dir: {out}")
    print(f"Running with lr={lr}, epochs={epochs}")

    return {"accuracy": 0.9 + lr * epochs / 100, "epochs_run": epochs}


@my_experiment.configs
def configs():
    """Return list of configs to run."""
    return [
        {"name": "fast", "learning_rate": 0.01, "epochs": 10},
        {"name": "medium", "learning_rate": 0.001, "epochs": 20},
        {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
    ]


@my_experiment.report
def report(results: Runs, out: Path):
    """Generate report from results."""
    print("\n=== Experiment Report ===")
    for exp in results:
        print(f"{exp.name}: accuracy={exp.result['accuracy']:.4f}, epochs={exp.result['epochs_run']}")
        print(f"  cfg: {exp.cfg}")
        print(f"  out: {exp.out}")
        print(f"  error: {exp.error}")

    best = max(results, key=lambda exp: exp.result["accuracy"])
    print(f"\nBest: {best.name} with accuracy {best.result['accuracy']:.4f}")


if __name__ == "__main__":
    my_experiment.run()
