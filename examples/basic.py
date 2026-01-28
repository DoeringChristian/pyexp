"""Basic example demonstrating pyexp usage."""

import pyexp
from pyexp import Config


@pyexp.experiment
def experiment(config: Config):
    """Run a single experiment with the given config."""
    lr = config.learning_rate
    epochs = config.epochs
    print(f"Running experiment with lr={lr}, epochs={epochs}")
    # Simulate experiment result
    return {"accuracy": 0.9 + lr * epochs / 100}


@experiment.configs
def configs() -> list[dict]:
    """Generate experiment configurations."""
    return [
        {"name": "fast", "learning_rate": 0.01, "epochs": 10},
        {"name": "medium", "learning_rate": 0.001, "epochs": 20},
        {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
    ]


@experiment.report
def report(results, report_dir):
    """Generate report from all experiment results.

    Each result has: .name, .config, .result, .error, .log
    """
    print("\n=== Experiment Report ===")
    for r in results:
        print(f"Config: {r.name} -> Accuracy: {r.result['accuracy']:.4f}")
    best = max(results, key=lambda r: r.result["accuracy"])
    print(f"\nBest: {best.name} with accuracy {best.result['accuracy']:.4f}")


if __name__ == "__main__":
    experiment.run()

    # Load latest results
    results = experiment.results()
    print(f"\nLoaded {len(list(results))} results from latest run")

    # Load results from a specific timestamp
    # results = experiment.results(timestamp="2026-01-28_10-30-00")
