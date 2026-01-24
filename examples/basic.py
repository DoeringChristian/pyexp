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
def report(results):
    """Generate report from all experiment results.

    Each result contains 'name', 'config', and experiment outputs.
    """
    print("\n=== Experiment Report ===")
    for r in results:
        print(f"Config: {r['name']} -> Accuracy: {r['accuracy']:.4f}")
    best_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])
    print(f"\nBest: {results[best_idx]['name']} with accuracy {results[best_idx]['accuracy']:.4f}")


if __name__ == "__main__":
    experiment.run()
