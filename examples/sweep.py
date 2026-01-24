"""Basic example demonstrating pyexp usage."""

import pyexp
from pyexp import Config, sweep


@pyexp.experiment
def experiment(config: Config):
    """Run a single experiment with the given config."""
    lr = config.learning_rate
    epochs = config.epochs
    batch_size = config.batch_size
    print(f"Running: lr={lr}, epochs={epochs}, batch_size={batch_size}")
    # Simulate experiment result
    return {"accuracy": 0.9 + lr * epochs / batch_size}


@experiment.configs
def configs() -> list[dict]:
    """Generate experiment configurations."""
    cfgs = [{"name": "exp"}]
    # Single sweep of learning_rate. Generates an array of configs.
    cfgs = sweep(
        cfgs,
        [
            {"learning_rate": 0.1},
            {"learning_rate": 0.01},
        ],
    )
    # Sweep combination of multiple parameters.
    cfgs = sweep(
        cfgs,
        [
            {"epochs": 10, "batch_size": 32},
            {"epochs": 20, "batch_size": 16},
        ],
    )
    # Should be 4 configs now (2 x 2).
    return cfgs


@experiment.report
def report(configs: list[dict], results: list):
    """Generate report from all experiment results."""
    print("\n=== Experiment Report ===")
    for config, result in zip(configs, results):
        print(f"Config: {config['name']} -> Accuracy: {result['accuracy']:.4f}")
    best_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])
    print(
        f"\nBest: {configs[best_idx]['name']} with accuracy {results[best_idx]['accuracy']:.4f}"
    )


if __name__ == "__main__":
    experiment.run()
