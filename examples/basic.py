"""Basic example demonstrating pyexp usage."""

import pyexp
from pyexp import Config


@pyexp.experiment
def exp(config: Config):
    """Run a single experiment with the given config."""
    lr = config.learning_rate
    epochs = config.epochs
    print(f"Running experiment with lr={lr}, epochs={epochs}")

    # Simulate experiment result
    return {"accuracy": 0.9 + lr * epochs / 100}


@exp.configs
def configs() -> list[dict]:
    """Generate experiment configurations."""
    return [
        {"name": "fast", "learning_rate": 0.01, "epochs": 10},
        {"name": "medium", "learning_rate": 0.001, "epochs": 20},
        {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
    ]


if __name__ == "__main__":
    exp.run()

    # Load latest results
    results = exp.results()
    print(f"\nLoaded {len(list(results))} results from latest run")

    # Access output directory via exp.out
    print(f"Output dir: {results[0].out}")
