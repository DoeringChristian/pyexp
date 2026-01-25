"""Basic example demonstrating pyexp usage."""

import pyexp
from pyexp import Config, Tensor, sweep


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
    cfgs = [
        {
            "name": "exp",
            "learning_rate": 0.01,
            "mlp": {
                "type": "MLP",
                "width": 32,
                "n_hidden": 2,
                "encoding": {
                    "type": "Sin",
                    "octaves": 4,
                },
            },
        }
    ]
    # Single sweep of learning_rate. Generates an array of configs.
    cfgs = sweep(
        cfgs,
        [
            {"learning_rate": 0.1, "name": "lr0.1"},
            {"learning_rate": 0.01, "name": "lr0.2"},
        ],
    )
    # Sweep combination of multiple parameters.
    cfgs = sweep(
        cfgs,
        [
            {
                "name": "epochs10",
                "epochs": 10,
                "batch_size": 32,
                "mlp.width": 32,  # Dot notation to access individual elements
                "mlp.encoding": None,
            },
            {
                "name": "epochs20",
                "epochs": 20,
                "batch_size": 16,
                "mlp.width": 64,
                "mlp.encoding": {
                    # Overwrites the dictionary i.e. this one should not have "octaves"
                    "type": "Tri",
                    "n_funcs": 4,
                },
            },
        ],
    )
    # Should be 4 configs now (2 x 2).
    # Names are "exp_lr0.1_epochs10", "exp_lr0.1_epochs20", etc.

    # Pattern matching examples:
    # cfgs["exp_lr0.1_*"]  -> shape (1, 1, 2), matches both epochs variants
    # cfgs["*_epochs10"]   -> shape (1, 2, 1), matches both lr variants
    # cfgs["exp_lr0.1_epochs10"]  -> single config (exact match)

    return cfgs


@experiment.report
def report(results: Tensor):
    """Generate report from all experiment results.

    Each result contains:
    - 'name': the combined config name
    - 'config': the full config dict
    - experiment outputs (e.g., 'accuracy')
    """
    print("\n=== Experiment Report ===")
    for r in results:
        print(f"Config: {r['name']} -> Accuracy: {r['accuracy']:.4f}")

    # Filter results by config values
    lr01_results = results[{"config.learning_rate": 0.1}]
    print(f"\nResults with lr=0.1: {lr01_results.shape}")

    # Find best result
    best_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])
    print(
        f"\nBest: {results[best_idx]['name']} with accuracy {results[best_idx]['accuracy']:.4f}"
    )


if __name__ == "__main__":
    experiment.run()
