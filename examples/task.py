"""Basic example demonstrating pyexp usage."""

import pyexp
from pyexp import Config


@pyexp.task
def exp(config: Config, pretrain=None):
    """Run a single experiment with the given config."""
    lr = config.learning_rate
    epochs = config.epochs
    print(f"Running experiment with lr={lr}, epochs={epochs}")

    # Simulate experiment result
    return {"accuracy": 0.9 + lr * epochs / 100}


@pyexp.flow
def run():
    datasets = ["fabric_hex"]

    for dataset in datasets:

        pretrain = exp(
            {
                "learning_rate": 0.01,
                "epochs": 10,
                "dataset": dataset,
            }
        ).name(f"{dataset}_pretrain")

        exp(
            {
                "learning_rate": 0.01,
                "epochs": 10,
                "dataset": dataset,
                "finetune": True,
            },
            pretrain,
        ).name(f"{dataset}_finetune")


if __name__ == "__main__":
    # Full run:    python examples/task.py
    # Spin:        python examples/task.py --spin fabric_hex_finetune
    run.run()
