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
def flow():
    datasets = ["fabric_hex"]

    for dataset in datasets:

        pretrain = exp(
            Config(
                {
                    "learning_rate": 0.01,
                    "epochs": 10,
                    "dataset": dataset,
                }
            ),
        ).name(f"{dataset}_pretrain")

        exp(
            Config(
                {
                    "learning_rate": 0.01,
                    "epochs": 10,
                    "dataset": dataset,
                    "finetune": True,
                }
            ),
            pretrain=pretrain,
        ).name(f"{dataset}_finetune")


if __name__ == "__main__":

    # Returns the last flow result
    last_flow = flow[-1]
    print(f"{last_flow=}")

    results = flow.results()
    task1 = results["fabric_hex_finetune"]
    task1.snapshot
    print(f"{task1=}")

    flow()
