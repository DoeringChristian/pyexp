"""Example demonstrating inter-run dependencies (pipeline)."""

import pyexp
from pyexp import Config


@pyexp.experiment
def experiment(config: Config, deps=None): ...


def config_pretrain(lr=0.01, epochs=100):
    # returns the hash of config and run, used in the filename as well. This should also include additional arguments.
    return experiment(
        {
            "lr": lr,
            "epochs": epochs,
        }
    )


def config_finetune(lr=0.01, epochs=100):
    # Deduplicated. Returns the same hash as above.
    pretrain = experiment(
        {
            "lr": lr,
            "epochs": epochs,
        }
    )

    return experiment(
        {
            "lr": lr,
            "epochs": epochs,
        },
        pretrain,
    )


@experiment.configs
def configs() -> list[dict]:
    """Pipeline: pretrain -> finetune -> evaluate."""

    pretrain01 = config_pretrain(0.01, 100)
    pretrain02 = config_pretrain(0.001, 10)
    config_finetune(0.01, 10)


if __name__ == "__main__":
    experiment.run()
