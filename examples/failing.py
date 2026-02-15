from os import abort
import pyexp
from pyexp import Config


def dosomething():
    print("do something")


def somethingdifferent():
    print("do something different")


@pyexp.experiment
def exp(config: Config):
    """Run a single experiment with the given config."""
    lr = config.learning_rate
    epochs = config.epochs
    print(f"Running experiment with lr={lr}, epochs={epochs}")

    exp.chkpt(dosomething)()

    # Should also be run the first time this script is run i.e. don't cache the function
    exp.chkpt(dosomething)()

    exp.chkpt(somethingdifferent)()

    abort()
    exit()

    # Simulate experiment result
    return {"accuracy": 0.9 + lr * epochs / 100}


@exp.configs
def configs() -> list[dict]:
    """Generate experiment configurations."""
    return [
        {"name": "fast", "learning_rate": 0.01, "epochs": 10},
        {"name": "medium", "learning_rate": 0.001, "epochs": 20},
        {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
        {"name": "slow_low", "learning_rate": 0.0001, "epochs": 20},
    ]


if __name__ == "__main__":
    results = exp.results()
    print(f"{results=}")

    exp.run()

    # Load latest results
    results = exp.results()
    print(f"\nLoaded {len(list(results))} results from latest run")
    print(f"{results[0].cfg=}")

    result = results[{"cfg.epochs": 20}]
    print(f"{result=}")

    # Access output directory via exp.out
    print(f"Output dir: {results[0].out}")

    # Load results from a specific timestamp
    # results = experiment.results(timestamp="2026-01-28_10-30-00")
