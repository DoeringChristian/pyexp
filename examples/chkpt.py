import pyexp
from pyexp import Config


@pyexp.chkpt(retry=3)
def pass1():

    for i in range(10):
        # do some training
        ...


@pyexp.chkpt
def pass2():

    for i in range(10):
        # do some finetuning
        ...


@pyexp.experiment
def experiment(config: Config):

    # Before running pass1, a checkpoint should be taken.
    # If it failed, it should be re-run up to 3 times.
    pass1()

    # If pass2 failed, we should be able to continue here instead of having to
    # re-run pass1.
    pass2()

    return {}


@experiment.configs
def configs() -> list[dict]:
    """Generate experiment configurations."""
    return [
        {"name": "fast", "learning_rate": 0.01, "epochs": 10},
        {"name": "medium", "learning_rate": 0.001, "epochs": 20},
        {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
    ]


@experiment.report
def report(results, report_dir): ...


if __name__ == "__main__":
    experiment.run()
