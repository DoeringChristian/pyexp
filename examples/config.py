from pyexp import sweep
from examples.basic import experiment


@experiment.configs
def configs() -> list[dict]:
    """Generate experiment configurations."""
    return [
        {"name": "fast", "learning_rate": 0.1, "epochs": 100},
        {"name": "medium", "learning_rate": 0.01, "epochs": 200},
        {"name": "slow", "learning_rate": 0.001, "epochs": 500},
    ]


if __name__ == "__main__":
    experiment.run()
