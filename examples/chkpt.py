"""Example demonstrating checkpoint decorator for fault-tolerant experiments."""

from pathlib import Path

from pyexp import Experiment, ExperimentRunner, Tensor, chkpt


class ChkptExperiment(Experiment):
    """Experiment with checkpointed phases."""

    model: dict
    finetuned_model: dict
    results: dict

    @chkpt(retry=3)
    def train(self):
        """Training phase - checkpointed with 3 retries."""
        print(f"Training with lr={self.cfg.learning_rate}")
        for i in range(10):
            # Simulate training
            pass
        self.model = {"weights": [1, 2, 3], "lr": self.cfg.learning_rate}

    @chkpt
    def finetune(self):
        """Finetuning phase - checkpointed."""
        print(f"Finetuning for {self.cfg.epochs} epochs")
        for i in range(10):
            # Simulate finetuning
            pass
        self.finetuned_model = {**self.model, "finetuned": True}

    def experiment(self):
        # Before running train(), a checkpoint is taken.
        # If it fails, it will be re-run up to 3 times.
        self.train()

        # If finetune() fails and we --continue, train() won't re-run
        # because its checkpoint exists.
        self.finetune()

        self.results = {"accuracy": 0.95}

    @staticmethod
    def configs() -> list[dict]:
        return [
            {"name": "fast", "learning_rate": 0.01, "epochs": 10},
            {"name": "medium", "learning_rate": 0.001, "epochs": 20},
            {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
        ]

    @staticmethod
    def report(results: Tensor["ChkptExperiment"], out: Path):
        print("\n=== Experiment Report ===")
        for exp in results:
            print(f"{exp.name}: accuracy={exp.results['accuracy']}")


if __name__ == "__main__":
    runner = ExperimentRunner(ChkptExperiment)
    runner.run()
