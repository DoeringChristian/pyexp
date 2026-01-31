"""Example demonstrating artifact saving with pyexp using the class-based API.

For experiments that need to save artifacts to the output directory,
use the class-based Experiment API which provides access to self.out.
"""

import json
from pathlib import Path
from pyexp import Experiment, ExperimentRunner, Tensor


class ArtifactExperiment(Experiment):
    """Experiment that saves artifacts to the output directory."""

    # Attributes set in experiment()
    accuracy: float
    loss: float

    def experiment(self):
        """Run experiment and save artifacts to out."""
        print(f"Running: {self.name}")

        # Access nested config with dot notation
        lr = self.cfg.optimizer.learning_rate
        batch_size = self.cfg.optimizer.batch_size

        # Simulate training
        self.accuracy = 0.85 + lr * 10
        self.loss = 0.5 - lr * 5

        # Save artifacts to the experiment's output directory
        with open(self.out / "metrics.json", "w") as f:
            json.dump({"accuracy": self.accuracy, "loss": self.loss}, f, indent=2)

    @staticmethod
    def configs() -> list[dict]:
        """Generate configs with nested structure."""
        return [
            {
                "name": "small-lr",
                "optimizer": {"learning_rate": 0.001, "batch_size": 32},
            },
            {
                "name": "large-lr",
                "optimizer": {"learning_rate": 0.01, "batch_size": 64},
            },
        ]

    @staticmethod
    def report(results: Tensor["ArtifactExperiment"], report_dir: Path):
        """Print summary report."""
        print("\n=== Results ===")
        for exp in results:
            print(f"{exp.name}: acc={exp.accuracy:.3f}, loss={exp.loss:.3f}")

            # Load saved artifacts
            metrics_path = exp.out / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    saved = json.load(f)
                print(f"  Saved metrics: {saved}")


if __name__ == "__main__":
    runner = ExperimentRunner(ArtifactExperiment)
    runner.run()
