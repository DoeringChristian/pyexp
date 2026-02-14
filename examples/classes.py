"""Example demonstrating the class-based Experiment API."""

from pathlib import Path
from pyexp import Experiment, ExperimentRunner, Runs


class MyExperiment(Experiment):
    """Example experiment class demonstrating the class-based API.

    The experiment instance IS the result - it gets pickled entirely after
    running, so you can set any attributes you want in experiment().
    """

    # Type hints for attributes set in experiment()
    accuracy: float
    epochs_run: int

    def experiment(self):
        """Run the experiment. Set attributes on self to store results."""
        # Access config via self.cfg (has dot notation access)
        lr = self.cfg.learning_rate
        epochs = self.cfg.epochs

        # Access output directory via self.out
        print(f"Output dir: {self.out}")
        print(f"Running with lr={lr}, epochs={epochs}")

        # Simulate training
        self.accuracy = 0.9 + lr * epochs / 100
        self.epochs_run = epochs

    @staticmethod
    def configs():
        """Return list of configs to run."""
        return [
            {"name": "fast", "learning_rate": 0.01, "epochs": 10},
            {"name": "medium", "learning_rate": 0.001, "epochs": 20},
            {"name": "slow", "learning_rate": 0.0001, "epochs": 50},
        ]

    @staticmethod
    def report(results: Runs["MyExperiment"], out: Path):
        """Generate report from results.

        Args:
            results: Runs of MyExperiment instances with full type info
            out: Path to report directory for saving outputs
        """
        print("\n=== Experiment Report ===")
        for exp in results:
            # Access custom attributes set in experiment()
            print(f"{exp.name}: accuracy={exp.accuracy:.4f}, epochs={exp.epochs_run}")

            # Access standard properties
            print(f"  cfg: {exp.cfg}")
            print(f"  out: {exp.out}")
            print(f"  error: {exp.error}")

        # Find best result
        best = max(results, key=lambda exp: exp.accuracy)
        print(f"\nBest: {best.name} with accuracy {best.accuracy:.4f}")


if __name__ == "__main__":
    # Create runner and execute
    runner = ExperimentRunner(MyExperiment)
    runner.run()
