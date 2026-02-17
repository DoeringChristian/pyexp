"""Example demonstrating inter-run dependencies (pipeline)."""

import pyexp
from pyexp import Config


@pyexp.experiment
def pipeline(config: Config, out, deps):
    """Run a pipeline stage.

    Args:
        config: Experiment config with dot notation access.
        out: Output directory for this run.
        deps: Runs[Result] of completed dependency runs.
    """
    if config.name.startswith("pretrain"):
        print(f"Pretraining with lr={config.lr}, epochs={config.epochs}")
        # Simulate pretraining
        model = {"weights": [0.1, 0.2, 0.3], "type": "base"}
        return {"model": model, "loss": 0.1}

    if config.name == "finetune":
        # deps["pretrain.*"] returns a Runs with all matching pretrain experiments
        pretrain_runs = deps["pretrain.*"]
        base_model = pretrain_runs[0].result["model"]
        print(
            f"Finetuning from {len(pretrain_runs)} pretrain run(s) (type={base_model['type']})"
        )
        finetuned = {**base_model, "type": "finetuned", "extra": [0.4]}
        return {"model": finetuned, "loss": 0.05}

    if config.name == "evaluate":
        model = deps["finetune"].result["model"]
        print(f"Evaluating model (type={model['type']})")
        return {"accuracy": 0.95}


@pipeline.configs
def configs() -> list[dict]:
    """Pipeline: pretrain -> finetune -> evaluate."""
    return [
        {"name": "pretrain1", "lr": 0.01, "epochs": 100},
        {"name": "pretrain2", "lr": 0.01, "epochs": 100},
        {"name": "finetune", "lr": 0.001, "epochs": 10, "depends_on": "pretrain.*"},
        {
            "name": "evaluate",
            "depends_on": [
                "finetune",
                "pretrain.*",
            ],
        },
    ]


if __name__ == "__main__":
    pipeline.run()
