"""Experiment runner: Experiment class and decorators."""

from functools import wraps
from pathlib import Path
from typing import Callable, Any
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import tempfile

import cloudpickle

from .config import Config, Tensor


def _config_hash(config: dict) -> str:
    """Generate a short hash of the config for cache identification."""
    config_without_name = {k: v for k, v in config.items() if k != "name"}
    config_str = json.dumps(config_without_name, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def _get_experiment_dir(config: dict, output_dir: Path) -> Path:
    """Get the cache directory path for an experiment config."""
    name = config.get("name", "experiment")
    hash_str = _config_hash(config)
    return output_dir / f"{name}-{hash_str}"


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Skip experiments and only generate report from cached results",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run experiments ignoring cache",
    )
    return parser.parse_args()


class Experiment:
    """An experiment that can be run with configs and report functions."""

    def __init__(self, fn: Callable[[dict], Any]):
        self._fn = fn
        self._configs_fn: Callable[[], list[dict]] | None = None
        self._report_fn: Callable[[Tensor], Any] | None = None
        wraps(fn)(self)

    def __call__(self, config: dict) -> Any:
        """Run the experiment function directly."""
        return self._fn(config)

    def configs(self, fn: Callable[[], list[dict]]) -> Callable[[], list[dict]]:
        """Decorator to register the configs generator function."""
        self._configs_fn = fn
        return fn

    def report(self, fn: Callable[[Tensor], Any]) -> Callable[[Tensor], Any]:
        """Decorator to register the report function.

        The report function receives a single Tensor of results. Each result
        contains the experiment output plus 'config' and 'name' keys for filtering.
        """
        self._report_fn = fn
        return fn

    def _run_in_subprocess(self, config: dict, experiment_dir: Path, result_path: Path) -> dict:
        """Run a single experiment in an isolated subprocess.

        Args:
            config: The experiment config.
            experiment_dir: Directory for this experiment's outputs.
            result_path: Path where result should be written.

        Returns:
            The experiment result (may contain __error__ key if failed).
        """
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_with_out = Config({**config, "out": experiment_dir})

        # Create payload with serialized function
        payload = {
            "fn": self._fn,
            "config": config_with_out,
            "result_path": str(result_path),
        }

        # Write payload to temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            payload_path = f.name
            cloudpickle.dump(payload, f)

        try:
            # Run worker subprocess
            proc = subprocess.run(
                [sys.executable, "-m", "pyexp.worker", payload_path],
                capture_output=True,
                text=True,
            )

            # Check if result was written
            if result_path.exists():
                with open(result_path, "rb") as f:
                    return pickle.load(f)
            else:
                # Subprocess crashed before writing result
                return {
                    "__error__": True,
                    "type": "SubprocessError",
                    "message": f"Subprocess exited with code {proc.returncode}",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
        finally:
            # Clean up payload file
            Path(payload_path).unlink(missing_ok=True)

    def run(
        self,
        configs: Callable[[], list[dict]] | None = None,
        report: Callable[[Tensor], Any] | None = None,
        output_dir: str | Path = "out",
        isolate: bool = True,
    ) -> Any:
        """Execute the full pipeline: configs -> experiments -> report.

        Args:
            configs: Optional configs function. If not provided, uses @experiment.configs decorated function.
            report: Optional report function. If not provided, uses @experiment.report decorated function.
                    Receives a Tensor where each result has 'config' and 'name' keys.
            output_dir: Directory for caching experiment results. Defaults to "out".
            isolate: If True, run each experiment in a separate subprocess for robustness
                     against crashes (including segfaults). Defaults to True.
        """
        configs_fn = configs or self._configs_fn
        report_fn = report or self._report_fn

        if configs_fn is None:
            raise RuntimeError("No configs function provided. Use @experiment.configs or pass configs= argument.")
        if report_fn is None:
            raise RuntimeError("No report function provided. Use @experiment.report or pass report= argument.")

        args = _parse_args()
        output_dir = Path(output_dir)
        config_list = configs_fn()

        # Get shape from config_list if it's a Tensor
        if isinstance(config_list, Tensor):
            shape = config_list.shape
        else:
            shape = (len(config_list),)

        results = []

        for config in config_list:
            assert "out" not in config, "Config cannot contain 'out' key; it is reserved"
            experiment_dir = _get_experiment_dir(config, output_dir)
            result_path = experiment_dir / "result.pkl"

            if args.report:
                if not result_path.exists():
                    raise RuntimeError(f"No cached result for config {config}. Run experiments first.")
                with open(result_path, "rb") as f:
                    result = pickle.load(f)
            elif args.rerun or not result_path.exists():
                if isolate:
                    result = self._run_in_subprocess(config, experiment_dir, result_path)
                else:
                    experiment_dir.mkdir(parents=True, exist_ok=True)
                    config_with_out = Config({**config, "out": experiment_dir})
                    result = self._fn(config_with_out)
                    with open(result_path, "wb") as f:
                        pickle.dump(result, f)
            else:
                with open(result_path, "rb") as f:
                    result = pickle.load(f)

            # Wrap result with config and name for filtering
            config_without_out = {k: v for k, v in config.items() if k != "out"}
            if isinstance(result, dict) and not result.get("__error__"):
                wrapped_result = {
                    "name": config.get("name", ""),
                    "config": config_without_out,
                    **result,
                }
            elif isinstance(result, dict) and result.get("__error__"):
                # Keep error info but add config context
                wrapped_result = {
                    "name": config.get("name", ""),
                    "config": config_without_out,
                    **result,
                }
            else:
                wrapped_result = {
                    "name": config.get("name", ""),
                    "config": config_without_out,
                    "value": result,
                }
            results.append(wrapped_result)

        results = Tensor(results, shape)

        return report_fn(results)


def experiment(fn: Callable[[dict], Any]) -> Experiment:
    """Decorator to create an Experiment from a function.

    Example usage:

        @pyexp.experiment
        def my_experiment(config):
            ...
            return {"accuracy": 0.95}

        @my_experiment.configs
        def configs():
            return [{"name": "exp", "lr": 0.01}, {"name": "exp2", "lr": 0.001}]

        @my_experiment.report
        def report(results):
            # Each result has 'name', 'config', and experiment outputs
            # Filter by config values:
            lr_001 = results[{"config.lr": 0.001}]
            # Access fields:
            for r in results:
                print(f"{r['name']}: {r['accuracy']}")

        my_experiment.run()

        # Option 2: Pass functions directly
        my_experiment.run(configs=configs_fn, report=report_fn)
    """
    return Experiment(fn)
