"""Experiment runner: Experiment class and decorators."""

from functools import wraps
from pathlib import Path
from typing import Callable, Any
import argparse
import hashlib
import json
import pickle

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
        self._report_fn: Callable[[list[dict], list[Any]], Any] | None = None
        wraps(fn)(self)

    def __call__(self, config: dict) -> Any:
        """Run the experiment function directly."""
        return self._fn(config)

    def configs(self, fn: Callable[[], list[dict]]) -> Callable[[], list[dict]]:
        """Decorator to register the configs generator function."""
        self._configs_fn = fn
        return fn

    def report(self, fn: Callable[[list[dict], list[Any]], Any]) -> Callable[[list[dict], list[Any]], Any]:
        """Decorator to register the report function."""
        self._report_fn = fn
        return fn

    def run(
        self,
        configs: Callable[[], list[dict]] | None = None,
        report: Callable[[list[dict], list[Any]], Any] | None = None,
        output_dir: str | Path = "out",
    ) -> Any:
        """Execute the full pipeline: configs -> experiments -> report.

        Args:
            configs: Optional configs function. If not provided, uses @experiment.configs decorated function.
            report: Optional report function. If not provided, uses @experiment.report decorated function.
            output_dir: Directory for caching experiment results. Defaults to "out".
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
            assert "out_dir" not in config, "Config cannot contain 'out_dir' key; it is reserved"
            experiment_dir = _get_experiment_dir(config, output_dir)
            result_path = experiment_dir / "result.pkl"

            if args.report:
                if not result_path.exists():
                    raise RuntimeError(f"No cached result for config {config}. Run experiments first.")
                with open(result_path, "rb") as f:
                    result = pickle.load(f)
            elif args.rerun or not result_path.exists():
                experiment_dir.mkdir(parents=True, exist_ok=True)
                config_with_out = Config({**config, "out_dir": experiment_dir})
                result = self._fn(config_with_out)
                with open(result_path, "wb") as f:
                    pickle.dump(result, f)
            else:
                with open(result_path, "rb") as f:
                    result = pickle.load(f)

            results.append(result)

        # Wrap configs and results in Tensors with matching shapes
        if not isinstance(config_list, Tensor):
            config_list = Tensor(list(config_list), shape)
        results = Tensor(results, shape)

        return report_fn(config_list, results)


def experiment(fn: Callable[[dict], Any]) -> Experiment:
    """Decorator to create an Experiment from a function.

    Example usage:

        @pyexp.experiment
        def my_experiment(config):
            ...

        # Option 1: Use decorators
        @my_experiment.configs
        def configs():
            return [{"lr": 0.01}, {"lr": 0.001}]

        @my_experiment.report
        def report(configs, results):
            ...

        my_experiment.run()

        # Option 2: Pass functions directly
        my_experiment.run(configs=configs_fn, report=report_fn)
    """
    return Experiment(fn)
