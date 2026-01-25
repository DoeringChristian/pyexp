"""Experiment runner: Experiment class and decorators."""

from functools import wraps
from pathlib import Path
from typing import Callable, Any
import argparse
import hashlib
import json
import pickle

from .config import Config, Tensor
from .executors import Executor, ExecutorName, get_executor


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


def _resolve_executor(
    executor: ExecutorName | Executor | None,
    default: ExecutorName | Executor,
) -> Executor:
    """Resolve executor parameter to an Executor instance."""
    if executor is None:
        return get_executor(default)
    return get_executor(executor)


class Experiment:
    """An experiment that can be run with configs and report functions."""

    def __init__(
        self,
        fn: Callable[[dict], Any],
        *,
        executor: ExecutorName | Executor = "subprocess",
    ):
        self._fn = fn
        self._executor_default = executor
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

    def run(
        self,
        configs: Callable[[], list[dict]] | None = None,
        report: Callable[[Tensor], Any] | None = None,
        output_dir: str | Path = "out",
        executor: ExecutorName | Executor | None = None,
    ) -> Any:
        """Execute the full pipeline: configs -> experiments -> report.

        Args:
            configs: Optional configs function. If not provided, uses @experiment.configs decorated function.
            report: Optional report function. If not provided, uses @experiment.report decorated function.
                    Receives a Tensor where each result has 'config' and 'name' keys.
            output_dir: Directory for caching experiment results. Defaults to "out".
            executor: Execution strategy for running experiments. Can be:
                - "subprocess": Run in isolated subprocess using cloudpickle (default, cross-platform)
                - "fork": Run in forked process (Unix only, guarantees same module state)
                - "inline": Run in same process (no isolation, useful for debugging)
                - "ray": Run using Ray for distributed execution (requires `pip install pyexp[ray]`)
                - An Executor instance: Use custom executor
                Defaults to the value set in @experiment decorator ("subprocess" if not specified).
        """
        exec_instance = _resolve_executor(executor, self._executor_default)
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
                experiment_dir.mkdir(parents=True, exist_ok=True)
                config_with_out = Config({**config, "out": experiment_dir})
                result = exec_instance.run(self._fn, config_with_out, result_path)
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


def experiment(
    fn: Callable[[dict], Any] | None = None,
    *,
    executor: ExecutorName | Executor = "subprocess",
) -> Experiment | Callable[[Callable[[dict], Any]], Experiment]:
    """Decorator to create an Experiment from a function.

    Args:
        executor: Default execution strategy for running experiments. Can be:
            - "subprocess": Run in isolated subprocess using cloudpickle (default, cross-platform)
            - "fork": Run in forked process (Unix only, guarantees same module state)
            - "inline": Run in same process (no isolation, useful for debugging)
            - "ray": Run using Ray for distributed execution (requires `pip install pyexp[ray]`)
            - An Executor instance: Use custom executor
            Can be overridden in run().

    Example usage:

        @pyexp.experiment
        def my_experiment(config):
            ...
            return {"accuracy": 0.95}

        # Or with arguments:
        @pyexp.experiment(executor="fork")
        def my_experiment(config):
            ...

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

        # Option 3: Override executor at runtime
        my_experiment.run(executor="inline")  # Run without isolation for debugging
    """
    def decorator(f: Callable[[dict], Any]) -> Experiment:
        return Experiment(f, executor=executor)

    if fn is not None:
        # Called without arguments: @experiment
        return Experiment(fn, executor=executor)
    else:
        # Called with arguments: @experiment(executor="fork")
        return decorator
