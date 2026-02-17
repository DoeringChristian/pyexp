"""Experiment: user-facing wrapper and decorator for running experiments."""

from functools import wraps
from pathlib import Path
from typing import Callable, Any
import argparse

from .config import Config, Runs
from .executors import Executor, ExecutorName
from .runner import (
    Result,
    ExperimentRunner,
    _validate_unique_names,
    _validate_dependencies,
    _topological_sort,
    _discover_experiment_dirs,
    _get_latest_timestamp,
    _get_latest_finished_timestamp,
    _load_experiments,
    _list_runs,
    _print_dependency_graph,
)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument(
        "--continue",
        dest="continue_run",
        nargs="?",
        const="latest",
        default=None,
        metavar="TIMESTAMP",
        help="Continue from a previous run. Without argument, continues the most recent. "
        "With argument (e.g., --continue=2024-01-25_14-30-00), continues that specific run.",
    )
    parser.add_argument(
        "-s",
        "--capture=no",
        dest="no_capture",
        action="store_true",
        help="Show subprocess output instead of progress bar",
    )
    # Override arguments for decorator/run() settings
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NAME",
        help="Override experiment name",
    )
    parser.add_argument(
        "--executor",
        type=str,
        default=None,
        metavar="EXECUTOR",
        help="Override executor (subprocess, fork, inline, ray, ray:<address>)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Override output directory (default: out)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=None,
        metavar="N",
        help="Number of retries on failure (default: 4)",
    )
    parser.add_argument(
        "--no-stash",
        action="store_true",
        help="Disable git stash (don't capture repository state)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all previous runs with their status",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Print the dependency graph and exit",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        metavar="REGEX",
        help="Filter configs by name using a regex pattern (e.g., --filter 'lr_0\\.01.*')",
    )
    return parser.parse_args()


class Experiment:
    """User-facing experiment wrapper.

    Created via the @experiment decorator. Provides configs registration,
    CLI argument parsing, result loading, and delegates execution to
    ExperimentRunner.

    Usage:
        @pyexp.experiment
        def my_exp(cfg):
            return {"accuracy": 0.95}

        @my_exp.configs
        def configs():
            return [{"name": "test", "lr": 0.01}]

        my_exp.run()
        result = my_exp["test"]           # load result by name
        results = my_exp["pretrain.*"]    # Runs of matching Results
    """

    def __init__(
        self,
        fn: Callable,
        *,
        name: str | None = None,
        output_dir: str | Path | None = None,
        executor: ExecutorName | Executor | str = "subprocess",
        retry: int = 4,
        stash: bool = True,
        hash_configs: bool = False,
    ):
        import inspect

        self._fn = fn
        self._name = name
        self._output_dir = Path(output_dir) if output_dir else None
        self._executor_default = executor
        self._retry_default = retry
        self._stash_default = stash
        self._hash_configs_default = hash_configs
        self._configs_fn: Callable[[], list[dict]] | None = None

        # Detect signature
        sig = inspect.signature(fn)
        n_params = len(sig.parameters)
        self._wants_out = n_params >= 2
        self._wants_deps = n_params >= 3

    def configs(self, fn: Callable[[], list[dict]]) -> Callable[[], list[dict]]:
        """Decorator to register configs generator."""
        self._configs_fn = fn
        return fn

    def results(
        self,
        timestamp: str | None = None,
        output_dir: str | Path | None = None,
        name: str | None = None,
        finished: bool = False,
    ) -> Runs[Result]:
        """Load results from disk.

        Args:
            timestamp: Timestamp of the run to load. If None or "latest",
                      loads the most recent run.
            output_dir: Base directory where experiment results are stored.
            name: Experiment name. Defaults to the experiment's name.
            finished: If True, only include finished experiments.

        Returns:
            1D Runs of Result instances.
        """
        exp_name = name or self._name
        resolved_output_dir = Path(output_dir) if output_dir else self._output_dir
        if resolved_output_dir is None:
            resolved_output_dir = Path("out")
        base_dir = resolved_output_dir / exp_name

        if timestamp is None or timestamp == "latest":
            if finished:
                latest = _get_latest_finished_timestamp(base_dir)
            else:
                latest = _get_latest_timestamp(base_dir)
            if latest is None:
                return Runs([])
            timestamp = latest

        experiment_dirs = _discover_experiment_dirs(base_dir, timestamp)
        if not experiment_dirs:
            raise FileNotFoundError(f"Run not found: {timestamp}")

        return _load_experiments(base_dir, timestamp, finished_only=finished)

    def run(
        self,
        configs: Callable[[], list[dict]] | None = None,
        output_dir: str | Path | None = None,
        executor: ExecutorName | Executor | str | None = None,
        name: str | None = None,
        retry: int | None = None,
        stash: bool | None = None,
        hash_configs: bool | None = None,
    ) -> None:
        """Full pipeline: parse CLI args, create runner, submit, execute.

        Args:
            configs: Optional configs function override.
            output_dir: Base directory for experiment results.
            executor: Execution strategy for running experiments.
            name: Experiment name for the output folder.
            retry: Number of times to retry a failed experiment.
            stash: If True, capture git repository state.
            hash_configs: Append config parameter hash to run directory names.
        """
        args = _parse_args()

        # Resolve parameters: CLI > run() > constructor
        resolved_executor = args.executor or executor or self._executor_default
        exp_name = args.name or name or self._name

        # Resolve configs function: run() arg > decorated
        if configs is not None:
            configs_fn = configs
        elif self._configs_fn is not None:
            configs_fn = self._configs_fn
        else:
            configs_fn = None

        # Output dir: CLI > run() arg > constructor arg
        if args.output_dir:
            resolved_output_dir = Path(args.output_dir)
        elif output_dir is not None:
            resolved_output_dir = Path(output_dir)
        elif self._output_dir is not None:
            resolved_output_dir = self._output_dir
        else:
            resolved_output_dir = Path("out")

        # Retry: CLI > run() arg > constructor arg
        if args.retry is not None:
            max_retries = args.retry
        elif retry is not None:
            max_retries = retry
        else:
            max_retries = self._retry_default

        # Stash: CLI > run() arg > constructor arg
        if args.no_stash:
            enable_stash = False
        elif stash is not None:
            enable_stash = stash
        else:
            enable_stash = self._stash_default

        # Hash configs: run() arg > constructor arg
        if hash_configs is not None:
            enable_hash_configs = hash_configs
        else:
            enable_hash_configs = self._hash_configs_default

        if configs_fn is None:
            raise RuntimeError(
                "No configs function provided. "
                "Use @exp.configs decorator or pass configs= argument."
            )

        base_dir = Path(resolved_output_dir) / exp_name

        # --list: show all runs and exit
        if args.list:
            print(f"Runs for {exp_name}:")
            _list_runs(base_dir)
            return None

        # --graph: print dependency graph and exit
        if args.graph:
            config_list = configs_fn()
            flat_configs = list(config_list)
            _validate_unique_names(flat_configs)
            _validate_dependencies(flat_configs)
            flat_configs = _topological_sort(flat_configs)
            print(f"Dependency graph for {exp_name}:")
            _print_dependency_graph(flat_configs)
            return None

        # Resolve configs
        config_list = configs_fn()

        # Create runner and submit all configs
        runner = ExperimentRunner(
            name=exp_name,
            output_dir=resolved_output_dir,
            retry=max_retries,
        )
        for cfg in config_list:
            runner.submit(self._fn, cfg)

        # Run
        runner.run(
            executor=resolved_executor,
            capture=not args.no_capture,
            stash=enable_stash,
            hash_configs=enable_hash_configs,
            filter=args.filter,
            continue_run=args.continue_run,
        )

    def __getitem__(self, key):
        """Load results from latest run and index by key."""
        return self.results()[key]

    def __call__(self, *args, **kwargs):
        """Call the underlying experiment function directly."""
        return self._fn(*args, **kwargs)


def experiment(
    fn: Callable[[dict], Any] | None = None,
    *,
    name: str | None = None,
    output_dir: str | Path | None = None,
    executor: ExecutorName | Executor | str = "subprocess",
    retry: int = 4,
    stash: bool = True,
    hash_configs: bool = False,
) -> "Experiment | Callable[[Callable[[dict], Any]], Experiment]":
    """Decorator to create an Experiment from a function.

    The decorated function becomes an experiment where the return value
    is stored as exp.result on the Result dataclass.

    Args:
        name: Experiment name for the output folder. Defaults to function name.
        output_dir: Base directory for experiment results. Defaults to "out" directory
            relative to the file containing the experiment function.
        executor: Default execution strategy for running experiments.
        retry: Number of times to retry a failed experiment.
        stash: If True, capture git repository state.
        hash_configs: Append config parameter hash to run directory names.

    Example usage:

        @pyexp.experiment
        def my_experiment(cfg):
            return {"accuracy": train(cfg.lr)}

        @my_experiment.configs
        def configs():
            return [{"name": "exp", "lr": 0.01}]

        my_experiment.run()
    """

    def make_experiment(f: Callable[[dict], Any]) -> Experiment:
        # Determine output directory and default name relative to function's file
        resolved_output_dir = output_dir
        fn_file = f.__globals__.get("__file__")
        if resolved_output_dir is None and fn_file:
            resolved_output_dir = Path(fn_file).parent / "out"

        # Default name: <filename>.<function_name>
        if name:
            resolved_name = name
        elif fn_file:
            resolved_name = f"{Path(fn_file).stem}.{f.__name__}"
        else:
            resolved_name = f.__name__

        exp = Experiment(
            f,
            name=resolved_name,
            output_dir=resolved_output_dir,
            executor=executor,
            retry=retry,
            stash=stash,
            hash_configs=hash_configs,
        )

        # Copy function metadata for nice repr
        wraps(f)(exp)
        return exp

    if fn is not None:
        return make_experiment(fn)
    else:
        return make_experiment
