"""Experiment runner: Experiment class and decorators."""

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Any
import argparse
import hashlib
import inspect
import json
import os
import pickle
import sys

from .config import Config, Result, Tensor
from .executors import Executor, ExecutorName, get_executor
from .log import LogReader


def _wants_logger(fn: Callable) -> bool:
    """Check if the experiment function wants a logger parameter."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    # If function has 2+ parameters, second one is the logger
    return len(params) >= 2


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


def _generate_timestamp() -> str:
    """Generate a timestamp string for the current time."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _get_latest_timestamp(base_dir: Path) -> str | None:
    """Find the most recent timestamp folder in the base directory."""
    if not base_dir.exists():
        return None
    # List directories that look like timestamps (YYYY-MM-DD_HH-MM-SS)
    timestamp_dirs = [
        d
        for d in base_dir.iterdir()
        if d.is_dir() and len(d.name) == 19 and d.name[4] == "-" and d.name[10] == "_"
    ]
    if not timestamp_dirs:
        return None
    # Sort by name (timestamps sort lexicographically)
    timestamp_dirs.sort(key=lambda d: d.name, reverse=True)
    return timestamp_dirs[0].name


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Skip experiments and only generate report from cached results",
    )
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
        "--viewer",
        action="store_true",
        help="Start the viewer after experiments complete",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Port for the viewer (default: 8765)",
    )
    parser.add_argument(
        "--no-stash",
        action="store_true",
        help="Disable git stash (don't capture repository state)",
    )
    return parser.parse_args()


class _ProgressBar:
    """Simple progress bar for experiment execution."""

    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.current = 0
        self.passed = 0
        self.failed = 0
        self.cached = 0
        self._render()  # Show initial state

    def start(self, name: str = ""):
        """Show that an experiment is starting."""
        self._render(name, running=True)

    def update(self, status: str, name: str = ""):
        """Update progress with status: 'passed', 'failed', or 'cached'."""
        self.current += 1
        if status == "passed":
            self.passed += 1
        elif status == "failed":
            self.failed += 1
        elif status == "cached":
            self.cached += 1
        self._render(name)

    def _render(self, name: str = "", running: bool = False):
        """Render the progress bar."""
        import sys

        pct = self.current / self.total if self.total > 0 else 1
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        # Build status string
        parts = []
        if self.passed:
            parts.append(f"\033[32m{self.passed} passed\033[0m")
        if self.failed:
            parts.append(f"\033[31m{self.failed} failed\033[0m")
        if self.cached:
            parts.append(f"\033[33m{self.cached} cached\033[0m")
        status = ", ".join(parts) if parts else ""

        # Truncate name if too long
        max_name_len = 30
        display_name = name[:max_name_len] + "..." if len(name) > max_name_len else name

        line = f"\r{bar} {self.current}/{self.total}"
        if status:
            line += f" {status}"
        if display_name:
            if running:
                line += f" \033[36m[running: {display_name}]\033[0m"
            else:
                line += f" [{display_name}]"

        # Clear to end of line and print
        sys.stderr.write(f"{line}\033[K")
        sys.stderr.flush()

    def finish(self):
        """Print final summary."""
        import sys

        sys.stderr.write("\n")
        sys.stderr.flush()


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
        name: str | None = None,
        executor: ExecutorName | Executor | str = "subprocess",
        retry: int = 4,
        viewer: bool = False,
        viewer_port: int = 8765,
        stash: bool = True,
    ):
        self._fn = fn
        self._name = name or fn.__name__
        self._executor_default = executor
        self._retry_default = retry
        self._viewer_default = viewer
        self._viewer_port_default = viewer_port
        self._stash_default = stash
        self._wants_logger = _wants_logger(fn)
        self._configs_fn: Callable[[], list[dict]] | None = None
        self._report_fn: Callable[[Tensor, Path], Any] | None = None
        wraps(fn)(self)

    def __call__(self, config: dict) -> Any:
        """Run the experiment function directly."""
        return self._fn(config)

    def configs(self, fn: Callable[[], list[dict]]) -> Callable[[], list[dict]]:
        """Decorator to register the configs generator function."""
        self._configs_fn = fn
        return fn

    def report(
        self, fn: Callable[[Tensor, Path], Any]
    ) -> Callable[[Tensor, Path], Any]:
        """Decorator to register the report function.

        The report function receives:
            results: A Tensor of Result objects with .config, .result, .error, .log
            out: Path to a 'report' directory for saving outputs (plots, tables, etc.)
        """
        self._report_fn = fn
        return fn

    def run(
        self,
        configs: Callable[[], list[dict]] | None = None,
        report: Callable[[Tensor, Path], Any] | None = None,
        output_dir: str | Path = "out",
        executor: ExecutorName | Executor | str | None = None,
        name: str | None = None,
        retry: int | None = None,
        viewer: bool | None = None,
        viewer_port: int | None = None,
        stash: bool | None = None,
    ) -> Any:
        """Execute the full pipeline: configs -> experiments -> report.

        Args:
            configs: Optional configs function. If not provided, uses @experiment.configs decorated function.
            report: Optional report function (results, out) -> Any. If not provided, uses
                    @experiment.report decorated function. Receives a Tensor of Results and
                    a Path to the report directory for saving outputs.
            output_dir: Base directory for caching experiment results. Defaults to "out".
            executor: Execution strategy for running experiments. Can be:
                - "subprocess": Run in isolated subprocess using cloudpickle (default, cross-platform)
                - "fork": Run in forked process (Unix only, guarantees same module state)
                - "inline": Run in same process (no isolation, useful for debugging)
                - "ray": Run using Ray locally (requires `pip install pyexp[ray]`)
                - "ray:<address>" or "ray://host:port": Run on Ray cluster (e.g., "ray:auto", "ray://cluster:10001")
                - An Executor instance: Use custom executor
                Defaults to the value set in @experiment decorator ("subprocess" if not specified).
            name: Experiment name for the output folder. Defaults to function name.
            retry: Number of times to retry a failed experiment. Defaults to decorator value (4).
            viewer: If True, start the viewer after experiments complete. Defaults to decorator value (False).
            viewer_port: Port for the viewer. Defaults to decorator value (8765).
            stash: If True, capture git repository state and log commit hash. Defaults to decorator value (True).

        Output folder structure:
            out/<name>/<timestamp>/<config_name>-<hash>/
            Report directory: out/<name>/<timestamp>/report/
        """
        args = _parse_args()

        # Resolve parameters with CLI override priority:
        # CLI args > run() args > decorator args
        resolved_executor = args.executor or executor or self._executor_default
        configs_fn = configs or self._configs_fn
        report_fn = report or self._report_fn
        exp_name = args.name or name or self._name
        resolved_output_dir = args.output_dir or output_dir

        # Retry: CLI > run() arg > decorator arg
        if args.retry is not None:
            max_retries = args.retry
        elif retry is not None:
            max_retries = retry
        else:
            max_retries = self._retry_default

        # Viewer: CLI > run() arg > decorator arg
        if args.viewer:
            start_viewer = True
        elif viewer is not None:
            start_viewer = viewer
        else:
            start_viewer = self._viewer_default

        # Viewer port: CLI > run() arg > decorator arg
        if args.viewer_port != 8765:  # Non-default means explicitly set
            resolved_viewer_port = args.viewer_port
        elif viewer_port is not None:
            resolved_viewer_port = viewer_port
        else:
            resolved_viewer_port = self._viewer_port_default

        # Stash: CLI > run() arg > decorator arg
        if args.no_stash:
            enable_stash = False
        elif stash is not None:
            enable_stash = stash
        else:
            enable_stash = self._stash_default

        # If executor is already an Executor instance, use it directly
        if isinstance(resolved_executor, Executor):
            exec_instance = resolved_executor
        # Parse "ray://..." (Ray URI) or "ray:<address>" format for remote Ray execution
        elif isinstance(resolved_executor, str) and (
            resolved_executor.startswith("ray://")
            or (
                resolved_executor.startswith("ray:")
                and not resolved_executor.startswith("ray://")
            )
        ):
            from .executors import RayExecutor

            if resolved_executor.startswith("ray://"):
                address = resolved_executor  # Use full URI as address
            else:
                address = resolved_executor[4:]  # Remove "ray:" prefix
            exec_instance = RayExecutor(
                address=address,
                runtime_env={"working_dir": "."},
            )
        else:
            exec_instance = get_executor(resolved_executor)

        if configs_fn is None:
            raise RuntimeError(
                "No configs function provided. Use @experiment.configs or pass configs= argument."
            )
        if report_fn is None:
            raise RuntimeError(
                "No report function provided. Use @experiment.report or pass report= argument."
            )

        base_dir = Path(resolved_output_dir) / exp_name

        # --report requires --continue (otherwise we'd create an empty new run)
        if args.report and not args.continue_run:
            raise RuntimeError(
                "--report requires --continue to specify which run to report from"
            )

        # Determine the run directory (always timestamped)
        if args.continue_run:
            if args.continue_run == "latest":
                # Continue from the most recent timestamp
                latest = _get_latest_timestamp(base_dir)
                if latest is None:
                    raise RuntimeError(f"No previous runs found in {base_dir}")
                run_dir = base_dir / latest
            else:
                # Continue from specified timestamp
                run_dir = base_dir / args.continue_run
                if not run_dir.exists():
                    raise RuntimeError(f"Run not found: {run_dir}")
        else:
            # Generate new timestamp
            run_dir = base_dir / _generate_timestamp()

        # Print run info
        timestamp = run_dir.name
        print(f"Run: {exp_name}/{timestamp}")

        # Start viewer in background if requested
        viewer_process = None
        if start_viewer:
            import subprocess as sp

            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Viewer: http://localhost:{resolved_viewer_port}")
            viewer_process = sp.Popen(
                [
                    sys.executable,
                    "-m",
                    "solara",
                    "run",
                    "pyexp._viewer_app:Page",
                    "--port",
                    str(resolved_viewer_port),
                ],
                env={**os.environ, "PYEXP_LOG_DIR": str(run_dir.absolute())},
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )

        config_list = configs_fn()

        # Get shape from config_list if it's a Tensor
        if isinstance(config_list, Tensor):
            shape = config_list.shape
        else:
            shape = (len(config_list),)

        # Flatten config_list for iteration
        flat_configs = list(config_list)

        # Initialize progress bar if capturing output
        show_progress = not args.no_capture and not args.report
        progress = _ProgressBar(len(flat_configs)) if show_progress else None

        results = []

        for config in flat_configs:
            assert (
                "out" not in config
            ), "Config cannot contain 'out' key; it is reserved"
            experiment_dir = _get_experiment_dir(config, run_dir)
            result_path = experiment_dir / "result.pkl"
            config_name = config.get("name", "")

            if args.report:
                if not result_path.exists():
                    raise RuntimeError(
                        f"No cached result for config {config}. Run experiments first."
                    )
                with open(result_path, "rb") as f:
                    structured = pickle.load(f)
                status = "cached"
                # Ensure marker file exists for viewer discovery
                marker_path = experiment_dir / ".pyexp"
                marker_path.touch(exist_ok=True)
            elif not result_path.exists():
                experiment_dir.mkdir(parents=True, exist_ok=True)
                config_with_out = Config(
                    {
                        **config,
                        "out": experiment_dir,
                        "_stash": enable_stash,
                        "_wants_logger": self._wants_logger,
                    }
                )

                # Show running status
                if progress:
                    progress.start(config_name)

                # Retry loop
                for attempt in range(max_retries + 1):
                    structured = exec_instance.run(
                        self._fn,
                        config_with_out,
                        result_path,
                        capture=not args.no_capture,
                    )
                    if not structured.get("error"):
                        break  # Success, exit retry loop
                    if attempt < max_retries:
                        # Delete result.pkl before retry so executor writes fresh
                        result_path.unlink(missing_ok=True)

                # Save log to plaintext file
                log_path = experiment_dir / "log.out"
                log_path.write_text(structured.get("log", ""))
                # Determine status based on error field
                if structured.get("error"):
                    status = "failed"
                else:
                    status = "passed"
            else:
                # Load cached result
                experiment_dir.mkdir(parents=True, exist_ok=True)
                # Ensure marker file exists for viewer discovery
                marker_path = experiment_dir / ".pyexp"
                marker_path.touch(exist_ok=True)
                with open(result_path, "rb") as f:
                    structured = pickle.load(f)
                status = "cached"

            # Update progress bar
            if progress:
                progress.update(status, config_name)

            # Create Result object with config (without 'out' key)
            config_without_out = {k: v for k, v in config.items() if k != "out"}

            # Add LogReader if logger was used
            log_reader = None
            if self._wants_logger:
                log_reader = LogReader(experiment_dir)

            result_obj = Result(
                config=config_without_out,
                result=structured.get("result"),
                error=structured.get("error"),
                log=structured.get("log", ""),
                logger=log_reader,
            )
            results.append(result_obj)

        # Finish progress bar
        if progress:
            progress.finish()

        results = Tensor(results, shape)

        # Create report directory
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        return report_fn(results, report_dir)


def experiment(
    fn: Callable[[dict], Any] | None = None,
    *,
    name: str | None = None,
    executor: ExecutorName | Executor | str = "subprocess",
    retry: int = 4,
    viewer: bool = False,
    viewer_port: int = 8765,
    stash: bool = True,
) -> Experiment | Callable[[Callable[[dict], Any]], Experiment]:
    """Decorator to create an Experiment from a function.

    Args:
        name: Experiment name for the output folder. Defaults to function name.
        executor: Default execution strategy for running experiments. Can be:
            - "subprocess": Run in isolated subprocess using cloudpickle (default, cross-platform)
            - "fork": Run in forked process (Unix only, guarantees same module state)
            - "inline": Run in same process (no isolation, useful for debugging)
            - "ray": Run using Ray locally (requires `pip install pyexp[ray]`)
            - "ray:<address>" or "ray://host:port": Run on Ray cluster (e.g., "ray:auto", "ray://cluster:10001")
            - An Executor instance: Use custom executor
            Can be overridden in run().
        retry: Number of times to retry a failed experiment. Defaults to 4.
            Can be overridden in run() or via --retry CLI argument.
        viewer: If True, start the viewer after experiments complete. Defaults to False.
            Can be overridden in run() or via --viewer CLI argument.
        viewer_port: Port for the viewer. Defaults to 8765.
            Can be overridden in run() or via --viewer-port CLI argument.
        stash: If True (default), capture git repository state and log commit hash at iteration 0.
            Can be overridden in run() or via --no-stash CLI argument.

    Output folder structure:
        out/<name>/<timestamp>/<config_name>-<hash>/

    Example usage:

        @pyexp.experiment
        def my_experiment(config):
            ...
            return {"accuracy": 0.95}

        # Or with arguments:
        @pyexp.experiment(name="mnist", executor="fork")
        def my_experiment(config):
            ...

        # Ray on remote cluster:
        @pyexp.experiment(executor="ray://cluster:10001")
        def my_experiment(config):
            ...

        @my_experiment.configs
        def configs():
            return [{"name": "exp", "lr": 0.01}, {"name": "exp2", "lr": 0.001}]

        @my_experiment.report
        def report(results, report_dir):
            # Each result has .name, .config, .result, .error, .log, .logger
            # Filter by config values:
            lr_001 = results[{"config.lr": 0.001}]
            # Access fields:
            for r in results:
                print(f"{r.name}: {r.result['accuracy']}")

        my_experiment.run()

        # Option 2: Pass functions directly
        my_experiment.run(configs=configs_fn, report=report_fn)

        # Option 3: Override settings at runtime
        my_experiment.run(executor="ray:auto")

    CLI arguments (override decorator/run settings):
        --name NAME           Override experiment name
        --executor EXECUTOR   Override executor (subprocess, fork, inline, ray, ray:<address>)
        --output-dir DIR      Override output directory
        --continue [TIMESTAMP] Continue a previous run (latest if no timestamp given)
        --retry N             Number of retries on failure (default: 4)
        --report              Only generate report from cached results
        -s, --capture=no      Show subprocess output instead of progress bar
        --viewer              Start the viewer after experiments complete
        --viewer-port PORT    Port for the viewer (default: 8765)
        --no-stash            Disable git stash (don't capture repository state)

    Priority: CLI args > run() args > decorator args
    """

    def decorator(f: Callable[[dict], Any]) -> Experiment:
        return Experiment(
            f,
            name=name,
            executor=executor,
            retry=retry,
            viewer=viewer,
            viewer_port=viewer_port,
            stash=stash,
        )

    if fn is not None:
        # Called without arguments: @experiment
        return Experiment(
            fn,
            name=name,
            executor=executor,
            retry=retry,
            viewer=viewer,
            viewer_port=viewer_port,
            stash=stash,
        )
    else:
        # Called with arguments: @experiment(name="mnist", executor="fork")
        return decorator
