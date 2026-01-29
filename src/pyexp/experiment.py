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


_VALID_CONFIG_TYPES = (int, float, str, bool, type(None), Path)


def _validate_config_value(value: Any, path: str = "") -> None:
    """Validate that a config value contains only base types.

    Allowed: int, float, str, bool, None, Path, and containers (dict, list, tuple, set)
    of these types. dict and Config are allowed as containers.

    Raises:
        TypeError: If a value is not a valid config type (e.g., functions, classes).
    """
    if isinstance(value, _VALID_CONFIG_TYPES):
        return
    if isinstance(value, dict):
        for k, v in value.items():
            key_path = f"{path}.{k}" if path else str(k)
            if not isinstance(k, str):
                raise TypeError(
                    f"Config key {key_path!r} must be a string, got {type(k).__name__}"
                )
            _validate_config_value(v, key_path)
        return
    if isinstance(value, (list, tuple, set)):
        for i, item in enumerate(value):
            _validate_config_value(item, f"{path}[{i}]")
        return
    raise TypeError(
        f"Invalid config value at '{path}': {type(value).__name__} = {value!r}. "
        f"Only base types (int, float, str, bool, None, Path) and containers "
        f"(dict, list, tuple, set) are allowed."
    )


def _validate_configs(configs: list[dict]) -> None:
    """Validate all configs in a list."""
    for i, config in enumerate(configs):
        name = config.get("name", f"config[{i}]")
        try:
            _validate_config_value(config)
        except TypeError as e:
            raise TypeError(f"Config '{name}': {e}") from None


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


def _save_configs_json(
    run_dir: Path, shape: tuple, paths: list[Path]
) -> None:
    """Save run references and shape to a JSON file for later loading.

    Args:
        run_dir: The run directory.
        shape: Shape of the config tensor.
        paths: List of paths to the experiment directories (one per config).
    """
    # Store just the relative paths to run folders
    runs = [str(p.relative_to(run_dir)) for p in paths]

    data = {
        "runs": runs,
        "shape": list(shape),
    }
    configs_path = run_dir / "configs.json"
    configs_path.write_text(json.dumps(data, indent=2, default=str))


def _load_configs_json(run_dir: Path) -> tuple[list[Path], tuple]:
    """Load run paths and shape from configs.json.

    Returns:
        Tuple of (list of experiment directory paths, shape tuple).
    """
    configs_path = run_dir / "configs.json"
    if not configs_path.exists():
        raise FileNotFoundError(f"No configs.json found in {run_dir}")
    data = json.loads(configs_path.read_text())

    # Handle both old format (configs list) and new format (runs list)
    if "runs" in data:
        # New format: just paths to run folders
        paths = [run_dir / run for run in data["runs"]]
    else:
        # Old format: full configs with 'out' field
        paths = []
        for config in data["configs"]:
            if "out" in config:
                paths.append(run_dir / config["out"])
            else:
                # Very old format: compute from hash
                paths.append(_get_experiment_dir(config, run_dir))

    return paths, tuple(data["shape"])


def _load_results_from_dir(
    run_dir: Path,
    wants_logger: bool = False,
) -> Tensor:
    """Load results from a run directory using configs.json.

    Args:
        run_dir: Path to the run directory containing configs.json and experiment dirs.
        wants_logger: Whether to attach LogReader to results.

    Returns:
        Tensor of Result objects with the original shape.
    """
    experiment_dirs, shape = _load_configs_json(run_dir)

    results = []
    for experiment_dir in experiment_dirs:
        # Load config from individual config.json
        config_path = experiment_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            # Convert 'out' string back to Path if present
            if "out" in config:
                config["out"] = Path(config["out"])
        else:
            config = {}

        # Ensure 'out' is set to the experiment directory
        config["out"] = experiment_dir

        result_path = experiment_dir / "result.pkl"

        if not result_path.exists():
            raise FileNotFoundError(
                f"No result found for config '{config.get('name', config)}' at {result_path}"
            )

        with open(result_path, "rb") as f:
            structured = pickle.load(f)

        # Add LogReader if requested
        log_reader = None
        if wants_logger:
            log_reader = LogReader(experiment_dir)

        result_obj = Result(
            config=config,
            result=structured.get("result"),
            error=structured.get("error"),
            log=structured.get("log", ""),
            logger=log_reader,
        )
        results.append(result_obj)

    return Tensor(results, shape)


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


def _list_runs(base_dir: Path) -> None:
    """List all runs under the experiment base directory with their status."""
    if not base_dir.exists():
        print("No runs found.")
        return

    timestamp_dirs = [
        d
        for d in base_dir.iterdir()
        if d.is_dir() and len(d.name) == 19 and d.name[4] == "-" and d.name[10] == "_"
    ]
    if not timestamp_dirs:
        print("No runs found.")
        return

    timestamp_dirs.sort(key=lambda d: d.name)

    for run_dir in timestamp_dirs:
        # Scan config directories inside the run
        config_dirs = [
            d for d in run_dir.iterdir()
            if d.is_dir() and d.name != "report"
        ]
        total = len(config_dirs)
        completed = 0
        failed = 0
        for config_dir in config_dirs:
            result_path = config_dir / "result.pkl"
            if result_path.exists():
                try:
                    with open(result_path, "rb") as f:
                        structured = pickle.load(f)
                    if structured.get("error"):
                        failed += 1
                    else:
                        completed += 1
                except Exception:
                    failed += 1

        pending = total - completed - failed

        # Build status string
        parts = []
        if completed:
            parts.append(f"\033[32m{completed} passed\033[0m")
        if failed:
            parts.append(f"\033[31m{failed} failed\033[0m")
        if pending:
            parts.append(f"\033[33m{pending} pending\033[0m")
        status = ", ".join(parts) if parts else "empty"

        print(f"  {run_dir.name}  {status}  ({total} configs)")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument(
        "--report",
        nargs="?",
        const="latest",
        default=None,
        metavar="TIMESTAMP",
        help="Generate report from cached results. Without argument, uses the most recent run. "
        "With argument (e.g., --report=2024-01-25_14-30-00), uses that specific run.",
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
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all previous runs with their status",
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
        output_dir: str | Path | None = None,
        executor: ExecutorName | Executor | str = "subprocess",
        retry: int = 4,
        viewer: bool = False,
        viewer_port: int = 8765,
        stash: bool = True,
    ):
        self._fn = fn
        self._name = name or fn.__name__

        # Determine the default output directory relative to the experiment file
        if output_dir is not None:
            self._output_dir_default = Path(output_dir)
        else:
            # Use directory of the file containing the experiment function
            fn_file = fn.__globals__.get("__file__")
            if fn_file:
                self._output_dir_default = Path(fn_file).parent / "out"
            else:
                # Fallback for interactive/REPL usage
                self._output_dir_default = Path("out")

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

    def results(
        self,
        timestamp: str | None = None,
        output_dir: str | Path | None = None,
        name: str | None = None,
    ) -> Tensor:
        """Load results from a previous experiment run.

        Args:
            timestamp: Timestamp of the run to load (e.g., "2026-01-28_10-30-00").
                      If None or "latest", loads the most recent run.
            output_dir: Base directory where experiment results are stored.
                       Defaults to directory relative to experiment file.
            name: Experiment name. Defaults to the function name.

        Returns:
            Tensor of Result objects with .config, .result, .error, .log, .logger

        Example:
            # Load latest results
            results = my_experiment.results()

            # Load specific run
            results = my_experiment.results(timestamp="2026-01-28_10-30-00")

            # Access results
            for r in results:
                print(f"{r.name}: {r.result}")
        """
        exp_name = name or self._name
        resolved_output_dir = Path(output_dir) if output_dir else self._output_dir_default
        base_dir = resolved_output_dir / exp_name

        if timestamp is None or timestamp == "latest":
            latest = _get_latest_timestamp(base_dir)
            if latest is None:
                raise FileNotFoundError(f"No runs found in {base_dir}")
            run_dir = base_dir / latest
        else:
            run_dir = base_dir / timestamp
            if not run_dir.exists():
                raise FileNotFoundError(f"Run not found: {run_dir}")

        return _load_results_from_dir(run_dir, wants_logger=self._wants_logger)

    def run(
        self,
        configs: Callable[[], list[dict]] | None = None,
        report: Callable[[Tensor, Path], Any] | None = None,
        output_dir: str | Path | None = None,
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
            output_dir: Base directory for caching experiment results.
                       Defaults to "out" directory relative to experiment file.
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
            <output_dir>/<name>/<timestamp>/<config_name>-<hash>/
            Report directory: <output_dir>/<name>/<timestamp>/report/
        """
        args = _parse_args()

        # Resolve parameters with CLI override priority:
        # CLI args > run() args > decorator args
        resolved_executor = args.executor or executor or self._executor_default
        configs_fn = configs or self._configs_fn
        report_fn = report or self._report_fn
        exp_name = args.name or name or self._name
        # Output dir: CLI > run() arg > decorator arg (file-relative default)
        if args.output_dir:
            resolved_output_dir = Path(args.output_dir)
        elif output_dir is not None:
            resolved_output_dir = Path(output_dir)
        else:
            resolved_output_dir = self._output_dir_default

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

        # --list: show all runs and exit
        if args.list:
            print(f"Runs for {exp_name}:")
            _list_runs(base_dir)
            return None

        # --report implies --continue (use report's timestamp or latest)
        if args.report is not None and not args.continue_run:
            args.continue_run = args.report

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
            from pyexp._viewer import _find_free_port

            run_dir.mkdir(parents=True, exist_ok=True)
            resolved_viewer_port = _find_free_port(resolved_viewer_port)
            print(f"Viewer: http://localhost:{resolved_viewer_port}")
            viewer_process = sp.Popen(
                [
                    sys.executable,
                    "-m",
                    "solara",
                    "run",
                    "pyexp._viewer_app:Page",
                    "--host",
                    "localhost",
                    "--port",
                    str(resolved_viewer_port),
                ],
                env={**os.environ, "PYEXP_LOG_DIR": str(run_dir.absolute())},
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )

        config_list = configs_fn()
        _validate_configs(list(config_list))

        # Get shape from config_list if it's a Tensor
        if isinstance(config_list, Tensor):
            shape = config_list.shape
        else:
            shape = (len(config_list),)

        # Flatten config_list for iteration
        flat_configs = list(config_list)

        # Compute experiment directories for each config
        experiment_dirs = [_get_experiment_dir(config, run_dir) for config in flat_configs]

        # Build configs with 'out' field
        configs_with_out = []
        for config, experiment_dir in zip(flat_configs, experiment_dirs):
            assert (
                "out" not in config
            ), "Config cannot contain 'out' key; it is reserved"
            configs_with_out.append(Config({**config, "out": experiment_dir}))

        # Create all experiment directories and save configs upfront
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_configs_json(run_dir, shape, experiment_dirs)

        for config_with_out, experiment_dir in zip(configs_with_out, experiment_dirs):
            experiment_dir.mkdir(parents=True, exist_ok=True)
            config_json_path = experiment_dir / "config.json"
            config_json_path.write_text(
                json.dumps(dict(config_with_out), indent=2, default=str)
            )

        # Run experiments (skip if report-only mode)
        if not args.report:
            # Initialize progress bar if capturing output
            show_progress = not args.no_capture
            progress = _ProgressBar(len(flat_configs)) if show_progress else None

            for config_with_out, experiment_dir in zip(configs_with_out, experiment_dirs):
                result_path = experiment_dir / "result.pkl"
                config_name = config_with_out.get("name", "")

                if not result_path.exists():
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
                            wants_logger=self._wants_logger,
                            stash=enable_stash,
                        )
                        if not structured.get("error"):
                            break  # Success, exit retry loop

                        # Print error immediately on each failure
                        error_msg = structured.get("error", "")
                        remaining = max_retries - attempt
                        retry_info = f" (retrying, {remaining} left)" if remaining > 0 else ""
                        print(f"\n--- Error in {config_name or 'experiment'}{retry_info} ---")
                        print(error_msg)
                        if structured.get("log"):
                            print("--- Log output ---")
                            print(structured["log"])
                        print("---")

                        if attempt < max_retries:
                            # Delete result.pkl before retry so executor writes fresh
                            result_path.unlink(missing_ok=True)

                    # Save log to plaintext file
                    log_path = experiment_dir / "log.out"
                    log_path.write_text(structured.get("log", ""))
                    status = "failed" if structured.get("error") else "passed"
                else:
                    # Ensure marker file exists for viewer discovery
                    marker_path = experiment_dir / ".pyexp"
                    marker_path.touch(exist_ok=True)
                    status = "cached"

                # Update progress bar
                if progress:
                    progress.update(status, config_name)

            # Finish progress bar
            if progress:
                progress.finish()

        # Always load results from disk for the report
        results = _load_results_from_dir(run_dir, wants_logger=self._wants_logger)

        # Create report directory and run report
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        return report_fn(results, report_dir)


def experiment(
    fn: Callable[[dict], Any] | None = None,
    *,
    name: str | None = None,
    output_dir: str | Path | None = None,
    executor: ExecutorName | Executor | str = "subprocess",
    retry: int = 4,
    viewer: bool = False,
    viewer_port: int = 8765,
    stash: bool = True,
) -> Experiment | Callable[[Callable[[dict], Any]], Experiment]:
    """Decorator to create an Experiment from a function.

    Args:
        name: Experiment name for the output folder. Defaults to function name.
        output_dir: Base directory for experiment results. Defaults to "out" directory
            relative to the file containing the experiment function.
            Can be overridden in run() or via --output-dir CLI argument.
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
        <output_dir>/<name>/<timestamp>/<config_name>-<hash>/

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
        --report [TIMESTAMP]  Generate report from cached results (latest run or specific timestamp)
        --list                List all previous runs with their status
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
            output_dir=output_dir,
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
            output_dir=output_dir,
            executor=executor,
            retry=retry,
            viewer=viewer,
            viewer_port=viewer_port,
            stash=stash,
        )
    else:
        # Called with arguments: @experiment(name="mnist", executor="fork")
        return decorator
