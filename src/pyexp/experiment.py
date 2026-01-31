"""Experiment runner: Experiment base class, ExperimentRunner, and decorators."""

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Any, TYPE_CHECKING
import argparse
import hashlib
import json
import os
import pickle
import re
import sys

from .config import Config, Tensor
from .executors import Executor, ExecutorName, get_executor
from .log import LogReader

if TYPE_CHECKING:
    from .log import Logger


class Experiment:
    """Base class for experiments. Users subclass this to define experiments.

    The experiment instance IS the result - it gets pickled entirely after
    running, so you can access any attributes you set in experiment().

    Read-only properties (set by runner via name-mangled attrs):
        cfg: The config for this run (Config object with dot notation access)
        out: Output directory for this experiment run
        error: Populated after failure with error message
        log: Populated after run with stdout/stderr output
        name: Shorthand for cfg.get("name", "")

    Example:
        class MyExperiment(Experiment):
            accuracy: float
            model: Any

            def experiment(self):
                self.model = train(self.cfg.lr)
                self.accuracy = evaluate(self.model)

            @staticmethod
            def configs():
                return [{"name": "fast", "lr": 0.01}]

            @staticmethod
            def report(results: Tensor["MyExperiment"], out: Path):
                for exp in results:
                    print(f"{exp.name}: {exp.accuracy}")

        runner = ExperimentRunner(MyExperiment)
        runner.run()
    """

    # Private attributes (set by runner via name mangling)
    _Experiment__cfg: Config | None = None
    _Experiment__out: Path | None = None
    _Experiment__error: str | None = None
    _Experiment__log: str = ""

    @property
    def cfg(self) -> Config:
        """The configuration for this experiment run."""
        if self._Experiment__cfg is None:
            raise RuntimeError(
                "Experiment not initialized. cfg is only available inside experiment()."
            )
        return self._Experiment__cfg

    @property
    def out(self) -> Path:
        """Output directory for this experiment run."""
        if self._Experiment__out is None:
            raise RuntimeError(
                "Experiment not initialized. out is only available inside experiment()."
            )
        return self._Experiment__out

    @property
    def error(self) -> str | None:
        """Error message if experiment failed, None otherwise."""
        return self._Experiment__error

    @property
    def log(self) -> str:
        """Captured stdout/stderr from the experiment run."""
        return self._Experiment__log

    @property
    def name(self) -> str:
        """Shorthand for cfg.get('name', '')."""
        if self._Experiment__cfg is None:
            return ""
        return self._Experiment__cfg.get("name", "")

    def experiment(self) -> None:
        """User implements this to define the experiment.

        Set attributes on self to store results. These will be accessible
        after loading the experiment from disk.

        Example:
            def experiment(self):
                self.accuracy = train_and_evaluate(self.cfg)
        """
        raise NotImplementedError("Subclass must implement experiment()")

    @staticmethod
    def configs() -> list[dict] | Tensor:
        """Return list of configs to run, or a Tensor of configs.

        Example:
            @staticmethod
            def configs():
                return [
                    {"name": "fast", "lr": 0.01},
                    {"name": "slow", "lr": 0.001},
                ]
        """
        raise NotImplementedError("Subclass must implement configs()")

    @staticmethod
    def report(results: Tensor, out: Path) -> Any:
        """Generate report from experiment results.

        Args:
            results: Tensor of experiment instances with full type info
            out: Path to report directory for saving outputs

        Example:
            @staticmethod
            def report(results: Tensor["MyExperiment"], out: Path):
                for exp in results:
                    print(f"{exp.name}: {exp.accuracy}")
        """
        raise NotImplementedError("Subclass must implement report()")


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


def _sanitize_name(name: str) -> str:
    """Sanitize a config name for use as a directory name.

    Replaces characters that are problematic in file paths:
    - / and \\ (path separators)
    - : (Windows drive separator)
    - * ? " < > | (Windows reserved)
    """
    # Replace problematic characters with underscore
    for char in r'/\:*?"<>|':
        name = name.replace(char, "_")
    return name


def _get_experiment_dir(config: dict, output_dir: Path) -> Path:
    """Get the cache directory path for an experiment config."""
    name = _sanitize_name(config.get("name", "experiment"))
    hash_str = _config_hash(config)
    return output_dir / f"{name}-{hash_str}"


def _filter_by_name(items: list, pattern: str, get_name: callable) -> list:
    """Filter items by name using a regex pattern.

    Args:
        items: List of items to filter.
        pattern: Regex pattern to match against names.
        get_name: Function to extract name from an item.

    Returns:
        Filtered list of items whose names match the pattern.
    """
    regex = re.compile(pattern)
    return [item for item in items if regex.search(get_name(item) or "")]


def _save_configs_json(run_dir: Path, shape: tuple, paths: list[Path]) -> None:
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
    paths = [run_dir / run for run in data["runs"]]
    return paths, tuple(data["shape"])


def _load_experiments_from_dir(run_dir: Path) -> Tensor[Experiment]:
    """Load experiment instances from a run directory using configs.json.

    Args:
        run_dir: Path to the run directory containing configs.json and experiment dirs.

    Returns:
        Tensor of Experiment instances with the original shape.
    """
    experiment_dirs, shape = _load_configs_json(run_dir)

    results = []
    for experiment_dir in experiment_dirs:
        experiment_path = experiment_dir / "experiment.pkl"

        if experiment_path.exists():
            with open(experiment_path, "rb") as f:
                instance = pickle.load(f)
            results.append(instance)
        else:
            config_path = experiment_dir / "config.json"
            config_name = "unknown"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                config_name = config.get("name", config_name)
            raise FileNotFoundError(
                f"No experiment found for config '{config_name}' at {experiment_dir}"
            )

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
            d for d in run_dir.iterdir() if d.is_dir() and d.name != "report"
        ]
        total = len(config_dirs)
        completed = 0
        failed = 0
        for config_dir in config_dirs:
            experiment_path = config_dir / "experiment.pkl"
            if experiment_path.exists():
                try:
                    with open(experiment_path, "rb") as f:
                        instance = pickle.load(f)
                    if instance.error:
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
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        metavar="REGEX",
        help="Filter configs by name using a regex pattern (e.g., --filter 'lr_0\\.01.*')",
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


class ExperimentRunner:
    """Runner for executing experiments defined as Experiment subclasses or decorated functions.

    This class handles:
    - CLI argument parsing
    - Config generation and validation
    - Experiment execution via executors
    - Result caching and loading
    - Report generation

    Can be used with either class-based or decorator-based experiments:

    Class-based:
        class MyExperiment(Experiment):
            def experiment(self):
                self.accuracy = train(self.cfg)

            @staticmethod
            def configs():
                return [{"name": "exp1", "lr": 0.01}]

            @staticmethod
            def report(results, out):
                for exp in results:
                    print(exp.accuracy)

        runner = ExperimentRunner(MyExperiment)
        runner.run()

    Decorator-based:
        @pyexp.experiment
        def my_experiment(cfg):
            return {"accuracy": train(cfg)}

        my_experiment.run()  # ExperimentRunner is created internally
    """

    def __init__(
        self,
        experiment_class: type[Experiment],
        *,
        name: str | None = None,
        output_dir: str | Path | None = None,
        executor: ExecutorName | Executor | str = "subprocess",
        retry: int = 4,
        viewer: bool = False,
        viewer_port: int = 8765,
        stash: bool = True,
    ):
        """Create an ExperimentRunner.

        Args:
            experiment_class: Experiment subclass to run.
            name: Experiment name for output folder. Defaults to class name.
            output_dir: Base directory for results. Defaults to "out" relative to caller.
            executor: Execution strategy ("subprocess", "fork", "inline", "ray", etc.)
            retry: Number of retries on failure.
            viewer: Start viewer after experiments complete.
            viewer_port: Port for the viewer.
            stash: Capture git repository state.
        """
        self._experiment_class = experiment_class
        self._name = name or experiment_class.__name__

        # Determine the default output directory
        if output_dir is not None:
            self._output_dir_default = Path(output_dir)
        else:
            # Use current working directory / out
            self._output_dir_default = Path("out")

        self._executor_default = executor
        self._retry_default = retry
        self._viewer_default = viewer
        self._viewer_port_default = viewer_port
        self._stash_default = stash
        self._configs_fn: Callable[[], list[dict]] | None = None
        self._report_fn: Callable[[Tensor, Path], Any] | None = None

    def configs(self, fn: Callable[[], list[dict]]) -> Callable[[], list[dict]]:
        """Decorator to register the configs generator function (for decorator API)."""
        self._configs_fn = fn
        return fn

    def report(
        self, fn: Callable[[Tensor, Path], Any]
    ) -> Callable[[Tensor, Path], Any]:
        """Decorator to register the report function (for decorator API)."""
        self._report_fn = fn
        return fn

    def results(
        self,
        timestamp: str | None = None,
        output_dir: str | Path | None = None,
        name: str | None = None,
    ) -> Tensor[Experiment]:
        """Load results from a previous experiment run.

        Args:
            timestamp: Timestamp of the run to load (e.g., "2026-01-28_10-30-00").
                      If None or "latest", loads the most recent run.
            output_dir: Base directory where experiment results are stored.
            name: Experiment name. Defaults to class/function name.

        Returns:
            Tensor of Experiment instances with full attribute access.
        """
        exp_name = name or self._name
        resolved_output_dir = (
            Path(output_dir) if output_dir else self._output_dir_default
        )
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

        return _load_experiments_from_dir(run_dir)

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
            configs: Optional configs function. Uses class method or @configs decorated function.
            report: Optional report function. Uses class method or @report decorated function.
            output_dir: Base directory for caching experiment results.
            executor: Execution strategy for running experiments.
            name: Experiment name for the output folder.
            retry: Number of times to retry a failed experiment.
            viewer: If True, start the viewer after experiments complete.
            viewer_port: Port for the viewer.
            stash: If True, capture git repository state.
        """
        args = _parse_args()

        # Resolve parameters with CLI override priority:
        # CLI args > run() args > constructor args
        resolved_executor = args.executor or executor or self._executor_default
        exp_name = args.name or name or self._name

        # Resolve configs function: run() arg > decorated > class method
        if configs is not None:
            configs_fn = configs
        elif self._configs_fn is not None:
            configs_fn = self._configs_fn
        else:
            # Use class's configs method
            try:
                _ = self._experiment_class.configs()
                configs_fn = self._experiment_class.configs
            except NotImplementedError:
                configs_fn = None

        # Resolve report function: run() arg > decorated > class method
        if report is not None:
            report_fn = report
        elif self._report_fn is not None:
            report_fn = self._report_fn
        else:
            # Use class's report method - test if it's actually implemented
            try:
                # Try calling with dummy args to see if it raises NotImplementedError
                self._experiment_class.report(None, None)
                report_fn = self._experiment_class.report
            except NotImplementedError:
                report_fn = None
            except Exception:
                # If it fails for other reasons, assume it's implemented
                report_fn = self._experiment_class.report

        # Output dir: CLI > run() arg > constructor arg
        if args.output_dir:
            resolved_output_dir = Path(args.output_dir)
        elif output_dir is not None:
            resolved_output_dir = Path(output_dir)
        else:
            resolved_output_dir = self._output_dir_default

        # Retry: CLI > run() arg > constructor arg
        if args.retry is not None:
            max_retries = args.retry
        elif retry is not None:
            max_retries = retry
        else:
            max_retries = self._retry_default

        # Viewer: CLI > run() arg > constructor arg
        if args.viewer:
            start_viewer = True
        elif viewer is not None:
            start_viewer = viewer
        else:
            start_viewer = self._viewer_default

        # Viewer port: CLI > run() arg > constructor arg
        if args.viewer_port != 8765:  # Non-default means explicitly set
            resolved_viewer_port = args.viewer_port
        elif viewer_port is not None:
            resolved_viewer_port = viewer_port
        else:
            resolved_viewer_port = self._viewer_port_default

        # Stash: CLI > run() arg > constructor arg
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
                "No configs function provided. Implement configs() in your Experiment class, "
                "use @runner.configs decorator, or pass configs= argument."
            )
        if report_fn is None:
            raise RuntimeError(
                "No report function provided. Implement report() in your Experiment class, "
                "use @runner.report decorator, or pass report= argument."
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

        # =================================================================
        # PHASE 1: Config Generation (only on fresh start)
        # =================================================================
        # On fresh start: generate configs, compute hashes, create directories
        # On --continue or --report: skip this phase, use saved configs
        is_fresh_start = not args.continue_run and not args.report

        if is_fresh_start:
            config_list = configs_fn()
            _validate_configs(list(config_list))

            # Get shape from config_list if it's a Tensor
            if isinstance(config_list, Tensor):
                shape = config_list.shape
            else:
                shape = (len(config_list),)

            # Flatten config_list for iteration
            flat_configs = list(config_list)

            # Compute experiment directories for each config (hashes computed here)
            experiment_dirs = [
                _get_experiment_dir(config, run_dir) for config in flat_configs
            ]

            # Build configs (without 'out' - that's set separately now)
            configs_for_save = []
            for config, experiment_dir in zip(flat_configs, experiment_dirs):
                assert (
                    "out" not in config
                ), "Config cannot contain 'out' key; it is reserved"
                configs_for_save.append(Config(config))

            # Create run directory and save configs.json
            run_dir.mkdir(parents=True, exist_ok=True)
            _save_configs_json(run_dir, shape, experiment_dirs)

            # Create all experiment directories and save individual config.json files
            for config, experiment_dir in zip(configs_for_save, experiment_dirs):
                experiment_dir.mkdir(parents=True, exist_ok=True)
                config_json_path = experiment_dir / "config.json"
                config_json_path.write_text(
                    json.dumps(dict(config), indent=2, default=str)
                )

        # =================================================================
        # PHASE 2: Experiment Execution (skip if --report)
        # =================================================================
        # Load configs from saved config.json files and run experiments
        if not args.report:
            # Load experiment directories and shape from configs.json
            experiment_dirs, shape = _load_configs_json(run_dir)

            # Apply filter if specified
            if args.filter:

                def get_config_name(exp_dir: Path) -> str:
                    config_path = exp_dir / "config.json"
                    if config_path.exists():
                        config = json.loads(config_path.read_text())
                        return config.get("name", "")
                    return ""

                original_count = len(experiment_dirs)
                experiment_dirs = _filter_by_name(
                    experiment_dirs, args.filter, get_config_name
                )
                if not experiment_dirs:
                    print(f"No configs match filter '{args.filter}'")
                    return None
                print(
                    f"Filter '{args.filter}': {len(experiment_dirs)}/{original_count} configs selected"
                )

            # Initialize progress bar if capturing output
            show_progress = not args.no_capture
            progress = _ProgressBar(len(experiment_dirs)) if show_progress else None

            for experiment_dir in experiment_dirs:
                experiment_path = experiment_dir / "experiment.pkl"

                # Load config from saved config.json
                config_json_path = experiment_dir / "config.json"
                config_data = json.loads(config_json_path.read_text())
                config = Config(config_data)
                config_name = config.get("name", "")

                if not experiment_path.exists():
                    # Show running status
                    if progress:
                        progress.start(config_name)

                    # Create fresh experiment instance
                    instance = self._experiment_class()
                    # Set private attributes via name mangling
                    instance._Experiment__cfg = config
                    instance._Experiment__out = experiment_dir

                    # Retry loop
                    for attempt in range(max_retries + 1):
                        exec_instance.run(
                            instance,
                            experiment_path,
                            capture=not args.no_capture,
                            stash=enable_stash,
                        )

                        # Reload instance to check result
                        with open(experiment_path, "rb") as f:
                            instance = pickle.load(f)

                        if not instance.error:
                            break  # Success, exit retry loop

                        # Print error immediately on each failure
                        error_msg = instance.error or ""
                        remaining = max_retries - attempt
                        retry_info = (
                            f" (retrying, {remaining} left)" if remaining > 0 else ""
                        )
                        print(
                            f"\n--- Error in {config_name or 'experiment'}{retry_info} ---"
                        )
                        print(error_msg)
                        if instance.log:
                            print("--- Log output ---")
                            print(instance.log)
                        print("---")

                        if attempt < max_retries:
                            # Delete experiment.pkl before retry so executor writes fresh
                            experiment_path.unlink(missing_ok=True)
                            # Create fresh instance for retry
                            instance = self._experiment_class()
                            instance._Experiment__cfg = config
                            instance._Experiment__out = experiment_dir

                    # Save log to plaintext file
                    log_path = experiment_dir / "log.out"
                    log_path.write_text(instance.log or "")
                    status = "failed" if instance.error else "passed"
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

        # =================================================================
        # PHASE 3: Report Generation (always runs)
        # =================================================================
        # Load results from disk and run report function
        results = _load_experiments_from_dir(run_dir)

        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        return report_fn(results, report_dir)


class _DecoratorExperiment(Experiment):
    """Dynamic Experiment subclass created by the @experiment decorator.

    Stores the function's return value in self.result.
    """

    result: Any = None
    # _fn is set on the class by make_runner to be the wrapped function
    _fn: Callable | None = None
    # _wants_out is set by make_runner based on function signature
    _wants_out: bool = False

    def experiment(self) -> None:
        fn = type(self)._fn
        if fn is not None:
            if type(self)._wants_out:
                self.result = fn(self.cfg, self.out)
            else:
                self.result = fn(self.cfg)

    @staticmethod
    def configs():
        raise NotImplementedError()

    @staticmethod
    def report(results, out):
        raise NotImplementedError()


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
) -> "ExperimentRunner | Callable[[Callable[[dict], Any]], ExperimentRunner]":
    """Decorator to create an ExperimentRunner from a function.

    The decorated function becomes an experiment where the return value
    is stored as exp.result on the experiment instance.

    Args:
        name: Experiment name for the output folder. Defaults to function name.
        output_dir: Base directory for experiment results. Defaults to "out" directory
            relative to the file containing the experiment function.
        executor: Default execution strategy for running experiments.
        retry: Number of times to retry a failed experiment.
        viewer: If True, start the viewer after experiments complete.
        viewer_port: Port for the viewer.
        stash: If True, capture git repository state.

    The decorated function can have one of two signatures:
        - fn(cfg) - receives only the config
        - fn(cfg, out) - receives config and output directory path

    Example usage:

        @pyexp.experiment
        def my_experiment(cfg):
            return {"accuracy": train(cfg.lr)}

        # With output directory access:
        @pyexp.experiment
        def my_experiment(cfg, out):
            logger = Logger(out)
            # ... training with logging ...
            return {"accuracy": 0.95}

        @my_experiment.configs
        def configs():
            return [{"name": "exp", "lr": 0.01}]

        @my_experiment.report
        def report(results, out):
            for exp in results:
                print(f"{exp.name}: {exp.result['accuracy']}")

        my_experiment.run()
    """

    def make_runner(f: Callable[[dict], Any]) -> ExperimentRunner:
        import inspect

        # Create a dynamic Experiment subclass that wraps the function
        class DynamicExperiment(_DecoratorExperiment):
            pass

        # Store the function on the class (as regular class attribute, not staticmethod)
        DynamicExperiment._fn = f

        # Check if function wants 'out' parameter (2+ parameters means cfg + out)
        sig = inspect.signature(f)
        DynamicExperiment._wants_out = len(sig.parameters) >= 2

        # Determine output directory relative to function's file
        resolved_output_dir = output_dir
        if resolved_output_dir is None:
            fn_file = f.__globals__.get("__file__")
            if fn_file:
                resolved_output_dir = Path(fn_file).parent / "out"

        # Create runner with the dynamic class
        runner = ExperimentRunner(
            DynamicExperiment,
            name=name or f.__name__,
            output_dir=resolved_output_dir,
            executor=executor,
            retry=retry,
            viewer=viewer,
            viewer_port=viewer_port,
            stash=stash,
        )

        # Copy function metadata to runner for nice repr
        wraps(f)(runner)
        return runner

    if fn is not None:
        # Called without arguments: @experiment
        return make_runner(fn)
    else:
        # Called with arguments: @experiment(name="mnist", executor="fork")
        return make_runner
