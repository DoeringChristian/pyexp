"""Experiment runner: Experiment dataclass, ExperimentRunner, and decorators."""

from dataclasses import dataclass, field
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

from .config import Config, Runs
from .executors import Executor, ExecutorName, get_executor
from .log import LogReader

if TYPE_CHECKING:
    from .log import Logger


@dataclass
class Experiment:
    """Result dataclass for a single experiment run.

    Fields:
        cfg: The config for this run (Config object with dot notation access).
        name: Shorthand for cfg.get("name", "").
        out: Output directory for this experiment run.
        result: The return value of the experiment function.
        error: Error message if experiment failed, None otherwise.
        log: Captured stdout/stderr from the experiment run.
        finished: Whether this experiment has finished executing.
        skipped: Whether this experiment was skipped due to a failed dependency.
    """

    cfg: Config
    name: str
    out: Path
    result: Any = None
    error: str | None = None
    log: str = ""
    finished: bool = False
    skipped: bool = False


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


def _config_hash(config: dict) -> str:
    """Generate a short hash of the config for directory naming.

    Hashes all config values except 'name', so configs with the same
    parameters but different names map to the same directory.
    """
    config_without_meta = {
        k: v for k, v in config.items() if k not in ("name", "depends_on")
    }
    config_str = json.dumps(config_without_meta, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def _validate_unique_names(configs: list[dict]) -> None:
    """Validate all configs have unique non-empty 'name' fields.

    Args:
        configs: List of config dicts (after sanitization).

    Raises:
        ValueError: If any config has an empty name or names are not unique.
    """
    names = []
    for i, config in enumerate(configs):
        name = config.get("name", "")
        if not name:
            raise ValueError(
                f"Config at index {i} has no 'name' field. All configs must have a unique non-empty 'name'."
            )
        names.append(name)
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"Duplicate config name '{name}'. All configs must have unique names."
            )
        seen.add(name)


def _validate_unique_hashes(configs: list[dict]) -> None:
    """Validate all configs produce unique hashes (unique parameter sets).

    Two configs with different names but identical parameters would collide
    on disk since the directory is based on the config hash (excluding name).

    Raises:
        ValueError: If any two configs produce the same hash.
    """
    seen: dict[str, str] = {}  # hash -> first config name
    for config in configs:
        h = _config_hash(config)
        name = config.get("name", "unnamed")
        if h in seen:
            raise ValueError(
                f"Configs '{seen[h]}' and '{name}' have the same parameters (hash {h}). "
                f"Each config must have unique parameters."
            )
        seen[h] = name


def _normalize_depends_on(deps: Any) -> list[str | dict]:
    """Normalize a depends_on value to a list of keys (str or dict)."""
    if deps is None:
        return []
    if isinstance(deps, (str, dict)):
        return [deps]
    return list(deps)


def _resolve_depends_on(
    depends_on: list[str | dict], configs_runs: Runs, config_name: str
) -> list[str]:
    """Resolve depends_on keys to concrete config names using Runs indexing.

    Each key in depends_on uses the same lookup mechanism as Runs.__getitem__:
    - str: glob-style pattern matching on name (e.g., "pretrain*")
    - dict: key-value matching on config fields (e.g., {"stage": "pretrain"})

    Raises:
        ValueError: If a key matches no configs, or matches the config itself.
    """
    resolved: list[str] = []
    for key in depends_on:
        try:
            result = configs_runs[key]
        except (IndexError, TypeError):
            raise ValueError(
                f"Config '{config_name}' depends on {key!r}, "
                f"which matches no config in this batch."
            )
        # Normalize to list
        if isinstance(result, Runs):
            matches = list(result)
        else:
            matches = [result]

        for item in matches:
            name = (
                item.get("name", "")
                if isinstance(item, dict)
                else getattr(item, "name", "")
            )
            if name == config_name:
                raise ValueError(
                    f"Config '{config_name}' depends on itself (via {key!r})."
                )
            if name not in resolved:
                resolved.append(name)
    return resolved


def _validate_dependencies(configs: list[dict]) -> None:
    """Validate depends_on references and detect cycles.

    depends_on values use the same indexing as Runs (glob patterns or dict queries).

    Raises:
        ValueError: If a dependency key matches no config,
                    a config depends on itself, or there is a cycle.
    """
    configs_runs = Runs(configs)

    # Build adjacency for cycle detection: name -> list of resolved dependency names
    adj: dict[str, list[str]] = {}
    for config in configs:
        config_name = config["name"]
        deps = _normalize_depends_on(config.get("depends_on"))
        resolved = _resolve_depends_on(deps, configs_runs, config_name)
        adj[config_name] = resolved

    # Cycle detection via DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in adj}

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        path.append(node)
        for neighbor in adj.get(node, []):
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                raise ValueError(f"Dependency cycle detected: {' -> '.join(cycle)}")
            if color[neighbor] == WHITE:
                dfs(neighbor, path)
        path.pop()
        color[node] = BLACK

    for name in adj:
        if color[name] == WHITE:
            dfs(name, [])


def _topological_sort(configs: list[dict]) -> list[dict]:
    """Return configs in dependency order (Kahn's algorithm).

    depends_on keys are resolved via Runs indexing.
    Configs with no dependencies come first.
    """
    configs_runs = Runs(configs)
    name_to_config = {c["name"]: c for c in configs}
    in_degree: dict[str, int] = {c["name"]: 0 for c in configs}
    dependents: dict[str, list[str]] = {c["name"]: [] for c in configs}

    for config in configs:
        deps = _normalize_depends_on(config.get("depends_on"))
        resolved = _resolve_depends_on(deps, configs_runs, config["name"])
        in_degree[config["name"]] = len(resolved)
        for dep in resolved:
            dependents[dep].append(config["name"])

    # Start with nodes that have no dependencies
    queue = [name for name, deg in in_degree.items() if deg == 0]
    result: list[dict] = []

    while queue:
        node = queue.pop(0)
        result.append(name_to_config[node])
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(configs):
        raise ValueError("Dependency cycle detected (topological sort failed).")

    return result


def _get_experiment_dir(
    config: dict, base_dir: Path, timestamp: str, *, hash_configs: bool = False
) -> Path:
    """Get the experiment directory path for a config.

    Returns base_dir / <name> / timestamp (default) or
    base_dir / <name>-<config_hash> / timestamp (with hash_configs=True).
    """
    name = _sanitize_name(config.get("name", "experiment"))
    if hash_configs:
        return base_dir / f"{name}-{_config_hash(config)}" / timestamp
    return base_dir / name / timestamp


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


def _save_batch_manifest(
    base_dir: Path, timestamp: str, run_dirs: list[str], commit: str | None = None
) -> None:
    """Save batch manifest to .batches/<timestamp>.json.

    Args:
        base_dir: The experiment base directory (e.g., out/experiment_name).
        timestamp: The timestamp string for this batch.
        run_dirs: List of run directory names for each run.
        commit: Optional git commit hash of the source snapshot.
    """
    batches_dir = base_dir / ".batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": timestamp,
        "runs": run_dirs,
    }
    if commit is not None:
        data["commit"] = commit

    manifest_path = batches_dir / f"{timestamp}.json"
    manifest_path.write_text(json.dumps(data, indent=2, default=str))


def _load_batch_manifest(base_dir: Path, timestamp: str) -> dict:
    """Load a batch manifest from .batches/<timestamp>.json.

    Returns:
        Dict with keys: timestamp, runs, and optionally commit.
    """
    manifest_path = base_dir / ".batches" / f"{timestamp}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No batch manifest found at {manifest_path}")
    return json.loads(manifest_path.read_text())


def _discover_experiment_dirs(base_dir: Path, timestamp: str) -> list[Path]:
    """Discover experiment run directories by scanning the filesystem.

    Looks for subdirectories of base_dir that contain a <timestamp>/config.json.
    Skips dot-directories (.batches, .snapshots, etc.).

    Returns:
        List of experiment directories (base_dir/<run_name>/<timestamp>),
        sorted alphabetically by run name.
    """
    if not base_dir.exists():
        return []
    dirs = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        experiment_dir = entry / timestamp
        if (experiment_dir / "config.json").exists():
            dirs.append(experiment_dir)
    return dirs


def _get_all_timestamps(base_dir: Path) -> list[str]:
    """Return all batch timestamps sorted newest-first by scanning run directories."""
    if not base_dir.exists():
        return []
    timestamps: set[str] = set()
    for entry in base_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        for ts_dir in entry.iterdir():
            if ts_dir.is_dir() and (ts_dir / "config.json").exists():
                timestamps.add(ts_dir.name)
    return sorted(timestamps, reverse=True)


def _get_latest_timestamp(base_dir: Path) -> str | None:
    """Find the most recent batch timestamp by scanning run directories.

    Looks at all timestamp subdirectories across all run directories and
    returns the lexicographically latest one.
    """
    timestamps = _get_all_timestamps(base_dir)
    return timestamps[0] if timestamps else None


def _is_batch_finished(base_dir: Path, timestamp: str) -> bool:
    """Check if all runs in a batch have finished (have .finished marker)."""
    experiment_dirs = _discover_experiment_dirs(base_dir, timestamp)
    if not experiment_dirs:
        return False
    return all((d / ".finished").exists() for d in experiment_dirs)


def _get_latest_finished_timestamp(base_dir: Path) -> str | None:
    """Find the most recent batch timestamp where all runs are finished."""
    for ts in _get_all_timestamps(base_dir):
        if _is_batch_finished(base_dir, ts):
            return ts
    return None


def _load_experiments(
    base_dir: Path, timestamp: str, *, finished_only: bool = False
) -> Runs[Experiment]:
    """Load all experiments for a batch into a 1D Runs.

    Discovers runs by scanning the filesystem for directories containing
    <timestamp>/config.json.

    Args:
        base_dir: The experiment base directory.
        timestamp: The batch timestamp.
        finished_only: If True, only load experiments that have a .finished marker.

    Returns:
        1D Runs of Experiment instances.
    """
    experiment_dirs = _discover_experiment_dirs(base_dir, timestamp)

    results = []
    for experiment_dir in experiment_dirs:
        experiment_path = experiment_dir / "experiment.pkl"

        if finished_only and not (experiment_dir / ".finished").exists():
            continue

        if experiment_path.exists():
            with open(experiment_path, "rb") as f:
                instance = pickle.load(f)
            results.append(instance)

    return Runs(results)


def _generate_timestamp() -> str:
    """Generate a timestamp string for the current time."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _list_runs(base_dir: Path) -> None:
    """List all runs under the experiment base directory with their status."""
    # Discover all timestamps by scanning run directories
    timestamps: set[str] = set()
    if base_dir.exists():
        for entry in base_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            for ts_dir in entry.iterdir():
                if ts_dir.is_dir() and (ts_dir / "config.json").exists():
                    timestamps.add(ts_dir.name)

    if not timestamps:
        print("No runs found.")
        return

    for timestamp in sorted(timestamps):
        experiment_dirs = _discover_experiment_dirs(base_dir, timestamp)
        total = len(experiment_dirs)
        completed = 0
        failed = 0
        for experiment_dir in experiment_dirs:
            finished_marker = experiment_dir / ".finished"
            experiment_path = experiment_dir / "experiment.pkl"
            if finished_marker.exists() and experiment_path.exists():
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

        print(f"  {timestamp}  {status}  ({total} configs)")


def _print_dependency_graph(configs: list[dict]) -> None:
    """Print the dependency graph for a list of configs."""
    configs_runs = Runs(configs)
    for config in configs:
        name = config["name"]
        deps = _normalize_depends_on(config.get("depends_on"))
        if deps:
            resolved = _resolve_depends_on(deps, configs_runs, name)
            print(f"  {name} <- {', '.join(resolved)}")
        else:
            print(f"  {name}")


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


class _ProgressBar:
    """Simple progress bar for experiment execution."""

    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.current = 0
        self.passed = 0
        self.failed = 0
        self.cached = 0
        self.skipped = 0
        self._render()  # Show initial state

    def start(self, name: str = ""):
        """Show that an experiment is starting."""
        self._render(name, running=True)

    def update(self, status: str, name: str = ""):
        """Update progress with status: 'passed', 'failed', 'cached', or 'skipped'."""
        self.current += 1
        if status == "passed":
            self.passed += 1
        elif status == "failed":
            self.failed += 1
        elif status == "cached":
            self.cached += 1
        elif status == "skipped":
            self.skipped += 1
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
        if self.skipped:
            parts.append(f"\033[90m{self.skipped} skipped\033[0m")
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
    """Runner for executing experiments defined as plain functions.

    This class handles:
    - CLI argument parsing
    - Config generation and validation
    - Experiment execution via executors
    - Result caching and loading
    - Report generation

    Decorator-based:
        @pyexp.experiment
        def my_experiment(cfg):
            return {"accuracy": train(cfg)}

        my_experiment.run()  # ExperimentRunner is created internally

    Programmatic:
        runner = ExperimentRunner(name="my_exp")
        runner.submit(train_fn, {"name": "fast", "lr": 0.01})
        runner.run()
    """

    def __init__(
        self,
        *,
        name: str = "experiment",
        output_dir: str | Path | None = None,
        executor: ExecutorName | Executor | str = "subprocess",
        retry: int = 4,
        viewer: bool = False,
        viewer_port: int = 8765,
        stash: bool = True,
        hash_configs: bool = False,
    ):
        """Create an ExperimentRunner.

        Args:
            name: Experiment name for output folder.
            output_dir: Base directory for results. Defaults to "out" relative to caller.
            executor: Execution strategy ("subprocess", "fork", "inline", "ray", etc.)
            retry: Number of retries on failure.
            viewer: Start viewer after experiments complete.
            viewer_port: Port for the viewer.
            stash: Capture git repository state.
            hash_configs: Append config parameter hash to run directory names.
        """
        self._name = name

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
        self._hash_configs_default = hash_configs
        self._configs_fn: Callable[[], list[dict]] | None = None
        self._report_fn: Callable[[Runs, Path], Any] | None = None
        self._fn: Callable | None = None
        self._wants_out: bool = False
        self._wants_deps: bool = False
        self._submissions: list[tuple[Callable, dict]] = []

    def submit(self, fn: Callable, cfg: dict) -> None:
        """Submit an experiment function with a specific config.

        Args:
            fn: The experiment function to run.
            cfg: The config dict for this experiment.
        """
        self._submissions.append((fn, cfg))

    def configs(self, fn: Callable[[], list[dict]]) -> Callable[[], list[dict]]:
        """Decorator to register the configs generator function (for decorator API)."""
        self._configs_fn = fn
        return fn

    def report(self, fn: Callable[[Runs, Path], Any]) -> Callable[[Runs, Path], Any]:
        """Decorator to register the report function (for decorator API)."""
        self._report_fn = fn
        return fn

    def results(
        self,
        timestamp: str | None = None,
        output_dir: str | Path | None = None,
        name: str | None = None,
        run: str | None = None,
        finished: bool = False,
    ) -> Runs[Experiment]:
        """Load results from a previous experiment run.

        Args:
            timestamp: Timestamp of the run to load (e.g., "2026-01-28_10-30-00").
                      If None or "latest", loads the most recent run.
            output_dir: Base directory where experiment results are stored.
            name: Experiment name. Defaults to class/function name.
            run: Optional single run name to load (returns 1D Runs with one element).
            finished: If True, only include finished experiments. When searching
                     for "latest", searches backwards in time until it finds a
                     batch where all runs are finished.

        Returns:
            1D Runs of Experiment instances with full attribute access.
        """
        exp_name = name or self._name
        resolved_output_dir = (
            Path(output_dir) if output_dir else self._output_dir_default
        )
        base_dir = resolved_output_dir / exp_name

        if timestamp is None or timestamp == "latest":
            if finished:
                latest = _get_latest_finished_timestamp(base_dir)
            else:
                latest = _get_latest_timestamp(base_dir)
            if latest is None:
                return Runs([])
            timestamp = latest

        # Verify that at least one run directory exists for this timestamp
        experiment_dirs = _discover_experiment_dirs(base_dir, timestamp)
        if not experiment_dirs:
            raise FileNotFoundError(f"Run not found: {timestamp}")

        return _load_experiments(base_dir, timestamp, finished_only=finished)

    def run(
        self,
        configs: Callable[[], list[dict]] | None = None,
        report: Callable[[Runs, Path], Any] | None = None,
        output_dir: str | Path | None = None,
        executor: ExecutorName | Executor | str | None = None,
        name: str | None = None,
        retry: int | None = None,
        viewer: bool | None = None,
        viewer_port: int | None = None,
        stash: bool | None = None,
        hash_configs: bool | None = None,
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
            hash_configs: Append config parameter hash to run directory names.
        """
        args = _parse_args()

        # Resolve parameters with CLI override priority:
        # CLI args > run() args > constructor args
        resolved_executor = args.executor or executor or self._executor_default
        exp_name = args.name or name or self._name

        # Resolve configs function: run() arg > decorated
        if configs is not None:
            configs_fn = configs
        elif self._configs_fn is not None:
            configs_fn = self._configs_fn
        else:
            configs_fn = None

        # Resolve report function: run() arg > decorated
        if report is not None:
            report_fn = report
        elif self._report_fn is not None:
            report_fn = self._report_fn
        else:
            report_fn = None

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

        # Hash configs: run() arg > constructor arg
        if hash_configs is not None:
            enable_hash_configs = hash_configs
        else:
            enable_hash_configs = self._hash_configs_default

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

        # Build submissions-based configs function if submissions exist
        if self._submissions and configs_fn is None:
            submission_configs = [cfg for _, cfg in self._submissions]
            configs_fn = lambda: submission_configs

        if configs_fn is None:
            raise RuntimeError(
                "No configs function provided. "
                "Use @runner.configs decorator, pass configs= argument, or use runner.submit()."
            )
        # report_fn is optional — if None, skip report phase

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

        # --report implies --continue (use report's timestamp or latest)
        if args.report is not None and not args.continue_run:
            args.continue_run = args.report

        # Track snapshot state
        snapshot_path: Path | None = None
        commit_hash: str | None = None

        # Determine the timestamp
        if args.continue_run:
            if args.continue_run == "latest":
                latest = _get_latest_timestamp(base_dir)
                if latest is None:
                    raise RuntimeError(f"No previous runs found in {base_dir}")
                timestamp = latest
            else:
                timestamp = args.continue_run
                if not _discover_experiment_dirs(base_dir, timestamp):
                    raise RuntimeError(f"Run not found: {timestamp}")

            # Detect existing snapshot from a .commit file in any run directory
            for exp_dir in _discover_experiment_dirs(base_dir, timestamp):
                commit_file = exp_dir / ".commit"
                if commit_file.exists():
                    prev_commit = commit_file.read_text().strip()
                    prev_snapshot = base_dir / ".snapshots" / prev_commit
                    if prev_snapshot.exists():
                        snapshot_path = prev_snapshot
                        commit_hash = prev_commit
                    break
        else:
            # Generate new timestamp
            timestamp = _generate_timestamp()

        # Print run info
        print(f"Run: {exp_name}/{timestamp}")

        # Start viewer in background if requested
        viewer_process = None
        if start_viewer:
            import subprocess as sp
            from pyexp._viewer import _find_free_port

            base_dir.mkdir(parents=True, exist_ok=True)
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
                env={**os.environ, "PYEXP_LOG_DIR": str(base_dir.absolute())},
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )

        # =================================================================
        # PHASE 1: Config Generation (only on fresh start)
        # =================================================================
        is_fresh_start = not args.continue_run and not args.report

        if is_fresh_start:
            config_list = configs_fn()
            _validate_configs(list(config_list))

            # Flatten config_list for iteration
            flat_configs = list(config_list)

            # Validate unique names (after flattening)
            _validate_unique_names(flat_configs)

            # Validate dependencies and topologically sort
            _validate_dependencies(flat_configs)
            flat_configs = _topological_sort(flat_configs)

            # Validate unique hashes when hashing is enabled
            if enable_hash_configs:
                _validate_unique_hashes(flat_configs)

            # Compute experiment directories
            experiment_dirs = [
                _get_experiment_dir(
                    config, base_dir, timestamp, hash_configs=enable_hash_configs
                )
                for config in flat_configs
            ]

            # Collect run directory names for the batch manifest
            run_dirs = []
            for config in flat_configs:
                name = _sanitize_name(config.get("name", "experiment"))
                if enable_hash_configs:
                    run_dirs.append(f"{name}-{_config_hash(config)}")
                else:
                    run_dirs.append(name)

            # Build configs (without 'out')
            configs_for_save = []
            for config in flat_configs:
                assert (
                    "out" not in config
                ), "Config cannot contain 'out' key; it is reserved"
                configs_for_save.append(Config(config))

            # Create base directory
            base_dir.mkdir(parents=True, exist_ok=True)

            # Create shared source snapshot if stash is enabled
            if enable_stash:
                try:
                    from .utils import stash_and_snapshot

                    # Use a temporary path first to get the commit hash
                    import tempfile

                    with tempfile.TemporaryDirectory() as tmp:
                        commit_hash, tmp_snapshot = stash_and_snapshot(
                            Path(tmp) / "src"
                        )

                    # Now create the shared snapshot at .snapshots/<commit>/
                    shared_snapshot = base_dir / ".snapshots" / commit_hash
                    if not shared_snapshot.exists():
                        _, snapshot_path = stash_and_snapshot(shared_snapshot)
                    else:
                        snapshot_path = shared_snapshot

                    print(f"Source snapshot: {commit_hash[:12]} -> {snapshot_path}")
                except Exception as e:
                    print(f"Warning: Could not create source snapshot: {e}")
                    snapshot_path = None
                    commit_hash = None

            # Save batch manifest
            _save_batch_manifest(base_dir, timestamp, run_dirs, commit=commit_hash)

            # Create all experiment directories and save individual config.json files
            for config, experiment_dir in zip(configs_for_save, experiment_dirs):
                experiment_dir.mkdir(parents=True, exist_ok=True)
                config_json_path = experiment_dir / "config.json"
                config_json_path.write_text(
                    json.dumps(dict(config), indent=2, default=str)
                )
                # Write commit reference so each run is self-contained
                if commit_hash is not None:
                    (experiment_dir / ".commit").write_text(commit_hash)

        # =================================================================
        # PHASE 2: Experiment Execution (skip if --report)
        # =================================================================
        if not args.report:

            # Build per-config function lookup from submissions
            submission_fns: dict[str, Callable] = {}
            if self._submissions:
                for fn_sub, cfg_sub in self._submissions:
                    sub_name = cfg_sub.get("name", "")
                    submission_fns[sub_name] = fn_sub

            # Discover experiment directories from filesystem
            experiment_dirs = _discover_experiment_dirs(base_dir, timestamp)

            # Topologically sort discovered dirs by depends_on from config.json
            dir_configs = []
            for d in experiment_dirs:
                cfg = json.loads((d / "config.json").read_text())
                dir_configs.append(cfg)
            if any(c.get("depends_on") for c in dir_configs):
                sorted_configs = _topological_sort(dir_configs)
                name_to_dir = {
                    c.get("name", ""): d for c, d in zip(dir_configs, experiment_dirs)
                }
                experiment_dirs = [name_to_dir[c["name"]] for c in sorted_configs]

            # Save full (unfiltered) list for dependency resolution
            all_experiment_dirs = list(experiment_dirs)

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

            # Build Runs of all configs (unfiltered) for dependency resolution
            all_dir_configs = []
            for exp_dir in all_experiment_dirs:
                config_json_path = exp_dir / "config.json"
                all_dir_configs.append(json.loads(config_json_path.read_text()))
            all_configs_runs = Runs(all_dir_configs)

            # Track completed experiments by name for dependency injection
            completed: dict[str, Experiment] = {}

            # Pre-load finished experiments from filtered-out dirs for dependency injection
            if args.filter:
                filtered_set = set(str(d) for d in experiment_dirs)
                for exp_dir in all_experiment_dirs:
                    if str(exp_dir) in filtered_set:
                        continue
                    finished_marker = exp_dir / ".finished"
                    exp_pkl = exp_dir / "experiment.pkl"
                    if finished_marker.exists() and exp_pkl.exists():
                        cfg_data = json.loads((exp_dir / "config.json").read_text())
                        dep_name = cfg_data.get("name", "")
                        with open(exp_pkl, "rb") as f:
                            completed[dep_name] = pickle.load(f)

            for experiment_dir in experiment_dirs:
                experiment_path = experiment_dir / "experiment.pkl"

                # Load config from saved config.json
                config_json_path = experiment_dir / "config.json"
                config_data = json.loads(config_json_path.read_text())
                config_name = config_data.get("name", "")
                dep_keys = _normalize_depends_on(config_data.get("depends_on"))

                # Resolve dependency keys to concrete names via Runs indexing
                resolved_deps = (
                    _resolve_depends_on(dep_keys, all_configs_runs, config_name)
                    if dep_keys
                    else []
                )

                # Strip depends_on before creating Config object
                config_data.pop("depends_on", None)
                config = Config(config_data)

                # Determine the function for this config
                fn = submission_fns.get(config_name, self._fn)

                # Check if a finished result already exists on disk
                finished_marker = experiment_dir / ".finished"
                is_cached = experiment_path.exists() and finished_marker.exists()

                if is_cached:
                    with open(experiment_path, "rb") as f:
                        exp_obj = pickle.load(f)
                if is_cached:
                    # Already completed (cached / --continue)
                    completed[config_name] = exp_obj
                    # Ensure marker file exists for viewer discovery
                    marker_path = experiment_dir / ".pyexp"
                    marker_path.touch(exist_ok=True)
                    status = "cached"
                else:
                    # Check if any dependency failed or was skipped
                    should_skip = False
                    skip_reason = None
                    for dep_name in resolved_deps:
                        dep_exp = completed.get(dep_name)
                        if dep_exp is None:
                            should_skip = True
                            skip_reason = f"Dependency '{dep_name}' was not found"
                            break
                        if dep_exp.skipped or dep_exp.error:
                            should_skip = True
                            skip_reason = (
                                f"Dependency '{dep_name}' failed or was skipped"
                            )
                            break

                    if should_skip:
                        # Create skipped experiment
                        exp_obj = Experiment(
                            cfg=config,
                            name=config_name,
                            out=experiment_dir,
                            skipped=True,
                            error=f"Skipped: {skip_reason}",
                            finished=True,
                        )
                        experiment_dir.mkdir(parents=True, exist_ok=True)
                        with open(experiment_path, "wb") as f:
                            pickle.dump(exp_obj, f)
                        (experiment_dir / ".finished").touch()
                        completed[config_name] = exp_obj
                        status = "skipped"
                    else:
                        # Show running status
                        if progress:
                            progress.start(config_name)

                        # Build deps Runs from completed experiments
                        dep_experiments = [completed[n] for n in resolved_deps]
                        deps_runs = Runs(dep_experiments)

                        # Create fresh Experiment dataclass
                        exp_obj = Experiment(
                            cfg=config,
                            name=config_name,
                            out=experiment_dir,
                        )

                        # Save initial experiment.pkl so partial/interrupted
                        # experiments are always discoverable on disk
                        experiment_dir.mkdir(parents=True, exist_ok=True)
                        with open(experiment_path, "wb") as f:
                            pickle.dump(exp_obj, f)

                        # Retry loop
                        for attempt in range(max_retries + 1):
                            exec_instance.run(
                                fn,
                                exp_obj,
                                deps_runs,
                                experiment_path,
                                capture=not args.no_capture,
                                stash=enable_stash,
                                snapshot_path=snapshot_path,
                                wants_out=self._wants_out,
                                wants_deps=self._wants_deps,
                            )

                            # Reload experiment to check result
                            with open(experiment_path, "rb") as f:
                                exp_obj = pickle.load(f)

                            if not exp_obj.error:
                                break  # Success, exit retry loop

                            # Print error immediately on each failure
                            error_msg = exp_obj.error or ""
                            remaining = max_retries - attempt
                            retry_info = (
                                f" (retrying, {remaining} left)"
                                if remaining > 0
                                else ""
                            )
                            print(
                                f"\n--- Error in {config_name or 'experiment'}{retry_info} ---"
                            )
                            print(error_msg)
                            if exp_obj.log:
                                print("--- Log output ---")
                                print(exp_obj.log)
                            print("---")

                            if attempt < max_retries:
                                # Create fresh experiment for retry
                                exp_obj = Experiment(
                                    cfg=config,
                                    name=config_name,
                                    out=experiment_dir,
                                )
                                # Re-save initial (unfinished) pkl for retry
                                with open(experiment_path, "wb") as f:
                                    pickle.dump(exp_obj, f)

                        completed[config_name] = exp_obj

                        # Save log to plaintext file
                        log_path = experiment_dir / "log.out"
                        log_path.write_text(exp_obj.log or "")
                        status = "failed" if exp_obj.error else "passed"

                # Update progress bar
                if progress:
                    progress.update(status, config_name)

            # Finish progress bar
            if progress:
                progress.finish()

        # =================================================================
        # PHASE 3: Report Generation (optional)
        # =================================================================
        if report_fn is not None:
            results = _load_experiments(base_dir, timestamp)

            report_dir = base_dir / "report" / timestamp
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
    hash_configs: bool = False,
) -> "ExperimentRunner | Callable[[Callable[[dict], Any]], ExperimentRunner]":
    """Decorator to create an ExperimentRunner from a function.

    The decorated function becomes an experiment where the return value
    is stored as exp.result on the Experiment dataclass.

    Args:
        name: Experiment name for the output folder. Defaults to function name.
        output_dir: Base directory for experiment results. Defaults to "out" directory
            relative to the file containing the experiment function.
        executor: Default execution strategy for running experiments.
        retry: Number of times to retry a failed experiment.
        viewer: If True, start the viewer after experiments complete.
        viewer_port: Port for the viewer.
        stash: If True, capture git repository state.
        hash_configs: Append config parameter hash to run directory names.

    The decorated function can have one of three signatures:
        - fn(cfg) - receives only the config
        - fn(cfg, out) - receives config and output directory path
        - fn(cfg, out, deps) - receives config, output directory, and dependency runs

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

        # Check if function wants 'out' and/or 'deps' parameter
        sig = inspect.signature(f)
        n_params = len(sig.parameters)
        wants_out = n_params >= 2
        wants_deps = n_params >= 3

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

        # Create runner directly
        runner = ExperimentRunner(
            name=resolved_name,
            output_dir=resolved_output_dir,
            executor=executor,
            retry=retry,
            viewer=viewer,
            viewer_port=viewer_port,
            stash=stash,
            hash_configs=hash_configs,
        )

        # Store the function and signature info on the runner
        runner._fn = f
        runner._wants_out = wants_out
        runner._wants_deps = wants_deps

        # Copy function metadata to runner for nice repr
        wraps(f)(runner)
        return runner

    if fn is not None:
        # Called without arguments: @experiment
        return make_runner(fn)
    else:
        # Called with arguments: @experiment(name="mnist", executor="fork")
        return make_runner
