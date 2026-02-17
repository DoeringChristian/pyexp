"""ExperimentRunner: stripped-down execution engine for running experiments."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, TYPE_CHECKING
import hashlib
import inspect
import json
import pickle
import re

from .config import Config, Runs
from .executors import Executor, ExecutorName, get_executor

if TYPE_CHECKING:
    from .log import Logger


@dataclass
class Result:
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
) -> Runs[Result]:
    """Load all experiments for a batch into a 1D Runs.

    Discovers runs by scanning the filesystem for directories containing
    <timestamp>/config.json.

    Args:
        base_dir: The experiment base directory.
        timestamp: The batch timestamp.
        finished_only: If True, only load experiments that have a .finished marker.

    Returns:
        1D Runs of Result instances.
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


def _write_run_dir(experiment_dir, config, commit_hash):
    """Create the experiment run directory, write config.json, and optionally .commit.

    Args:
        experiment_dir: Path to the experiment run directory.
        config: Config object to serialize as config.json.
        commit_hash: Optional git commit hash to write as .commit.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_json_path = experiment_dir / "config.json"
    config_json_path.write_text(
        json.dumps(dict(config), indent=2, default=str)
    )
    if commit_hash is not None:
        (experiment_dir / ".commit").write_text(commit_hash)


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


class ExperimentRunner:
    """Stripped-down execution engine for running experiments.

    Handles: directory setup, config validation, execution loop, retry,
    caching, dependencies, progress bar.

    Usage:
        runner = ExperimentRunner(name="my_exp", output_dir="out", retry=4)
        runner.submit(train_fn, {"name": "fast", "lr": 0.01})
        runner.run()
    """

    def __init__(
        self,
        *,
        name: str = "experiment",
        output_dir: str | Path | None = None,
        retry: int = 4,
    ):
        self._name = name
        self._output_dir = Path(output_dir) if output_dir else Path("out")
        self._retry = retry
        self._submissions: list[tuple[Callable, dict]] = []

    def submit(self, fn: Callable, cfg: dict) -> None:
        """Submit an experiment function with a specific config."""
        self._submissions.append((fn, cfg))

    def run(
        self,
        *,
        executor: ExecutorName | Executor | str = "subprocess",
        capture: bool = True,
        stash: bool = True,
        snapshot_path: Path | None = None,
        hash_configs: bool = False,
        filter: str | None = None,
        continue_run: str | None = None,
    ) -> None:
        """Execute all submitted experiments.

        Args:
            executor: Execution strategy for running experiments.
            capture: If True, capture output and show progress bar.
            stash: If True, capture git repository state.
            snapshot_path: Pre-existing snapshot path to use.
            hash_configs: Append config parameter hash to run directory names.
            filter: Regex pattern to filter configs by name.
            continue_run: Timestamp of a previous run to continue.
        """
        if not self._submissions:
            raise RuntimeError(
                "No experiments submitted. Use runner.submit(fn, cfg) first."
            )

        # Resolve executor
        if isinstance(executor, Executor):
            exec_instance = executor
        elif isinstance(executor, str) and (
            executor.startswith("ray://")
            or (executor.startswith("ray:") and not executor.startswith("ray://"))
        ):
            from .executors import RayExecutor

            address = executor if executor.startswith("ray://") else executor[4:]
            exec_instance = RayExecutor(
                address=address,
                runtime_env={"working_dir": "."},
            )
        else:
            exec_instance = get_executor(executor)

        base_dir = self._output_dir / self._name
        max_retries = self._retry

        # Track snapshot state
        commit_hash: str | None = None

        # Determine the timestamp
        if continue_run:
            if continue_run == "latest":
                latest = _get_latest_timestamp(base_dir)
                if latest is None:
                    raise RuntimeError(f"No previous runs found in {base_dir}")
                timestamp = latest
            else:
                timestamp = continue_run
                if not _discover_experiment_dirs(base_dir, timestamp):
                    raise RuntimeError(f"Run not found: {timestamp}")

            # Detect existing snapshot
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
            timestamp = _generate_timestamp()

        # Print run info
        print(f"Run: {self._name}/{timestamp}")

        # =================================================================
        # Config Generation (only on fresh start)
        # =================================================================
        is_fresh_start = not continue_run

        if is_fresh_start:
            flat_configs = [cfg for _, cfg in self._submissions]
            _validate_configs(flat_configs)
            _validate_unique_names(flat_configs)
            _validate_dependencies(flat_configs)
            flat_configs = _topological_sort(flat_configs)

            if hash_configs:
                _validate_unique_hashes(flat_configs)

            experiment_dirs = [
                _get_experiment_dir(
                    config, base_dir, timestamp, hash_configs=hash_configs
                )
                for config in flat_configs
            ]

            run_dirs = []
            for config in flat_configs:
                cfg_name = _sanitize_name(config.get("name", "experiment"))
                if hash_configs:
                    run_dirs.append(f"{cfg_name}-{_config_hash(config)}")
                else:
                    run_dirs.append(cfg_name)

            configs_for_save = []
            for config in flat_configs:
                assert (
                    "out" not in config
                ), "Config cannot contain 'out' key; it is reserved"
                configs_for_save.append(Config(config))

            base_dir.mkdir(parents=True, exist_ok=True)

            if stash:
                try:
                    from .utils import stash_and_snapshot
                    import tempfile

                    with tempfile.TemporaryDirectory() as tmp:
                        commit_hash, tmp_snapshot = stash_and_snapshot(
                            Path(tmp) / "src"
                        )

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

            _save_batch_manifest(base_dir, timestamp, run_dirs, commit=commit_hash)

        # =================================================================
        # Experiment Execution
        # =================================================================

        # Build per-config function lookup from submissions
        submission_fns: dict[str, Callable] = {}
        for fn_sub, cfg_sub in self._submissions:
            sub_name = cfg_sub.get("name", "")
            submission_fns[sub_name] = fn_sub

        if is_fresh_start:
            # Use in-memory data from Phase A (no filesystem discovery needed)
            all_configs_runs = Runs(flat_configs)

            # Build mapping from dir path to Config object for deferred writing
            dir_to_config_obj: dict[str, Config] = {
                str(d): c for d, c in zip(experiment_dirs, configs_for_save)
            }

            all_experiment_dirs = list(experiment_dirs)

            # Apply filter using in-memory config names
            if filter:
                dir_to_flat = {
                    str(d): fc for d, fc in zip(experiment_dirs, flat_configs)
                }

                def get_config_name_mem(exp_dir: Path) -> str:
                    fc = dir_to_flat.get(str(exp_dir))
                    return fc.get("name", "") if fc else ""

                original_count = len(experiment_dirs)
                experiment_dirs = _filter_by_name(
                    experiment_dirs, filter, get_config_name_mem
                )
                if not experiment_dirs:
                    print(f"No configs match filter '{filter}'")
                    return None
                print(
                    f"Filter '{filter}': {len(experiment_dirs)}/{original_count} configs selected"
                )
        else:
            # Continue run: discover from filesystem (existing logic)
            experiment_dirs = _discover_experiment_dirs(base_dir, timestamp)

            # Topologically sort discovered dirs by depends_on from config.json
            dir_configs = []
            for d in experiment_dirs:
                cfg = json.loads((d / "config.json").read_text())
                dir_configs.append(cfg)
            if any(c.get("depends_on") for c in dir_configs):
                sorted_configs = _topological_sort(dir_configs)
                name_to_dir = {
                    c.get("name", ""): d
                    for c, d in zip(dir_configs, experiment_dirs)
                }
                experiment_dirs = [
                    name_to_dir[c["name"]] for c in sorted_configs
                ]

            all_experiment_dirs = list(experiment_dirs)

            # Apply filter if specified
            if filter:

                def get_config_name(exp_dir: Path) -> str:
                    config_path = exp_dir / "config.json"
                    if config_path.exists():
                        config = json.loads(config_path.read_text())
                        return config.get("name", "")
                    return ""

                original_count = len(experiment_dirs)
                experiment_dirs = _filter_by_name(
                    experiment_dirs, filter, get_config_name
                )
                if not experiment_dirs:
                    print(f"No configs match filter '{filter}'")
                    return None
                print(
                    f"Filter '{filter}': {len(experiment_dirs)}/{original_count} configs selected"
                )

            all_dir_configs = []
            for exp_dir in all_experiment_dirs:
                config_json_path = exp_dir / "config.json"
                all_dir_configs.append(json.loads(config_json_path.read_text()))
            all_configs_runs = Runs(all_dir_configs)

        show_progress = capture
        progress = _ProgressBar(len(experiment_dirs)) if show_progress else None

        completed: dict[str, Result] = {}

        if filter:
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

            if is_fresh_start:
                # Use in-memory config data
                config_obj = dir_to_config_obj[str(experiment_dir)]
                config_data = dict(config_obj)
            else:
                # Read config from filesystem (continue path)
                config_json_path = experiment_dir / "config.json"
                config_data = json.loads(config_json_path.read_text())

            config_name = config_data.get("name", "")
            dep_keys = _normalize_depends_on(config_data.get("depends_on"))

            resolved_deps = (
                _resolve_depends_on(dep_keys, all_configs_runs, config_name)
                if dep_keys
                else []
            )

            config_data_clean = {
                k: v for k, v in config_data.items() if k != "depends_on"
            }
            config = Config(config_data_clean)

            fn = submission_fns.get(config_name)

            # Detect signature for wants_out/wants_deps per-submission
            if fn is not None:
                sig = inspect.signature(fn)
                n_params = len(sig.parameters)
                wants_out = n_params >= 2
                wants_deps = n_params >= 3
            else:
                wants_out = False
                wants_deps = False

            finished_marker = experiment_dir / ".finished"
            is_cached = experiment_path.exists() and finished_marker.exists()

            if is_cached:
                with open(experiment_path, "rb") as f:
                    exp_obj = pickle.load(f)
            if is_cached:
                completed[config_name] = exp_obj
                marker_path = experiment_dir / ".pyexp"
                marker_path.touch(exist_ok=True)
                status = "cached"
            else:
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
                        skip_reason = f"Dependency '{dep_name}' failed or was skipped"
                        break

                if should_skip:
                    exp_obj = Result(
                        cfg=config,
                        name=config_name,
                        out=experiment_dir,
                        skipped=True,
                        error=f"Skipped: {skip_reason}",
                        finished=True,
                    )
                    if is_fresh_start:
                        _write_run_dir(
                            experiment_dir,
                            dir_to_config_obj[str(experiment_dir)],
                            commit_hash,
                        )
                    else:
                        experiment_dir.mkdir(parents=True, exist_ok=True)
                    with open(experiment_path, "wb") as f:
                        pickle.dump(exp_obj, f)
                    (experiment_dir / ".finished").touch()
                    completed[config_name] = exp_obj
                    status = "skipped"
                else:
                    if progress:
                        progress.start(config_name)

                    dep_experiments = [completed[n] for n in resolved_deps]
                    deps_runs = Runs(dep_experiments)

                    exp_obj = Result(
                        cfg=config,
                        name=config_name,
                        out=experiment_dir,
                    )

                    if is_fresh_start:
                        _write_run_dir(
                            experiment_dir,
                            dir_to_config_obj[str(experiment_dir)],
                            commit_hash,
                        )
                    else:
                        experiment_dir.mkdir(parents=True, exist_ok=True)
                    with open(experiment_path, "wb") as f:
                        pickle.dump(exp_obj, f)

                    for attempt in range(max_retries + 1):
                        exec_instance.run(
                            fn,
                            exp_obj,
                            deps_runs,
                            experiment_path,
                            capture=capture,
                            stash=stash,
                            snapshot_path=snapshot_path,
                            wants_out=wants_out,
                            wants_deps=wants_deps,
                        )

                        with open(experiment_path, "rb") as f:
                            exp_obj = pickle.load(f)

                        if not exp_obj.error:
                            break

                        error_msg = exp_obj.error or ""
                        remaining = max_retries - attempt
                        retry_info = (
                            f" (retrying, {remaining} left)" if remaining > 0 else ""
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
                            exp_obj = Result(
                                cfg=config,
                                name=config_name,
                                out=experiment_dir,
                            )
                            with open(experiment_path, "wb") as f:
                                pickle.dump(exp_obj, f)

                    completed[config_name] = exp_obj

                    log_path = experiment_dir / "log.out"
                    log_path.write_text(exp_obj.log or "")
                    status = "failed" if exp_obj.error else "passed"

            if progress:
                progress.update(status, config_name)

        if progress:
            progress.finish()
