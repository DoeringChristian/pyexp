from functools import wraps
from pathlib import Path
from typing import Callable, Any
import hashlib
import json
import pickle


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


class Runner:
    """Orchestrates experiment execution with configs, experiment, and report phases."""

    def __init__(self):
        self._configs_fn: Callable[[], list[dict]] | None = None
        self._experiment_fn: Callable[[dict], Any] | None = None
        self._report_fn: Callable[[list[dict], list[Any]], Any] | None = None

    def configs(self, fn: Callable[[], list[dict]]) -> Callable[[], list[dict]]:
        """Decorator to register the configs generator function."""
        @wraps(fn)
        def wrapper() -> list[dict]:
            return fn()
        self._configs_fn = wrapper
        return wrapper

    def experiment(self, fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
        """Decorator to register the experiment function."""
        @wraps(fn)
        def wrapper(config: dict) -> Any:
            return fn(config)
        self._experiment_fn = wrapper
        return wrapper

    def report(self, fn: Callable[[list[dict], list[Any]], Any]) -> Callable[[list[dict], list[Any]], Any]:
        """Decorator to register the report function."""
        @wraps(fn)
        def wrapper(configs: list[dict], results: list[Any]) -> Any:
            return fn(configs, results)
        self._report_fn = wrapper
        return wrapper

    def run(self, output_dir: str | Path = "out") -> Any:
        """Execute the full pipeline: configs -> experiments -> report.

        Args:
            output_dir: Directory for caching experiment results. Defaults to "out".
        """
        if self._configs_fn is None:
            raise RuntimeError("No configs function registered. Use @runner.configs decorator.")
        if self._experiment_fn is None:
            raise RuntimeError("No experiment function registered. Use @runner.experiment decorator.")
        if self._report_fn is None:
            raise RuntimeError("No report function registered. Use @runner.report decorator.")

        output_dir = Path(output_dir)
        configs = self._configs_fn()
        results = []

        for config in configs:
            assert "out_dir" not in config, "Config cannot contain 'out_dir' key; it is reserved"
            experiment_dir = _get_experiment_dir(config, output_dir)
            result_path = experiment_dir / "result.pkl"

            if result_path.exists():
                with open(result_path, "rb") as f:
                    result = pickle.load(f)
            else:
                experiment_dir.mkdir(parents=True, exist_ok=True)
                config_with_out = {**config, "out_dir": experiment_dir}
                result = self._experiment_fn(config_with_out)
                with open(result_path, "wb") as f:
                    pickle.dump(result, f)

            results.append(result)

        return self._report_fn(configs, results)


# Default runner instance for simple usage
_default_runner = Runner()

configs = _default_runner.configs
experiment = _default_runner.experiment
report = _default_runner.report
run = _default_runner.run
