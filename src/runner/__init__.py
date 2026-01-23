from functools import wraps
from typing import Callable, Any


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

    def run(self) -> Any:
        """Execute the full pipeline: configs -> experiments -> report."""
        if self._configs_fn is None:
            raise RuntimeError("No configs function registered. Use @runner.configs decorator.")
        if self._experiment_fn is None:
            raise RuntimeError("No experiment function registered. Use @runner.experiment decorator.")
        if self._report_fn is None:
            raise RuntimeError("No report function registered. Use @runner.report decorator.")

        configs = self._configs_fn()
        results = []
        for config in configs:
            result = self._experiment_fn(config)
            results.append(result)

        return self._report_fn(configs, results)


# Default runner instance for simple usage
_default_runner = Runner()

configs = _default_runner.configs
experiment = _default_runner.experiment
report = _default_runner.report
run = _default_runner.run
