from functools import wraps
from pathlib import Path
from typing import Callable, Any
import argparse
import hashlib
import json
import pickle


class Config(dict):
    """A dictionary that supports dot notation access for keys.

    Nested dictionaries are automatically converted to Config objects.

    Example:
        config = Config({"optimizer": {"learning_rate": 0.01}})
        config.optimizer.learning_rate  # 0.01
        config["optimizer"]["learning_rate"]  # also works
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Config):
                super().__setitem__(key, Config(value))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")


def _deep_copy_dict(d: dict) -> dict:
    """Create a deep copy of a dict, recursively copying nested dicts."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        else:
            result[k] = v
    return result


def merge(base: dict, update: dict) -> dict:
    """Merge update into base dict, supporting dot-notation for nested access.

    This function does NOT deep merge nested dicts - setting a dict value
    replaces the entire dict. Use dot-notation to update individual nested keys.

    Args:
        base: Base dictionary to merge into.
        update: Dictionary with updates. Keys can use dot-notation for nested access.

    Returns:
        New dictionary with updates applied (base is not mutated).

    Example:
        base = {"mlp": {"width": 32, "depth": 2}}
        merge(base, {"mlp.width": 64})  # {"mlp": {"width": 64, "depth": 2}}
        merge(base, {"mlp": {"width": 64}})  # {"mlp": {"width": 64}} - depth is gone!
    """
    result = _deep_copy_dict(base)

    for key, value in update.items():
        if "." in key:
            # Dot notation: navigate to nested location
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Regular key: direct replacement
            result[key] = value

    return result


class Tensor:
    """A tensor-like container that tracks shape across sweeps.

    Stores data in a flattened list while tracking the shape from successive
    sweep operations. Supports advanced indexing to select elements along dimensions.
    Used for both configs and results to enable matching access patterns.

    Example:
        configs = Tensor([{"name": "exp"}])  # shape: (1,)
        configs = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])  # shape: (1, 2)
        configs = sweep(configs, [{"epochs": 10}, {"epochs": 20}])  # shape: (1, 2, 2)

        # Index along dimensions
        configs[:, 0, :]  # All configs with first lr value
        configs[:, :, 1]  # All configs with second epochs value
    """

    def __init__(self, data: list, shape: tuple[int, ...] | None = None):
        """Create a Tensor.

        Args:
            data: List of items (flattened).
            shape: Shape tuple. If None, inferred as (len(data),).
        """
        self._data = list(data)
        self._shape = shape if shape is not None else (len(data),)
        expected_size = 1
        for dim in self._shape:
            expected_size *= dim
        if expected_size != len(self._data):
            raise ValueError(f"Shape {self._shape} does not match data length {len(self._data)}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the config tensor."""
        return self._shape

    def __len__(self) -> int:
        """Return total number of configs."""
        return len(self._data)

    def __iter__(self):
        """Iterate over all configs (flattened)."""
        return iter(self._data)

    def __getitem__(self, key):
        """Advanced indexing to select elements along dimensions.

        Supports integers, slices, and tuples of these.
        - Single integer: flat index into data (for backwards compatibility)
        - Tuple: multi-dimensional indexing
        Returns a new Tensor or single element.
        """
        # Single integer = flat index for backwards compatibility
        if isinstance(key, int):
            if key < 0:
                key = len(self._data) + key
            return self._data[key]

        if not isinstance(key, tuple):
            key = (key,)

        # Pad key with slices for remaining dimensions
        key = key + (slice(None),) * (len(self._shape) - len(key))

        if len(key) != len(self._shape):
            raise IndexError(f"Too many indices: got {len(key)}, shape has {len(self._shape)} dimensions")

        # Convert each key element to a list of indices
        index_lists = []
        new_shape = []
        for k, dim_size in zip(key, self._shape):
            if isinstance(k, int):
                if k < 0:
                    k = dim_size + k
                if k < 0 or k >= dim_size:
                    raise IndexError(f"Index {k} out of range for dimension of size {dim_size}")
                index_lists.append([k])
                # Integer index collapses dimension (not added to new_shape)
            elif isinstance(k, slice):
                indices = list(range(*k.indices(dim_size)))
                index_lists.append(indices)
                new_shape.append(len(indices))
            else:
                raise TypeError(f"Invalid index type: {type(k)}")

        # Generate all combinations of indices
        selected_data = []
        self._select_recursive(index_lists, 0, [], selected_data)

        # If all dimensions collapsed, return single config
        if not new_shape:
            return self._data[self._flat_index(tuple(il[0] for il in index_lists))]

        return Tensor(selected_data, tuple(new_shape))

    def _select_recursive(self, index_lists: list[list[int]], dim: int, current: list[int], result: list):
        """Recursively select configs based on index lists."""
        if dim == len(index_lists):
            result.append(self._data[self._flat_index(tuple(current))])
            return
        for idx in index_lists[dim]:
            self._select_recursive(index_lists, dim + 1, current + [idx], result)

    def _flat_index(self, indices: tuple[int, ...]) -> int:
        """Convert multi-dimensional indices to flat index."""
        flat = 0
        stride = 1
        for i in range(len(indices) - 1, -1, -1):
            flat += indices[i] * stride
            stride *= self._shape[i]
        return flat

    def tolist(self) -> list:
        """Return data as a flat list."""
        return list(self._data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self._shape}, len={len(self._data)})"


def sweep(configs: list[dict] | Tensor, variations: list[dict]) -> Tensor:
    """Generate cartesian product of configs and parameter variations.

    Each config is combined with each variation using merge(), which supports
    dot-notation for nested key access. Returns a Tensor with an additional
    dimension for the variations.

    Args:
        configs: Base configurations to expand (list or Tensor).
        variations: List of parameter variations to sweep over.
            Keys can use dot-notation for nested access (e.g., "mlp.width").

    Returns:
        Tensor with shape (*old_shape, len(variations)).

    Example:
        configs = [{"name": "exp", "mlp": {"width": 32}}]
        configs = sweep(configs, [{"mlp.width": 64}, {"mlp.width": 128}])
        # Returns Tensor with shape (1, 2), configs have mlp.width updated
    """
    if isinstance(configs, Tensor):
        old_shape = configs.shape
        old_data = configs._data
    else:
        old_shape = (len(configs),)
        old_data = list(configs)

    result = []
    for config in old_data:
        for variation in variations:
            result.append(merge(config, variation))

    new_shape = old_shape + (len(variations),)
    return Tensor(result, new_shape)


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

        @runner.experiment
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
