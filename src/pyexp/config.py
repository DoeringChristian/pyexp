"""Configuration utilities: Config, Tensor, merge, sweep."""

from typing import Any


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
