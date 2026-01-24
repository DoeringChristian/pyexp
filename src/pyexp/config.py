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

        # Index by name (if names are set)
        configs["exp", "lr0.1", "epochs10"]
    """

    def __init__(
        self,
        data: list,
        shape: tuple[int, ...] | None = None,
        names: list[list[str]] | None = None,
    ):
        """Create a Tensor.

        Args:
            data: List of items (flattened).
            shape: Shape tuple. If None, inferred as (len(data),).
            names: List of name lists, one per dimension. Each name list has
                   length equal to that dimension's size. Used for name-based indexing.
        """
        self._data = list(data)
        self._shape = shape if shape is not None else (len(data),)
        self._names = names  # List of name lists per dimension
        expected_size = 1
        for dim in self._shape:
            expected_size *= dim
        if expected_size != len(self._data):
            raise ValueError(f"Shape {self._shape} does not match data length {len(self._data)}")
        if names is not None:
            if len(names) != len(self._shape):
                raise ValueError(f"Names has {len(names)} dimensions but shape has {len(self._shape)}")
            for i, (name_list, dim_size) in enumerate(zip(names, self._shape)):
                if len(name_list) != dim_size:
                    raise ValueError(f"Names dimension {i} has {len(name_list)} names but shape has {dim_size}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the config tensor."""
        return self._shape

    @property
    def names(self) -> list[list[str]] | None:
        """Return the name mappings per dimension, or None if not set."""
        return self._names

    def __len__(self) -> int:
        """Return total number of configs."""
        return len(self._data)

    def __iter__(self):
        """Iterate over all configs (flattened)."""
        return iter(self._data)

    def _name_to_index(self, name: str, dim: int) -> int:
        """Convert a name to an index for the given dimension."""
        if self._names is None:
            raise IndexError("Cannot use string indexing: Tensor has no name mappings")
        try:
            return self._names[dim].index(name)
        except ValueError:
            raise IndexError(f"Name '{name}' not found in dimension {dim}. Available: {self._names[dim]}")

    def __getitem__(self, key):
        """Advanced indexing to select elements along dimensions.

        Supports integers, slices, strings (name lookup), and tuples of these.
        - Single integer: flat index into data (for backwards compatibility)
        - String: name-based lookup (requires names to be set)
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
        new_names = [] if self._names is not None else None
        for dim, (k, dim_size) in enumerate(zip(key, self._shape)):
            if isinstance(k, str):
                # Name-based indexing
                idx = self._name_to_index(k, dim)
                index_lists.append([idx])
                # String index collapses dimension (not added to new_shape)
            elif isinstance(k, int):
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
                if new_names is not None:
                    new_names.append([self._names[dim][i] for i in indices])
            else:
                raise TypeError(f"Invalid index type: {type(k)}")

        # Generate all combinations of indices
        selected_data = []
        self._select_recursive(index_lists, 0, [], selected_data)

        # If all dimensions collapsed, return single config
        if not new_shape:
            return self._data[self._flat_index(tuple(il[0] for il in index_lists))]

        return Tensor(selected_data, tuple(new_shape), new_names if new_names else None)

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

    The "name" key is handled specially:
    - Names from variations become the name list for the new dimension
    - Config names are combined with underscore separator (e.g., "exp_lr0.1")
    - This enables name-based indexing: configs["exp", "lr0.1", "epochs10"]

    Args:
        configs: Base configurations to expand (list or Tensor).
        variations: List of parameter variations to sweep over.
            Keys can use dot-notation for nested access (e.g., "mlp.width").
            Each variation should have a "name" key for name-based indexing.

    Returns:
        Tensor with shape (*old_shape, len(variations)).

    Example:
        configs = [{"name": "exp", "mlp": {"width": 32}}]
        configs = sweep(configs, [{"name": "w64", "mlp.width": 64}, {"name": "w128", "mlp.width": 128}])
        # Returns Tensor with shape (1, 2), names = [["exp"], ["w64", "w128"]]
        # Config names become "exp_w64", "exp_w128"
    """
    if isinstance(configs, Tensor):
        old_shape = configs.shape
        old_data = configs._data
        old_names = configs._names
    else:
        old_shape = (len(configs),)
        old_data = list(configs)
        old_names = None

    # Extract names from the first dimension if not already tracked
    # Only do this for lists (new sweep chains) or single-dim Tensors
    if old_names is None and len(old_shape) == 1:
        # First sweep: extract names from base configs
        first_dim_names = [c.get("name", f"cfg{i}") for i, c in enumerate(old_data)]
        old_names = [first_dim_names]

    # Extract names from variations for the new dimension (only if tracking names)
    new_dim_names = [v.get("name", f"var{i}") for i, v in enumerate(variations)] if old_names is not None else None

    result = []
    for config in old_data:
        base_name = config.get("name", "")
        for variation in variations:
            merged = merge(config, variation)
            # Combine names with underscore (only if tracking names)
            if old_names is not None:
                var_name = variation.get("name", "")
                if base_name and var_name:
                    merged["name"] = f"{base_name}_{var_name}"
                elif var_name:
                    merged["name"] = var_name
                # else keep base_name or no name
            result.append(merged)

    new_shape = old_shape + (len(variations),)
    new_names = old_names + [new_dim_names] if old_names is not None else None
    return Tensor(result, new_shape, new_names)
