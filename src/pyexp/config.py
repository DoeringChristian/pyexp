"""Configuration utilities: Config, Runs, merge, sweep, load_config, registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Generic, Iterator, TypeVar

import yaml

# Global registry for class instantiation from config
_registry: dict[str, type] = {}

T = TypeVar("T")
_T = TypeVar("_T")


def register(cls: type[T]) -> type[T]:
    """Decorator to register a class in the global registry.

    Allows classes to be instantiated by name from configuration files.
    The class name becomes the key in the registry.

    Args:
        cls: Class to register

    Returns:
        The same class (allows use as a decorator)

    Raises:
        RuntimeError: If a class with the same name is already registered

    Example:
        @register
        class MyModel:
            def __init__(self, hidden_size: int = 256):
                self.hidden_size = hidden_size

        # In YAML config:
        # model:
        #   type: MyModel
        #   hidden_size: 512

        config = load_config("config.yaml")
        model = build(MyModel, config["model"])
    """
    if cls.__name__ in _registry:
        raise RuntimeError(
            f"Class with name '{cls.__name__}' already exists in registry!"
        )
    _registry[cls.__name__] = cls
    return cls


def build(tp: type[T], cfg: dict | T | None, *args, **kwargs) -> T:
    """Instantiate an object from configuration using the registry.

    If cfg is a dict with a 'type' key, looks up the class in the registry
    and instantiates it with the remaining config parameters. Otherwise,
    assumes cfg is already an instance and returns it directly.

    Args:
        tp: Expected type of the returned object (for type checking)
        cfg: Configuration dict with 'type' key, or an existing instance
        *args: Additional positional arguments to pass to constructor
        **kwargs: Additional keyword arguments to pass to constructor

    Returns:
        Instance of the specified type

    Raises:
        TypeError: If the returned object is not of type tp
        KeyError: If the type specified in cfg is not in the registry

    Example:
        @register
        class Optimizer:
            def __init__(self, lr: float = 0.001):
                self.lr = lr

        # From dict config
        opt = build(Optimizer, {"type": "Optimizer", "lr": 0.01})

        # Pass existing instance (returned as-is)
        opt2 = build(Optimizer, opt)

        # With additional kwargs (override config)
        opt3 = build(Optimizer, {"type": "Optimizer"}, lr=0.1)
    """
    if cfg is None:
        raise ValueError("Cannot build from None config")

    if isinstance(cfg, dict):
        if "type" not in cfg:
            raise KeyError("Config dict must have a 'type' key to build from registry")
        type_name = cfg["type"]
        if type_name not in _registry:
            available = ", ".join(_registry.keys()) if _registry else "(none)"
            raise KeyError(
                f"Type '{type_name}' not in registry. Available: {available}"
            )
        cls = _registry[type_name]
        # Merge config values with kwargs (kwargs take precedence)
        for k, v in cfg.items():
            if k == "type":
                continue
            if k not in kwargs:
                kwargs[k] = v
        obj = cls(*args, **kwargs)
    else:
        obj = cfg

    if not isinstance(obj, tp):
        raise TypeError(f"Expected type {tp.__name__}, got {type(obj).__name__}")
    return obj


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

    def __repr__(self) -> str:
        return yaml.dump(
            to_dict(self), default_flow_style=False, sort_keys=False
        ).rstrip("\n")


def to_dict(cfg: Config) -> dict:
    """Convert a Config to a plain dict recursively."""
    return {k: to_dict(v) if isinstance(v, Config) else v for k, v in cfg.items()}


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

    Prefix a key with ** to deep-merge a dict value into the existing dict
    at that location instead of replacing it. The merge is one level deep:
    each key in the ** value is set individually, but nested dicts within
    it still replace (not merge) their targets.

    Args:
        base: Base dictionary to merge into.
        update: Dictionary with updates. Keys can use dot-notation for nested access.
            Keys prefixed with ** will deep-merge their dict value.

    Returns:
        New dictionary with updates applied (base is not mutated).

    Example:
        base = {"mlp": {"width": 32, "depth": 2}}
        merge(base, {"mlp.width": 64})  # {"mlp": {"width": 64, "depth": 2}}
        merge(base, {"mlp": {"width": 64}})  # {"mlp": {"width": 64}} - depth is gone!
        merge(base, {"**mlp": {"width": 64}})  # {"mlp": {"width": 64, "depth": 2}}
    """
    result = _deep_copy_dict(base)

    for key, value in update.items():
        # ** prefix: deep-merge the dict value into the target
        if key.startswith("**"):
            key = key[2:]
            if not isinstance(value, dict):
                raise ValueError(
                    f"**{key} requires a dict value, got {type(value).__name__}"
                )
            # Expand into dot-notation updates and recurse
            flat = {}
            for k, v in value.items():
                flat[f"{key}.{k}" if key else k] = v
            result = merge(result, flat)
        elif "." in key:
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


class Runs(Generic[_T]):
    """A flat, named collection of items (configs or experiment results).

    Supports lookup by integer index, name pattern (regex via re.search), or
    dict-based filtering on item attributes.

    Example:
        configs = Runs([{"name": "exp_a", "lr": 0.1}, {"name": "exp_b", "lr": 0.01}])

        # Integer index
        configs[0]  # {"name": "exp_a", "lr": 0.1}

        # Pattern matching on name (regex via re.search)
        configs["exp_.*"]  # Runs with all matching items
        configs["exp_a"]  # Single item if exactly one match

        # Dict matching on values
        configs[{"lr": 0.1}]  # All items where lr == 0.1
        configs[{"mlp.width": 32}]  # Dot notation for nested keys
    """

    def __init__(self, data: list[_T]):
        """Create a Runs collection.

        Args:
            data: List of items.
        """
        self._data = list(data)

    def __len__(self) -> int:
        """Return total number of items."""
        return len(self._data)

    def __iter__(self) -> Iterator[_T]:
        """Iterate over all items."""
        return iter(self._data)

    def __getitem__(self, key: int | str | dict) -> "_T | Runs[_T]":
        """Look up items by index, name pattern, or dict filter.

        - int: flat index into data
        - str: pattern match on item name (regex via re.search)
        - dict: match items where all key-value pairs match
        Returns a single item or a new Runs collection.
        """
        if isinstance(key, int):
            if key < 0:
                key = len(self._data) + key
            return self._data[key]

        if isinstance(key, str):
            return self._match_pattern(key)

        if isinstance(key, dict):
            return self._match_dict(key)

        raise TypeError(f"Invalid index type: {type(key)}")

    def _match_pattern(self, pattern: str) -> "_T | Runs[_T]":
        """Match items by name pattern (regex via re.search).

        Returns a single item if exactly one match, otherwise a new Runs.
        """
        matching = []
        for item in self._data:
            if isinstance(item, dict):
                name = item.get("name", "")
            else:
                name = getattr(item, "name", "") or ""
            if re.search(pattern, name):
                matching.append(item)

        if not matching:
            raise IndexError(f"No configs match pattern '{pattern}'")

        if len(matching) == 1:
            return matching[0]

        return Runs(matching)

    def _match_dict(self, query: dict) -> "_T | Runs[_T]":
        """Match items by key-value pairs.

        Returns a single item if exactly one match, otherwise a new Runs.
        Supports dot-notation keys for attribute/dict access.
        """
        matching = []
        for item in self._data:
            if self._item_matches(item, query):
                matching.append(item)

        if not matching:
            raise IndexError(f"No configs match query {query}")

        if len(matching) == 1:
            return matching[0]

        return Runs(matching)

    def _item_matches(self, item: Any, query: dict) -> bool:
        """Check if item matches all key-value pairs in query.

        Supports both dict items and objects with attributes.
        Dot notation navigates through nested dicts or object attributes.
        """
        for key, expected in query.items():
            try:
                value = self._get_nested_value(item, key)
            except (KeyError, AttributeError, TypeError):
                return False

            # Compare values
            if isinstance(expected, dict) and isinstance(value, dict):
                # Recursive dict matching
                if not self._item_matches(value, expected):
                    return False
            elif value != expected:
                return False

        return True

    def _get_nested_value(self, item: Any, key: str) -> Any:
        """Get a nested value from an item using dot notation.

        Works with dicts (using []) and objects (using getattr).
        """
        if "." in key:
            parts = key.split(".")
        else:
            parts = [key]

        value = item
        for part in parts:
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)
        return value

    def tolist(self) -> list[_T]:
        """Return data as a flat list."""
        return list(self._data)

    def __repr__(self) -> str:
        return f"Runs(len={len(self._data)})"


def sweep(configs: list[dict] | Runs, variations: list[dict]) -> Runs:
    """Generate cartesian product of configs and parameter variations.

    Each config is combined with each variation using merge(), which supports
    dot-notation for nested key access. Returns a Runs collection containing
    all combinations.

    The "name" key is handled specially: names are combined with underscore
    separator (e.g., "exp" + "lr0.1" -> "exp_lr0.1"). This enables pattern-based
    selection: configs["exp_lr0.1_*"] matches all configs with that prefix.

    Args:
        configs: Base configurations to expand (list or Runs).
        variations: List of parameter variations to sweep over.
            Keys can use dot-notation for nested access (e.g., "mlp.width").
            Each variation should have a "name" key for pattern matching.

    Returns:
        Runs collection with all combinations.

    Example:
        configs = [{"name": "exp", "mlp": {"width": 32}}]
        configs = sweep(configs, [{"name": "w64", "mlp.width": 64}, {"name": "w128", "mlp.width": 128}])
        # Returns Runs(len=2)
        # Config names become "exp_w64", "exp_w128"
        # Access via: configs["exp_w64"] or configs["exp_*"]
    """
    if isinstance(configs, Runs):
        old_data = configs._data
    else:
        old_data = list(configs)

    result = []
    for config in old_data:
        base_name = config.get("name", "")
        for variation in variations:
            merged = merge(config, variation)
            # Combine names with underscore
            var_name = variation.get("name", "")
            if base_name and var_name:
                merged["name"] = f"{base_name}_{var_name}"
            elif var_name:
                merged["name"] = var_name
            # else keep base_name or no name
            result.append(merged)

    return Runs(result)


def load_config(paths: list[Path] | Path | str | list[str]) -> Config:
    """Load configuration files with support for imports (composition).

    Uses merge() for all merging, which supports:
    - Regular keys: complete replacement
    - Dotted keys: nested update preserving siblings

    Equivalent to a config file with `imports: [*paths]`. Each path is loaded
    recursively (processing its own imports field), then merged in order.

    Args:
        paths: Single path or list of paths to load and merge in order.
               Can be Path objects or strings.

    Returns:
        Merged configuration as a Config object.

    Example:
        # base.yaml
        model:
          hidden_size: 256
          num_layers: 4

        # experiment.yaml
        imports:
          - base.yaml
        model.hidden_size: 512  # Override using dot notation
        learning_rate: 0.001

        # Load single file (with imports resolved)
        config = load_config("experiment.yaml")

        # Load and merge multiple files
        config = load_config(["base.yaml", "overrides.yaml"])
    """
    # Handle single path for convenience
    if isinstance(paths, (str, Path)):
        paths = [paths]

    # Convert strings to Paths
    paths = [Path(p) if isinstance(p, str) else p for p in paths]

    if not paths:
        return Config()

    # Start with empty result
    result: dict = {}

    # Process each path in order (like imports: [*paths])
    for path in paths:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}

        # Recursively process imports field in this config
        if "imports" in cfg:
            import_paths = cfg.pop("imports")
            if isinstance(import_paths, str):
                import_paths = [import_paths]
            # Resolve paths relative to the config file's directory
            resolved_paths = [path.parent / p for p in import_paths]
            imported = load_config(resolved_paths)
            result = merge(result, dict(imported))

        # Merge this config (without imports key)
        result = merge(result, cfg)

    return Config(result)
