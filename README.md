# pyexp

A lightweight Python library for running experiments with automatic caching, configuration management, parameter sweeps, robust execution, and reporting.

## Installation

```bash
pip install git+https://github.com/username/pyexp.git

# With Ray support for distributed execution
pip install "git+https://github.com/username/pyexp.git#egg=pyexp[ray]"
```

## Quick Start

```python
import pyexp
from pyexp import Config

@pyexp.experiment(name="mnist")
def experiment(config: Config):
    lr = config.learning_rate
    epochs = config.epochs
    # Run your experiment...
    return {"accuracy": 0.95}

@experiment.configs
def configs():
    return [
        {"name": "fast", "learning_rate": 0.01, "epochs": 10},
        {"name": "slow", "learning_rate": 0.001, "epochs": 100},
    ]

@experiment.report
def report(results):
    # Each result contains 'name', 'config', and experiment outputs
    for r in results:
        print(f"{r['name']}: {r['accuracy']}")

    # Filter by config values
    fast_results = results[{"config.learning_rate": 0.01}]

if __name__ == "__main__":
    experiment.run()
```

## Features

### Automatic Caching

Experiment results are automatically cached. Re-running the script skips completed experiments:

```bash
python main.py                              # Runs all experiments (new timestamp)
python main.py                              # New run (different timestamp)
python main.py --timestamp 2024-01-25_14-30-00  # Continue specific run
python main.py --timestamp 2024-01-25_14-30-00 --rerun  # Rerun specific run
python main.py --timestamp 2024-01-25_14-30-00 --report # Report from specific run
```

### Output Folder Structure

Results are organized by experiment name and optional timestamp:

```
out/
  <experiment_name>/
    <timestamp>/                    # When timestamp=True (default)
      <config_name>-<hash>/
        result.pkl
        model.pt                    # Your saved artifacts
      <config_name>-<hash>/
        ...
    <timestamp>/                    # Another run
      ...
```

Control the structure with `name` and `timestamp` parameters:

```python
# Default: uses function name, with timestamp
@pyexp.experiment
def my_experiment(config): ...
# -> out/my_experiment/2024-01-25_14-30-00/<config>-<hash>/

# Custom name, no timestamp (overwrites previous runs)
@pyexp.experiment(name="mnist", timestamp=False)
def my_experiment(config): ...
# -> out/mnist/<config>-<hash>/

# Override at runtime
experiment.run(name="experiment_v2", timestamp=False)
```

### Robust Execution

By default, each experiment runs in an isolated subprocess, protecting against crashes (including segfaults). Failed experiments return error information without stopping other experiments:

```python
@experiment.report
def report(results):
    for r in results:
        if r.get("__error__"):
            print(f"{r['name']} failed: {r['type']}: {r['message']}")
        else:
            print(f"{r['name']}: {r['accuracy']}")
```

### Execution Modes

Choose how experiments are executed using the `executor` parameter:

```python
# In decorator (sets default)
@pyexp.experiment(executor="subprocess")  # Default: isolated subprocess
@pyexp.experiment(executor="fork")        # Unix only: fast, same module state
@pyexp.experiment(executor="inline")      # No isolation, useful for debugging
@pyexp.experiment(executor="ray")         # Distributed execution with Ray

# Or override at runtime
experiment.run(executor="inline")  # Debug mode
```

| Executor | Platform | Isolation | Use Case |
|----------|----------|-----------|----------|
| `"subprocess"` | All | Strong | Default, crash-safe, cross-platform |
| `"fork"` | Unix | Strong | Fast startup, identical module state |
| `"inline"` | All | None | Debugging, quick iteration |
| `"ray"` | All | Strong | Distributed/multi-machine execution |

### Distributed Execution with Ray

For cluster execution, use `RayExecutor` with configuration:

```python
from pyexp import RayExecutor

# Local Ray execution
experiment.run(executor="ray")

# Connect to cluster with code sync
executor = RayExecutor(
    address="auto",  # Or "ray://cluster-head:10001"
    runtime_env={
        "working_dir": ".",            # Sync current directory to workers
        "excludes": ["data/", "*.pt"], # Don't upload large files
        "pip": ["pandas", "numpy"],    # Install packages on workers
    },
    num_cpus=4,
    num_gpus=1,
)

experiment.run(executor=executor)
```

### Config Class

Configs support both dict-style and dot notation access:

```python
config["learning_rate"]  # dict-style
config.learning_rate     # dot notation
config.optimizer.lr      # nested access
```

### Output Directory

Each experiment receives an `out` in its config pointing to its cache directory. Use this to save artifacts:

```python
@pyexp.experiment
def experiment(config: Config):
    model = train(config)
    torch.save(model, config.out / "model.pt")
    return {"accuracy": 0.95}
```

### Parameter Sweeps

Use `sweep()` to generate cartesian products of configurations:

```python
from pyexp import sweep

@experiment.configs
def configs():
    cfgs = [{"name": "exp", "base_lr": 0.01}]

    # Sweep over learning rates
    cfgs = sweep(cfgs, [
        {"name": "lr0.1", "learning_rate": 0.1},
        {"name": "lr0.01", "learning_rate": 0.01},
    ])

    # Sweep over epochs (creates 2x2 = 4 configs)
    cfgs = sweep(cfgs, [
        {"name": "e10", "epochs": 10},
        {"name": "e100", "epochs": 100},
    ])

    return cfgs  # Shape: (1, 2, 2), names: exp_lr0.1_e10, exp_lr0.1_e100, ...
```

Names are automatically combined with underscores across sweeps.

### Dot Notation in Sweeps

Use dot notation to update nested config values without replacing the entire dict:

```python
cfgs = [{"name": "exp", "mlp": {"width": 32, "depth": 2}}]
cfgs = sweep(cfgs, [
    {"name": "w64", "mlp.width": 64},   # Only updates width, keeps depth
    {"name": "w128", "mlp.width": 128},
])
```

### Result Filtering

The report function receives a `Tensor` of results. Each result contains:
- `name`: the combined config name
- `config`: the full config dict
- All experiment outputs

Filter results using pattern matching or dict queries:

```python
@experiment.report
def report(results):
    # Pattern matching on name (glob-style)
    lr01_results = results["exp_lr0.1_*"]  # All with lr0.1
    epoch10_results = results["*_e10"]      # All with e10

    # Dict matching on config values
    lr01_results = results[{"config.learning_rate": 0.1}]

    # Multiple constraints
    specific = results[{"config.learning_rate": 0.1, "config.epochs": 10}]

    # Nested config access
    wide_results = results[{"config.mlp.width": 128}]
```

### Tensor Indexing

Results preserve the shape from sweep operations:

```python
@experiment.report
def report(results):
    # Shape is (1, 2, 2) from sweeps
    print(results.shape)

    # Index by position
    first_lr = results[:, 0, :]  # All configs with first lr value

    # Access individual result
    r = results[0, 1, 0]
    print(r["name"], r["accuracy"])
```

### Custom Executors

Create custom execution strategies by subclassing `Executor`:

```python
from pyexp import Executor

class MyExecutor(Executor):
    def run(self, fn, config, result_path):
        # Custom execution logic
        result = fn(config)
        with open(result_path, "wb") as f:
            pickle.dump(result, f)
        return result

experiment.run(executor=MyExecutor())
```

### Alternative API

You can also pass configs and report functions directly:

```python
experiment.run(configs=my_configs_fn, report=my_report_fn)
```

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--report` | Skip experiments, only generate report from cache |
| `--rerun` | Re-run all experiments, ignore cache |
| `--timestamp TIMESTAMP` | Use specific timestamp folder (e.g., `2024-01-25_14-30-00`) to continue or rerun a previous run |

## Configuration

The output directory defaults to `out/` but can be customized:

```python
experiment.run(output_dir="results/")
```

## API Reference

### Functions

- `sweep(configs, variations)` - Generate cartesian product of configs with variations
- `merge(base, update)` - Merge dicts with dot-notation support
- `get_executor(name)` - Get executor by name or return instance

### Classes

- `Config` - Dict subclass with dot notation access
- `Tensor` - Shape-preserving container with advanced indexing
- `Experiment` - Experiment runner with caching
- `Executor` - Abstract base class for custom executors
- `InlineExecutor` - Runs in same process (no isolation)
- `SubprocessExecutor` - Runs in isolated subprocess (default)
- `ForkExecutor` - Runs in forked process (Unix only)
- `RayExecutor` - Runs with Ray for distributed execution

### Decorators

- `@pyexp.experiment` - Define an experiment function
- `@pyexp.experiment(name="...", executor="...", timestamp=True/False)` - Define with options
- `@experiment.configs` - Register configs generator
- `@experiment.report` - Register report function

### Types

- `ExecutorName` - Literal type: `"inline" | "subprocess" | "fork" | "ray"`
