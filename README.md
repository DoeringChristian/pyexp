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

Use `executor="ray"` for Ray-based execution:

```python
# Local Ray execution
experiment.run(executor="ray")

# Remote cluster execution (syncs working directory automatically)
experiment.run(executor="ray://cluster:10001")
experiment.run(executor="ray:auto")  # Connect to existing cluster

# Set defaults in decorator
@pyexp.experiment(executor="ray://cluster:10001")
def my_experiment(config):
    ...
```

For advanced configuration, pass a `RayExecutor` instance:

```python
from pyexp import RayExecutor

executor = RayExecutor(
    address="ray://cluster:10001",
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

### Loading Configs from YAML

Use `load_config()` to load configurations from YAML files with composition support:

```python
from pyexp import load_config

# Load a single config file
config = load_config("config.yaml")

# Load and merge multiple files (later files override earlier ones)
config = load_config(["base.yaml", "experiment.yaml", "overrides.yaml"])
```

**Composable configs with imports:**

Config files can import other configs using the `imports` field:

```yaml
# base.yaml
model:
  hidden_size: 256
  num_layers: 4
learning_rate: 0.001
```

```yaml
# experiment.yaml
imports:
  - base.yaml

# Override specific values (dot notation supported)
model.hidden_size: 512
batch_size: 32
```

```python
# Loading experiment.yaml automatically resolves imports
config = load_config("experiment.yaml")
# Result: {model: {hidden_size: 512, num_layers: 4}, learning_rate: 0.001, batch_size: 32}
```

**Using with experiments:**

```python
from pyexp import load_config, sweep

@experiment.configs
def configs():
    base = load_config("configs/base.yaml")
    return sweep([base], [
        {"name": "small", "model.hidden_size": 128},
        {"name": "large", "model.hidden_size": 1024},
    ])
```

### Registry and Build

Use `@register` and `build()` to instantiate classes from config:

```python
from pyexp import register, build

@register
class MLP:
    def __init__(self, hidden_size: int = 256, num_layers: int = 2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

@register
class Transformer:
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
```

```yaml
# config.yaml
model:
  type: MLP
  hidden_size: 512
  num_layers: 4
```

```python
config = load_config("config.yaml")
model = build(MLP, config["model"])  # Creates MLP(hidden_size=512, num_layers=4)

# Or use base class for polymorphism
from abc import ABC
class Model(ABC): pass

@register
class MLP(Model): ...

model = build(Model, config["model"])  # Type-checked against Model
```

**Features:**
- Pass existing instance (returned as-is with type check)
- Override config with kwargs: `build(MLP, cfg, hidden_size=1024)`
- Works with `Config` objects from `load_config()`

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

CLI arguments can override settings from the decorator or `run()` function.

**Priority:** CLI args > `run()` args > decorator args

| Argument | Description |
|----------|-------------|
| `--name NAME` | Override experiment name |
| `--executor EXECUTOR` | Override executor (`subprocess`, `fork`, `inline`, `ray`, `ray:<address>`) |
| `--output-dir DIR` | Override output directory |
| `--no-timestamp` | Disable timestamp folders |
| `--timestamp TIMESTAMP` | Use specific timestamp folder (e.g., `2024-01-25_14-30-00`) to continue or rerun a previous run |
| `--report` | Skip experiments, only generate report from cache |
| `--rerun` | Re-run all experiments, ignore cache |
| `-s`, `--capture=no` | Show subprocess output instead of progress bar |

By default, experiment output is captured and a progress bar is shown. Use `-s` to see live output from each experiment (similar to pytest).

```bash
# Override executor from command line
python main.py --executor inline

# Run without timestamps
python main.py --no-timestamp

# Change output directory
python main.py --output-dir results/
```

## Configuration

The output directory defaults to `out/` but can be customized:

```python
experiment.run(output_dir="results/")
```

## Logging and Visualization

pyexp includes a logging system for tracking metrics, text, and figures during experiments, with a TensorBoard-like viewer for visualization.

### Installation

```bash
# With viewer support
pip install "pyexp[viewer]"
```

### Logger

Use `Logger` to record scalars, text, and figures during training:

```python
from pyexp import Logger

logger = Logger("out/my_experiment/run1")

for epoch in range(100):
    logger.set_global_it(epoch)

    # Log scalar metrics
    logger.add_scalar("loss", loss_value)
    logger.add_scalar("accuracy", acc_value)

    # Log text (e.g., model outputs, debug info)
    logger.add_text("sample_output", generated_text)

    # Log figures
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(history)

    # interactive=True (default): renders as interactive widget in viewer
    # interactive=False: renders as static image (faster loading)
    logger.add_figure("training_curve", fig, interactive=False)

    # 3D plots work great as interactive
    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    logger.add_figure("loss_landscape", fig_3d, interactive=True)

    plt.close('all')

# Flush ensures all async writes complete
logger.flush()
```

**Storage structure:**
```
log_dir/
├── .pyexp                  # Marker file identifying this as a pyexp log
└── <iteration>/
    ├── scalars.json        # {tag: value, ...}
    ├── text.json           # {tag: text, ...}
    └── figures/
        ├── <tag>.cpkl      # Pickled figure (always saved)
        └── <tag>.meta      # Metadata (interactive flag)
```

### LogReader

Use `LogReader` to explore and load logged data programmatically:

```python
from pyexp import LogReader

# Discover runs in a directory
reader = LogReader("out/my_experiment")
print(reader.runs)  # ['run1', 'run2', ...]
print(reader.is_run)  # False (parent directory)

# Get a specific run
run = reader.get_run("run1")
print(run.is_run)  # True
print(run.iterations)  # [0, 1, 2, ..., 99]

# Available tags
print(run.scalar_tags)  # {'loss', 'accuracy'}
print(run.text_tags)    # {'sample_output'}
print(run.figure_tags)  # {'training_curve', 'loss_landscape'}

# Load scalar time series
loss_data = run.load_scalars("loss")  # [(0, 0.5), (1, 0.45), ...]

# Load text
texts = run.load_text("sample_output")  # [(0, "Hello..."), ...]

# Load figures
fig = run.load_figure("loss_landscape", iteration=50)
fig.show()  # Display the matplotlib figure

# Get iterations where a figure was logged
iters = run.figure_iterations("loss_landscape")  # [0, 25, 50, 75]
```

### Viewer

Launch the TensorBoard-like web viewer:

```bash
# View logs in a directory
uv run --extra viewer python -m pyexp.viewer out/my_experiment

# Or specify a port
uv run --extra viewer python -m pyexp.viewer out/my_experiment --port 8080
```

Or from Python:

```python
from pyexp.viewer import run
run("out/my_experiment", port=8765)
```

**Viewer features:**

- **Run discovery**: Automatically finds all runs under the specified directory
- **Multi-run selection**: Select multiple runs to compare side-by-side
- **Tabs**: Switch between Scalars, Text, and Figures views

**Scalars tab:**
- Plotly charts with hover tooltips showing exact values
- Multiple runs plotted on the same chart for comparison
- Per-tag log scale toggle
- Collapsible sections for each tag

**Text tab:**
- View logged text for each run
- Iteration slider to browse through logged text over time
- Collapsible sections for each tag

**Figures tab:**
- Interactive matplotlib figures (zoom, pan, rotate 3D)
- Static image rendering for figures logged with `interactive=False`
- Iteration slider to browse through logged figures over time
- Collapsible sections for each tag

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
- `Logger` - Log scalars, text, and figures during experiments
- `LogReader` - Explore and load logged data programmatically

### Decorators

- `@pyexp.experiment` - Define an experiment function
- `@pyexp.experiment(name="...", executor="...", timestamp=True/False)` - Define with options
- `@experiment.configs` - Register configs generator
- `@experiment.report` - Register report function

### Logger Methods

- `Logger(log_dir)` - Create a logger for the specified directory
- `set_global_it(it)` - Set the current iteration number
- `add_scalar(tag, value)` - Log a scalar value
- `add_text(tag, text)` - Log a text string
- `add_figure(tag, figure, interactive=True)` - Log a matplotlib figure
- `flush()` - Wait for all pending async writes to complete

### LogReader Properties & Methods

- `LogReader(log_dir)` - Create a reader for the specified directory
- `path` - The log directory path
- `is_run` - Whether this directory is a pyexp run
- `runs` - List of run names under this directory
- `get_run(name)` - Get a LogReader for a specific run
- `iterations` - List of iteration numbers in this run
- `scalar_tags` - Set of scalar tag names
- `text_tags` - Set of text tag names
- `figure_tags` - Set of figure tag names
- `load_scalars(tag)` - Load scalar values as (iteration, value) pairs
- `load_text(tag)` - Load text values as (iteration, text) pairs
- `load_figure(tag, iteration)` - Load a figure object
- `figure_iterations(tag)` - Get iterations where a figure was logged

### Types

- `ExecutorName` - Literal type: `"inline" | "subprocess" | "fork" | "ray"`
