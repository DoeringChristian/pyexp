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
def report(results, report_dir):
    # Each result has: name, config, result, error, log, logger, out
    for r in results:
        print(f"{r.name}: {r.result['accuracy']}")

    # Filter by config values
    fast_results = results[{"config.learning_rate": 0.01}]

if __name__ == "__main__":
    experiment.run()
```

## Features

### Automatic Caching

Each run creates a new timestamped folder. Use `--continue` to resume a previous run:

```bash
python main.py                              # New run (new timestamp)
python main.py --continue                   # Continue most recent run
python main.py --continue=2024-01-25_14-30-00  # Continue specific run
python main.py --report                        # Report from most recent run
python main.py --report=2024-01-25_14-30-00    # Report from specific run
```

### Execution Phases

The experiment framework separates execution into three phases:

1. **Config Generation** (only on fresh start - no `--continue` or `--report`):
   - Runs the `@experiment.configs` function to generate configurations
   - Computes directory hashes and creates experiment folders
   - Saves `runs.json` (folder references) and individual `config.json` files

2. **Experiment Execution** (fresh start or `--continue`, skipped for `--report`):
   - Loads configs from saved `config.json` files (not recomputed)
   - Runs experiments and saves results to `result.pkl`

3. **Report Generation** (always runs):
   - Loads all results from disk
   - Runs the `@experiment.report` function

This separation ensures that `--continue` and `--report` modes are immune to changes in the config generation code - they always use the saved configurations from the original run.

### Loading Previous Results

Use the `results()` method to load results from a previous run programmatically:

```python
# Load results from the latest run
results = experiment.results()

# Load results from a specific timestamp
results = experiment.results(timestamp="2024-01-25_14-30-00")

# Override output directory
results = experiment.results(output_dir="/data/experiments")

# Work with loaded results
for r in results:
    print(f"{r.name}: {r.result['accuracy']}")

# Results preserve tensor shape from sweeps
print(results.shape)  # e.g., (1, 2, 2)
filtered = results[{"config.learning_rate": 0.01}]
```

### Output Folder Structure

Results are organized by experiment name and timestamp. By default, the output directory is created relative to the experiment file:

```
<experiment_file_dir>/out/
  <experiment_name>/
    <timestamp>/                    # Each run gets a new timestamp
      runs.json                  # References to run folders and shape
      <config_name>-<hash>/
        config.json                 # Full config (includes 'out' path)
        result.pkl
        log.out                     # Captured stdout/stderr
        model.pt                    # Your saved artifacts
      <config_name>-<hash>/
        ...
      report/                       # Report outputs
    <timestamp>/                    # Another run
      ...
```

**Directory name sanitization:** Config names are automatically sanitized for use as directory names. Characters `/\:*?"<>|` are replaced with `_` to ensure cross-platform compatibility and prevent accidental subdirectory creation.

```python
# Default: output_dir is relative to experiment file
# If my_experiment.py is at /project/experiments/my_experiment.py:
@pyexp.experiment
def my_experiment(config): ...
# -> /project/experiments/out/my_experiment/<timestamp>/<config>-<hash>/

# Custom output directory in decorator
@pyexp.experiment(output_dir="/data/results")
def my_experiment(config): ...
# -> /data/results/my_experiment/<timestamp>/<config>-<hash>/

# Override at runtime
experiment.run(output_dir="results/")

# Override via CLI
# python my_experiment.py --output-dir /tmp/test

# Custom experiment name
@pyexp.experiment(name="mnist")
def my_experiment(config): ...

# Override name at runtime
experiment.run(name="experiment_v2")
```

### Robust Execution

By default, each experiment runs in an isolated subprocess, protecting against crashes (including segfaults). Failed experiments return error information without stopping other experiments:

```python
@experiment.report
def report(results, report_dir):
    for r in results:
        if r.error:
            print(f"{r.name} failed: {r.error}")
        else:
            print(f"{r.name}: {r.result['accuracy']}")
```

### Checkpoints

Use the `@chkpt` decorator to create checkpoint boundaries within experiments. Checkpoints enable:
- **Partial resumption**: If an experiment fails mid-way, `--continue` resumes from the last checkpoint
- **Automatic retry**: Failed checkpoint methods can be automatically retried

```python
from pyexp import Experiment, ExperimentRunner, chkpt

class MyExperiment(Experiment):
    model: dict
    results: dict

    @chkpt(retry=3)
    def train(self):
        """Training phase - retried up to 3 times on failure."""
        self.model = train_model(self.cfg)

    @chkpt
    def evaluate(self):
        """Evaluation phase - checkpointed."""
        self.results = evaluate(self.model)

    def experiment(self):
        self.train()      # Checkpoint saved after completion
        self.evaluate()   # If this fails, train() won't re-run on --continue

    @staticmethod
    def configs():
        return [{"name": "exp", "lr": 0.01}]

    @staticmethod
    def report(results, out):
        for exp in results:
            print(f"{exp.name}: {exp.results}")
```

Checkpoints are stored in `<experiment_dir>/.checkpoints/<method_name>.pkl` and contain the experiment instance's state after the method completes.

**How it works:**
1. Before running a `@chkpt` method, check if a checkpoint file exists
2. If exists: restore `self` state from checkpoint and skip execution
3. If not: run the method, then save `self` state to checkpoint
4. On failure with `retry > 1`: retry the method up to N times before giving up

**Note:** The `@chkpt` decorator only works with class-based experiments (subclasses of `Experiment`), not with the decorator API.

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

@experiment.report
def report(results, report_dir):
    for r in results:
        # Access output directory via result.out or result.config.out
        model = torch.load(r.out / "model.pt")
        print(f"{r.name}: saved at {r.out}")
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

### Deep Merge with `**` Prefix

By default, setting a dict value replaces the entire dict. Prefix a key with `**` to deep-merge instead, updating only the specified keys while preserving the rest:

```python
base = {"bsdf": {"type": "Diffuse", "color": 5, "roughness": 0.1}}

# Without **: replaces the entire dict (roughness is lost)
merge(base, {"bsdf": {"type": "Test", "color": 10}})
# -> {"bsdf": {"type": "Test", "color": 10}}

# With **: merges into existing dict (roughness is preserved)
merge(base, {"**bsdf": {"type": "Test", "color": 10}})
# -> {"bsdf": {"type": "Test", "color": 10, "roughness": 0.1}}
```

This also works with dot-notation paths and in sweeps:

```python
cfgs = sweep(cfgs, [
    {"name": "test", "**model.encoder": {"type": "ResNet", "pretrained": True}},
])
```

### Result Filtering

The report function receives a `Tensor` of results. Each result contains:
- `name`: the combined config name
- `config`: the full config dict (includes `out` path)
- `result`: the experiment return value
- `error`: error message if failed, else `None`
- `log`: captured stdout/stderr
- `logger`: `LogReader` if logging was used
- `out`: path to experiment output directory (same as `config.out`)

Filter results using pattern matching or dict queries:

```python
@experiment.report
def report(results, report_dir):
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
def report(results, report_dir):
    # Shape is (1, 2, 2) from sweeps
    print(results.shape)

    # Index by position
    first_lr = results[:, 0, :]  # All configs with first lr value

    # Access individual result
    r = results[0, 1, 0]
    print(r.name, r.result["accuracy"])
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
| `--continue [TIMESTAMP]` | Continue a previous run. Without argument, continues most recent. With argument, continues that specific timestamp. |
| `--report [TIMESTAMP]` | Generate report from cache. Without argument, uses most recent run. With argument, uses that specific timestamp. |
| `--list` | List all previous runs with their status |
| `--filter REGEX` | Only run configs whose names match the regex pattern |
| `-s`, `--capture=no` | Show subprocess output instead of progress bar |
| `--viewer` | Start the viewer after experiments complete |
| `--viewer-port PORT` | Port for the viewer (default: 8765) |
| `--no-stash` | Disable git stash (don't capture repository state) |

By default, experiment output is captured and a progress bar is shown. Use `-s` to see live output from each experiment (similar to pytest).

```bash
# Override executor from command line
python main.py --executor inline

# Continue most recent run
python main.py --continue

# Continue specific run
python main.py --continue=2024-01-25_14-30-00

# Change output directory
python main.py --output-dir results/
```

## Configuration

The output directory defaults to `out/` relative to the experiment file's location:

```python
# Override in decorator
@pyexp.experiment(output_dir="/data/experiments")
def my_experiment(config): ...

# Override in run()
experiment.run(output_dir="results/")

# Override via CLI
# python my_experiment.py --output-dir /tmp/test
```

Priority: CLI `--output-dir` > `run(output_dir=...)` > `@experiment(output_dir=...)` > file-relative default

## Logging and Visualization

pyexp includes a logging system for tracking metrics, text, and figures during experiments, with a TensorBoard-like viewer for visualization.

### Installation

```bash
# With viewer support
pip install "pyexp[viewer]"
```

### Automatic Logger in Experiments

When using `@pyexp.experiment`, add a second `logger` parameter to your function to receive an automatically-created `Logger` instance. If no second parameter is present, no logger is created:

```python
from pyexp import Logger

@pyexp.experiment
def my_experiment(config, logger: Logger):
    for epoch in range(config.epochs):
        logger.set_global_it(epoch)

        # Train your model...
        loss = train_step()

        logger.add_scalar("loss", loss)
        logger.add_scalar("accuracy", accuracy)

    return {"final_loss": loss}

# Without logger (no logging overhead)
@pyexp.experiment
def simple_experiment(config):
    return {"result": compute(config)}
```

When a logger is used, each result includes a `LogReader` for accessing the logged data:

```python
@my_experiment.report
def report(results, report_dir):
    for r in results:
        if r.logger:
            # Access logged data via LogReader
            loss_data = r.logger.load_scalars("loss")
            print(f"{r.name}: final_loss={loss_data[-1][1]:.4f}")
```

### Standalone Logger

You can also use `Logger` directly for standalone logging:

```python
from pyexp import Logger

# Default: protobuf format (single events.pb file, faster, smaller)
logger = Logger("out/my_experiment/run1")

# JSONL format (human-readable, multiple files)
logger = Logger("out/my_experiment/run1", use_protobuf=False)

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

**Storage formats:**

By default, pyexp uses a protobuf-based format similar to TensorBoard for efficient storage:

```
log_dir/
├── .pyexp                  # Marker file identifying this as a pyexp log
└── events.pb               # All events in a single protobuf file
```

With `use_protobuf=False`, the traditional JSONL format is used:

```
log_dir/
├── .pyexp                  # Marker file identifying this as a pyexp log
├── scalars.jsonl           # {"it": N, "tag": "...", "value": V} per line
├── text.jsonl              # {"it": N, "tag": "...", "text": "..."} per line
└── <iteration>/
    ├── figures/
    │   ├── <tag>.cpkl      # Pickled figure
    │   └── <tag>.meta      # Metadata (interactive flag)
    └── checkpoints/
        └── <tag>.cpkl      # Pickled checkpoint object
```

The protobuf format provides ~4x faster writes and ~2x smaller file sizes compared to JSONL. The `LogReader` automatically detects the format.

**Automatic logging in experiments:**

When using `@pyexp.experiment` with a logger parameter, the framework automatically logs at iteration 0:
- **`config`**: The experiment configuration as YAML text
- **`git_commit`**: The git stash commit hash (if stash is enabled)

This enables exact reproducibility of experiments. Disable git stash via decorator or CLI:

```python
# Disable via experiment decorator
@pyexp.experiment(stash=False)
def my_experiment(config, logger):
    ...

# Or via CLI
python main.py --no-stash
```

View the logged config and commit via the result's LogReader:

```python
@my_experiment.report
def report(results, report_dir):
    for r in results:
        it, config_yaml = r.logger["config"]
        it, commit = r.logger["git_commit"]
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
print(run.scalar_tags)      # {'loss', 'accuracy'}
print(run.text_tags)        # {'sample_output'}
print(run.figure_tags)      # {'training_curve', 'loss_landscape'}
print(run.checkpoint_tags)  # {'model', 'optimizer'}

# Quick access to last value via __getitem__
it, loss = run["loss"]                  # Last scalar value
it, fig = run["loss_landscape"]         # Last figure
it, state = run["model"]                # Last checkpoint

# Load full time series
loss_data = run.load_scalars("loss")    # [(0, 0.5), (1, 0.45), ...]
texts = run.load_text("sample_output")  # [(0, "Hello..."), ...]

# Load specific iteration
fig = run.load_figure("loss_landscape", iteration=50)
state = run.load_checkpoint("model", iteration=100)

# Get iterations where a tag was logged
iters = run.figure_iterations("loss_landscape")      # [0, 25, 50, 75]
iters = run.checkpoint_iterations("model")           # [0, 100, 200, ...]
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

**Integrated with experiments:**

Start the viewer automatically alongside experiments for real-time inspection:

```python
# Via decorator
@pyexp.experiment(viewer=True)
def my_experiment(config):
    ...

# Via run()
my_experiment.run(viewer=True, viewer_port=8080)

# Via CLI
python main.py --viewer
python main.py --viewer --viewer-port 8080
```

The viewer starts as a background process before experiments begin, allowing you to monitor metrics, text, and figures in real-time as they are logged.

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
- `merge(base, update)` - Merge dicts with dot-notation and `**` deep-merge support
- `get_executor(name)` - Get executor by name or return instance

### Classes

- `Config` - Dict subclass with dot notation access
- `Result` - Experiment result with config, result, error, log, logger, out
- `Tensor` - Shape-preserving container with advanced indexing
- `Experiment` - Experiment runner with caching
- `Executor` - Abstract base class for custom executors
- `InlineExecutor` - Runs in same process (no isolation)
- `SubprocessExecutor` - Runs in isolated subprocess (default)
- `ForkExecutor` - Runs in forked process (Unix only)
- `RayExecutor` - Runs with Ray for distributed execution
- `Logger` - Log scalars, text, and figures during experiments
- `LogReader` - Explore and load logged data programmatically

### Result Properties

- `result.name` - Config name
- `result.config` - Full config dict (includes `out`)
- `result.result` - Experiment return value
- `result.error` - Error message if failed, else `None`
- `result.log` - Captured stdout/stderr
- `result.logger` - `LogReader` if logging was used, else `None`
- `result.out` - Path to experiment output directory

### Decorators

- `@pyexp.experiment` - Define an experiment function
- `@pyexp.experiment(name="...", output_dir="...", executor="...")` - Define with options
- `@experiment.configs` - Register configs generator
- `@experiment.report` - Register report function
- `@chkpt` - Checkpoint a method in a class-based experiment
- `@chkpt(retry=N)` - Checkpoint with automatic retry on failure

### Experiment Methods

- `experiment.run(...)` - Execute the full pipeline: configs → experiments → report
- `experiment.results(timestamp=None, output_dir=None)` - Load results from a previous run
  - `timestamp`: Specific timestamp or `"latest"` (default: latest)
  - `output_dir`: Override output directory (default: file-relative)
  - Returns: `Tensor` of `Result` objects preserving sweep shape

### Logger Methods

- `Logger(log_dir, use_protobuf=True)` - Create a logger for the specified directory
- `set_global_it(it)` - Set the current iteration number
- `add_scalar(tag, value)` - Log a scalar value
- `add_text(tag, text)` - Log a text string
- `add_figure(tag, figure, interactive=True)` - Log a matplotlib figure
- `add_checkpoint(tag, obj)` - Log an arbitrary picklable object
- `flush()` - Wait for all pending async writes to complete

### LogReader Properties & Methods

- `LogReader(log_dir)` - Create a reader for the specified directory
- `reader[tag]` - Get last `(iteration, value)` for any tag type
- `path` - The log directory path
- `is_run` - Whether this directory is a pyexp run
- `runs` - List of run names under this directory
- `get_run(name)` - Get a LogReader for a specific run
- `iterations` - List of iteration numbers in this run
- `scalar_tags` - Set of scalar tag names
- `text_tags` - Set of text tag names
- `figure_tags` - Set of figure tag names
- `checkpoint_tags` - Set of checkpoint tag names
- `load_scalars(tag)` - Load scalar values as (iteration, value) pairs
- `load_text(tag)` - Load text values as (iteration, text) pairs
- `load_figure(tag, iteration)` - Load a figure object
- `load_checkpoint(tag, iteration)` - Load a checkpoint object
- `load_checkpoints(tag)` - Load all checkpoints as (iteration, object) pairs
- `figure_iterations(tag)` - Get iterations where a figure was logged
- `checkpoint_iterations(tag)` - Get iterations where a checkpoint was logged

### Types

- `ExecutorName` - Literal type: `"inline" | "subprocess" | "fork" | "ray"`
