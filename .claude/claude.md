# pyexp Project Conventions

## CLI Override Pattern

All decorator and `run()` function arguments should be overridable via CLI arguments.

**Priority order (highest to lowest):**
1. CLI arguments
2. `run()` function arguments
3. `@experiment` decorator arguments

### Current CLI Arguments

| CLI Argument | Overrides | Default |
|--------------|-----------|---------|
| `--name NAME` | `name` parameter | function name |
| `--executor EXECUTOR` | `executor` parameter | `"subprocess"` |
| `--output-dir DIR` | `output_dir` parameter | `<file_dir>/out/` (relative to experiment file) |
| `--continue [TIMESTAMP]` | continue previous run (latest if no arg) | new timestamp |
| `--report [TIMESTAMP]` | report from cache (latest or specific run) | - |
| `--list` | list all runs with status | - |
| `-s`, `--capture=no` | show live output | captured |

### Adding New Parameters

When adding a new parameter to `@experiment` decorator or `run()`:

1. Add the parameter to the decorator/run function signature
2. Add a corresponding CLI argument in `_parse_args()`
3. Update the resolution logic in `run()` to respect priority:
   ```python
   resolved_param = args.param or param or self._param_default
   ```
4. Update docstrings with the new CLI argument
5. Update README.md CLI Arguments table

### Example Resolution Pattern

```python
def run(self, ..., my_param=None):
    args = _parse_args()

    # CLI > run() > decorator
    resolved_param = args.my_param or my_param or self._my_param_default
```

## Progress Bar

- By default, subprocess output is suppressed and a progress bar is shown
- Use `-s` or `--capture=no` for live output (like pytest)
- Progress bar shows: passed (green), failed (red), cached (yellow)

## Executor System

- All executors must implement `run(instance, result_path, capture=True, stash=True)`
- `instance` is an `Experiment` subclass instance with `cfg` and `out` already set
- The `capture` parameter controls output suppression
- Custom executors should accept `**kwargs` for forward compatibility

## YAML Config Loading

Configs can be loaded from YAML files with composition support via `load_config()`.

### Import Resolution

- Configs can import other configs using the `imports` field
- Imports are resolved relative to the importing file's directory
- Import order: imports are merged first, then the current file's values override
- Dot notation in YAML keys updates nested values without replacing siblings

### Example Config Structure

```yaml
# base.yaml
model:
  hidden_size: 256
  num_layers: 4

# experiment.yaml
imports:
  - base.yaml
model.hidden_size: 512  # Updates only hidden_size, keeps num_layers
```

### Merge Semantics

Uses `merge()` function which supports:
- Regular keys: complete replacement
- Dotted keys (e.g., `model.hidden_size`): nested update preserving siblings

### Usage Pattern

```python
from pyexp import load_config, sweep

@experiment.configs
def configs():
    base = load_config("configs/base.yaml")
    return sweep([base], variations)
```

## Registry System

Classes can be registered and instantiated from config using `@register` and `build()`.

### Pattern

```python
from pyexp import register, build

@register
class MyModel:
    def __init__(self, size: int = 256):
        self.size = size

# In config: {"type": "MyModel", "size": 512}
model = build(MyModel, config["model"])
```

### Rules

- `@register` uses class name as registry key
- Duplicate registration raises `RuntimeError`
- `build()` requires `type` key in dict config
- Passing existing instance returns it unchanged (with type check)
- kwargs override config values

## Output Directory

The default output directory is relative to the experiment file's location:
- If `my_experiment.py` is at `/project/experiments/my_experiment.py`
- Output goes to `/project/experiments/out/<name>/<timestamp>/`

Override order: CLI `--output-dir` > `run(output_dir=...)` > `@experiment(output_dir=...)` > file-relative default

## Results Loading

The `results()` method loads results from previous runs:

```python
# Load latest run
results = experiment.results()

# Load specific run
results = experiment.results(timestamp="2024-01-25_14-30-00")
```

### Batch Manifests

Each run saves a `.batches/<timestamp>.json` manifest containing:
- `timestamp`: The batch timestamp
- `runs`: List of run names (sanitized config names)
- `commit`: Optional git commit hash of the source snapshot

This enables `results()` to load all experiments from a previous run into a flat `Runs` collection.

## Experiment Object

Each experiment instance has these properties:
- `exp.name` - Config name (shorthand for `exp.cfg.get("name", "")`)
- `exp.cfg` - Config object for this run
- `exp.out` - Path to experiment output directory
- `exp.error` - Error message if failed, else `None`
- `exp.log` - Captured stdout/stderr
- `exp.result` - Experiment return value (when using decorator API)

The `out` path is available both during experiment execution (`self.out`) and in results (`exp.out`).

## Checkpoint System

The `@chkpt` decorator enables partial resumption of experiments. It works only with class-based experiments (subclasses of `Experiment`).

### How It Works

1. Before running a `@chkpt` method, check if `<exp.out>/.checkpoints/<method_name>.pkl` exists
2. If exists: restore `self` state from checkpoint (excluding runner-managed `_Experiment__*` attrs) and return cached return value
3. If not: run the method, then pickle `self` state and return value to checkpoint file
4. On failure with `retry > 1`: retry the method before giving up

### State Captured

The checkpoint saves:
- All user-defined attributes on `self` (everything except `_Experiment__cfg`, `_Experiment__out`, etc.)
- The return value of the method (if any)

On restore, these are copied back onto `self`, effectively skipping the method execution.

### Usage Pattern

```python
from pyexp import Experiment, chkpt

class MyExperiment(Experiment):
    @chkpt(retry=3)
    def train(self):
        self.model = expensive_training(self.cfg)

    @chkpt
    def evaluate(self):
        self.accuracy = evaluate(self.model)

    def experiment(self):
        self.train()      # Checkpointed
        self.evaluate()   # If this fails, train() won't re-run on --continue
```

### Limitations

- Only works with class-based experiments (not decorator API)
- State must be picklable (uses cloudpickle)
- Checkpoints are per-method, not per-iteration within a method
