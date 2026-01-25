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
| `--output-dir DIR` | `output_dir` parameter | `"out"` |
| `--no-timestamp` | `timestamp=False` | - |
| `--timestamp TIMESTAMP` | specific timestamp folder | - |
| `--report` | skip experiments, report only | - |
| `--rerun` | ignore cache | - |
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

- All executors must implement `run(fn, config, result_path, capture=True)`
- The `capture` parameter controls output suppression
- Custom executors should accept `**kwargs` for forward compatibility
