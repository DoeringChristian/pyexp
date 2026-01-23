# Runner

A lightweight Python library for running experiments with automatic caching, configuration management, and reporting.

## Installation

```bash
pip install git+https://github.com/username/runner.git
```

## Quick Start

```python
import runner
from runner import Config

@runner.experiment
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
def report(configs, results):
    for config, result in zip(configs, results):
        print(f"{config['name']}: {result['accuracy']}")

if __name__ == "__main__":
    experiment.run()
```

## Features

### Automatic Caching

Experiment results are cached in `out/<name>-<hash>/result.pkl`. Re-running the script skips completed experiments:

```bash
python main.py          # Runs all experiments
python main.py          # Skips experiments, loads from cache
python main.py --rerun  # Forces re-run, ignores cache
python main.py --report # Only runs report from cached results
```

### Config Class

Configs support both dict-style and dot notation access:

```python
config["learning_rate"]  # dict-style
config.learning_rate     # dot notation
config.optimizer.lr      # nested access
```

### Output Directory

Each experiment receives an `out_dir` in its config pointing to its cache directory. Use this to save artifacts:

```python
@runner.experiment
def experiment(config: Config):
    model = train(config)
    torch.save(model, config.out_dir / "model.pt")
    return {"accuracy": 0.95}
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

## Configuration

The output directory defaults to `out/` but can be customized:

```python
experiment.run(output_dir="results/")
```
