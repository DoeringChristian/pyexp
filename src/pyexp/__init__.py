"""pyexp - A library for running Python experiments."""

from .config import Config, Tensor, merge, sweep
from .experiment import Experiment, experiment
from .executors import (
    Executor,
    ExecutorName,
    InlineExecutor,
    SubprocessExecutor,
    ForkExecutor,
    get_executor,
)

__all__ = [
    "Config",
    "Tensor",
    "merge",
    "sweep",
    "Experiment",
    "experiment",
    "Executor",
    "ExecutorName",
    "InlineExecutor",
    "SubprocessExecutor",
    "ForkExecutor",
    "get_executor",
]
