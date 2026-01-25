"""pyexp - A library for running Python experiments."""

from .config import Config, Result, Tensor, build, load_config, merge, register, sweep
from .experiment import Experiment, experiment
from .executors import (
    Executor,
    ExecutorName,
    InlineExecutor,
    SubprocessExecutor,
    ForkExecutor,
    RayExecutor,
    get_executor,
)

__all__ = [
    "Config",
    "Result",
    "Tensor",
    "build",
    "load_config",
    "merge",
    "register",
    "sweep",
    "Experiment",
    "experiment",
    "Executor",
    "ExecutorName",
    "InlineExecutor",
    "SubprocessExecutor",
    "ForkExecutor",
    "RayExecutor",
    "get_executor",
]
