"""pyexp - A library for running Python experiments."""

from .config import (
    Config,
    Runs,
    build,
    load_config,
    merge,
    register,
    sweep,
    to_dict,
)
from .experiment import Result, ExperimentRunner, Experiment, experiment
from .executors import (
    Executor,
    ExecutorName,
    ForkExecutor,
    InlineExecutor,
    RayExecutor,
    SubprocessExecutor,
    get_executor,
)
from .utils import checkout_snapshot, stash, stash_and_snapshot
from .log import LazyFigure, Logger, LogReader

__all__ = [
    "Config",
    "Runs",
    "build",
    "load_config",
    "merge",
    "register",
    "checkout_snapshot",
    "stash",
    "stash_and_snapshot",
    "sweep",
    "to_dict",
    "Result",
    "ExperimentRunner",
    "Experiment",
    "experiment",
    "Executor",
    "ExecutorName",
    "ForkExecutor",
    "InlineExecutor",
    "RayExecutor",
    "SubprocessExecutor",
    "get_executor",
    "LazyFigure",
    "Logger",
    "LogReader",
]
