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
from .runner import Result, ExperimentRunner
from .experiment import Experiment, experiment
from .executors import (
    Executor,
    ExecutorName,
    FnFuture,
    FnResult,
    ForkExecutor,
    InlineExecutor,
    RayExecutor,
    SubprocessExecutor,
    get_executor,
)
from .utils import (
    checkout_snapshot,
    content_hash,
    package_files,
    stash,
    stash_and_snapshot,
    DEFAULT_PACKAGE_SUFFIXES,
)
from .log import LazyFigure, Logger, LogReader

__all__ = [
    "Config",
    "Runs",
    "build",
    "load_config",
    "merge",
    "register",
    "checkout_snapshot",
    "content_hash",
    "package_files",
    "stash",
    "stash_and_snapshot",
    "sweep",
    "to_dict",
    "Result",
    "ExperimentRunner",
    "Experiment",
    "experiment",
    "FnFuture",
    "FnResult",
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
    "DEFAULT_PACKAGE_SUFFIXES",
]
