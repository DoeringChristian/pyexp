"""pyexp - A library for running Python experiments."""

from .config import (
    Config,
    ConfigTensor,
    Result,
    ResultTensor,
    Tensor,
    build,
    load_config,
    merge,
    register,
    sweep,
)
from .experiment import Experiment, experiment
from .executors import (
    Executor,
    ExecutorName,
    ForkExecutor,
    InlineExecutor,
    RayExecutor,
    SubprocessExecutor,
    get_executor,
)
from .utils import stash

__all__ = [
    "Config",
    "ConfigTensor",
    "Result",
    "ResultTensor",
    "Tensor",
    "build",
    "load_config",
    "merge",
    "register",
    "stash",
    "sweep",
    "Experiment",
    "experiment",
    "Executor",
    "ExecutorName",
    "ForkExecutor",
    "InlineExecutor",
    "RayExecutor",
    "SubprocessExecutor",
    "get_executor",
]
