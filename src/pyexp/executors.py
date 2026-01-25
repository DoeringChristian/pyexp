"""Executor classes for running experiments in different isolation modes.

This module provides a modular system for experiment execution. Each executor
implements a different isolation strategy:

- InlineExecutor: Runs in the same process (no isolation)
- SubprocessExecutor: Runs in a subprocess using cloudpickle (cross-platform)
- ForkExecutor: Runs in a forked process (Unix only, fastest isolation)
- RayExecutor: Runs using Ray for distributed execution (requires `pip install pyexp[ray]`)

Custom executors can be created by subclassing Executor.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal
import os
import pickle
import subprocess
import sys
import tempfile
import traceback

import cloudpickle

from .config import Config

# Valid executor names
ExecutorName = Literal["inline", "subprocess", "fork", "ray"]


class Executor(ABC):
    """Abstract base class for experiment executors.

    Subclass this to implement custom execution strategies.
    """

    @abstractmethod
    def run(
        self,
        fn: Callable[[Config], Any],
        config: Config,
        result_path: Path,
        capture: bool = True,
    ) -> dict:
        """Run a single experiment and return the result.

        Args:
            fn: The experiment function to execute.
            config: The experiment config (already has 'out' set).
            result_path: Path where result should be cached.
            capture: If True (default), capture output. If False, show output live.

        Returns:
            The experiment result dict. If execution failed, should contain
            '__error__': True along with 'type', 'message', and optionally 'traceback'.
        """
        pass


class InlineExecutor(Executor):
    """Executor that runs experiments in the same process.

    This provides no isolation - crashes will affect the main process.
    Useful for debugging or when isolation overhead is not desired.
    """

    def run(
        self,
        fn: Callable[[Config], Any],
        config: Config,
        result_path: Path,
        capture: bool = True,
    ) -> dict:
        """Run experiment inline and cache result."""
        # Note: capture has no effect for inline execution since output
        # goes directly to the parent process stdout/stderr
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = fn(config)
        with open(result_path, "wb") as f:
            pickle.dump(result, f)
        return result


class SubprocessExecutor(Executor):
    """Executor that runs experiments in isolated subprocesses using cloudpickle.

    This provides strong isolation - crashes (including segfaults) won't affect
    the main process. Works cross-platform (Windows, macOS, Linux).

    The experiment function is serialized using cloudpickle, which captures
    the function's bytecode. However, imported modules are serialized by
    reference and will be re-imported in the subprocess.
    """

    def run(
        self,
        fn: Callable[[Config], Any],
        config: Config,
        result_path: Path,
        capture: bool = True,
    ) -> dict:
        """Run experiment in subprocess via cloudpickle serialization."""
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Create payload with serialized function
        payload = {
            "fn": fn,
            "config": config,
            "result_path": str(result_path),
        }

        # Write payload to temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            payload_path = f.name
            cloudpickle.dump(payload, f)

        try:
            # Run worker subprocess
            if capture:
                proc = subprocess.run(
                    [sys.executable, "-m", "pyexp.worker", payload_path],
                    capture_output=True,
                    text=True,
                )
                stdout, stderr = proc.stdout, proc.stderr
            else:
                # Show output live
                proc = subprocess.run(
                    [sys.executable, "-m", "pyexp.worker", payload_path],
                )
                stdout, stderr = "", ""

            # Check if result was written
            if result_path.exists():
                with open(result_path, "rb") as f:
                    return pickle.load(f)
            else:
                # Subprocess crashed before writing result
                return {
                    "__error__": True,
                    "type": "SubprocessError",
                    "message": f"Subprocess exited with code {proc.returncode}",
                    "stdout": stdout,
                    "stderr": stderr,
                }
        finally:
            # Clean up payload file
            Path(payload_path).unlink(missing_ok=True)


class ForkExecutor(Executor):
    """Executor that runs experiments in forked processes (Unix only).

    This provides isolation while guaranteeing that all forked processes have
    identical interpreter state (same loaded modules, same code versions).
    Uses os.fork() which is very efficient (copy-on-write memory).

    Advantages over SubprocessExecutor:
    - All processes share exact same module state (no re-importing)
    - Faster startup (no need to re-import modules)
    - No serialization of the function needed

    Limitations:
    - Unix only (Linux, macOS) - not available on Windows
    - Some libraries have issues with fork (CUDA, some macOS frameworks)
    - File descriptors are shared with parent (can cause issues)
    """

    def __init__(self):
        if not hasattr(os, "fork"):
            raise RuntimeError("ForkExecutor is only available on Unix systems")

    def run(
        self,
        fn: Callable[[Config], Any],
        config: Config,
        result_path: Path,
        capture: bool = True,
    ) -> dict:
        """Run experiment in a forked process."""
        result_path.parent.mkdir(parents=True, exist_ok=True)

        pid = os.fork()

        if pid == 0:
            # Child process
            try:
                # Suppress output if capturing
                if capture:
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, 1)  # stdout
                    os.dup2(devnull, 2)  # stderr
                    os.close(devnull)

                result = fn(config)
                with open(result_path, "wb") as f:
                    pickle.dump(result, f)
                os._exit(0)
            except Exception as e:
                # Write error information
                error_result = {
                    "__error__": True,
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                try:
                    with open(result_path, "wb") as f:
                        pickle.dump(error_result, f)
                except Exception:
                    pass
                os._exit(1)
        else:
            # Parent process - wait for child
            _, status = os.waitpid(pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)

            if result_path.exists():
                with open(result_path, "rb") as f:
                    return pickle.load(f)
            else:
                # Child crashed before writing result
                return {
                    "__error__": True,
                    "type": "ForkError",
                    "message": f"Forked process exited with code {exit_code}",
                }


class RayExecutor(Executor):
    """Executor that runs experiments using Ray for distributed execution.

    Ray provides distributed computing capabilities, allowing experiments to run
    across multiple cores or machines. Ray handles serialization internally
    (uses cloudpickle under the hood).

    Args:
        address: Ray cluster address. Use "auto" to connect to an existing cluster,
                 or None to start a local Ray instance. Default: None.
        runtime_env: Runtime environment configuration for distributing code to workers.
                     Useful for cluster execution. Example:
                     {"working_dir": ".", "pip": ["pandas"], "excludes": ["*.pt"]}
        num_cpus: Number of CPUs to use. Default: all available.
        num_gpus: Number of GPUs to use. Default: all available.
        **ray_init_kwargs: Additional arguments passed to ray.init().

    Example:
        # Local execution (default)
        executor = RayExecutor()

        # Connect to existing cluster with code sync
        executor = RayExecutor(
            address="auto",
            runtime_env={"working_dir": ".", "excludes": ["data/", "*.pt"]}
        )

        # Specify resources
        executor = RayExecutor(num_cpus=4, num_gpus=1)

    Advantages:
    - Can distribute work across multiple machines
    - Efficient multi-core utilization
    - Built-in fault tolerance and retry mechanisms
    - Good for long-running experiments

    Limitations:
    - Requires Ray to be installed (`pip install pyexp[ray]`)
    - Additional overhead for simple single-machine workloads
    - Ray cluster setup required for multi-machine execution
    """

    def __init__(
        self,
        address: str | None = None,
        runtime_env: dict | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        **ray_init_kwargs,
    ):
        try:
            import ray
            self._ray = ray
        except ImportError:
            raise RuntimeError(
                "RayExecutor requires Ray to be installed. "
                "Install it with: pip install pyexp[ray]"
            )

        self._runtime_env = runtime_env

        # Initialize Ray if not already running
        if not self._ray.is_initialized():
            init_kwargs = {
                "ignore_reinit_error": True,
                **ray_init_kwargs,
            }
            if address is not None:
                init_kwargs["address"] = address
            if runtime_env is not None:
                init_kwargs["runtime_env"] = runtime_env
            if num_cpus is not None:
                init_kwargs["num_cpus"] = num_cpus
            if num_gpus is not None:
                init_kwargs["num_gpus"] = num_gpus

            self._ray.init(**init_kwargs)

    def run(
        self,
        fn: Callable[[Config], Any],
        config: Config,
        result_path: Path,
        capture: bool = True,
    ) -> dict:
        """Run experiment as a Ray task."""
        result_path.parent.mkdir(parents=True, exist_ok=True)

        @self._ray.remote
        def _run_experiment(fn, config, result_path_str, capture):
            """Ray remote function to execute experiment."""
            import os
            import pickle
            import sys
            import traceback
            from pathlib import Path

            result_path = Path(result_path_str)
            try:
                # Suppress output if capturing
                if capture:
                    devnull = open(os.devnull, "w")
                    old_stdout, old_stderr = sys.stdout, sys.stderr
                    sys.stdout, sys.stderr = devnull, devnull

                try:
                    result = fn(config)
                finally:
                    if capture:
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                        devnull.close()

                with open(result_path, "wb") as f:
                    pickle.dump(result, f)
                return result
            except Exception as e:
                error_result = {
                    "__error__": True,
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                with open(result_path, "wb") as f:
                    pickle.dump(error_result, f)
                return error_result

        # Submit task and wait for result
        future = _run_experiment.remote(fn, config, str(result_path), capture)
        try:
            result = self._ray.get(future)
            return result
        except Exception as e:
            return {
                "__error__": True,
                "type": "RayError",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }


# Registry of built-in executors
EXECUTORS: dict[str, type[Executor]] = {
    "inline": InlineExecutor,
    "subprocess": SubprocessExecutor,
    "fork": ForkExecutor,
    "ray": RayExecutor,
}


def get_executor(executor: ExecutorName | Executor) -> Executor:
    """Get an executor instance from a string name or executor instance.

    Args:
        executor: Either an executor name ("inline", "subprocess", "fork")
                  or an Executor instance.

    Returns:
        An Executor instance.

    Raises:
        ValueError: If the executor name is not recognized.
        RuntimeError: If the executor is not available on this platform.
    """
    if isinstance(executor, Executor):
        return executor

    if isinstance(executor, str):
        if executor not in EXECUTORS:
            available = ", ".join(EXECUTORS.keys())
            raise ValueError(f"Unknown executor '{executor}'. Available: {available}")
        return EXECUTORS[executor]()

    raise TypeError(f"executor must be str or Executor, got {type(executor).__name__}")
