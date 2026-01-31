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
from typing import Any, Callable, Literal, TYPE_CHECKING
import os
import pickle
import subprocess
import sys
import tempfile
import traceback

import cloudpickle

if TYPE_CHECKING:
    from .experiment import Experiment

# Valid executor names
ExecutorName = Literal["inline", "subprocess", "fork", "ray"]


class Executor(ABC):
    """Abstract base class for experiment executors.

    Subclass this to implement custom execution strategies.
    """

    @abstractmethod
    def run(
        self,
        instance: "Experiment",
        result_path: Path,
        capture: bool = True,
        stash: bool = True,
    ) -> None:
        """Run a single experiment and pickle the result.

        Args:
            instance: The experiment instance (cfg/out already set).
            result_path: Path where the pickled instance should be saved.
            capture: If True (default), capture output. If False, show output live.
            stash: If True, capture git commit hash at the start of the experiment.

        The executor should:
        1. Call instance.experiment()
        2. On error: set instance._Experiment__error
        3. Set instance._Experiment__log with captured stdout/stderr
        4. Pickle the entire instance to result_path
        """
        pass


class InlineExecutor(Executor):
    """Executor that runs experiments in the same process.

    This provides no isolation - crashes will affect the main process.
    Useful for debugging or when isolation overhead is not desired.
    """

    def run(
        self,
        instance: "Experiment",
        result_path: Path,
        capture: bool = True,
        stash: bool = True,
    ) -> None:
        """Run experiment inline and cache result."""
        import io
        import sys

        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Capture output if requested
        log = ""
        if capture:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

        try:
            instance.experiment()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            instance._Experiment__error = error_msg
        finally:
            if capture:
                log = sys.stdout.getvalue() + sys.stderr.getvalue()
                sys.stdout, sys.stderr = old_stdout, old_stderr

        instance._Experiment__log = log

        # Pickle entire instance using cloudpickle for dynamic classes
        with open(result_path, "wb") as f:
            cloudpickle.dump(instance, f)


class SubprocessExecutor(Executor):
    """Executor that runs experiments in isolated subprocesses using cloudpickle.

    This provides strong isolation - crashes (including segfaults) won't affect
    the main process. Works cross-platform (Windows, macOS, Linux).

    The experiment instance is serialized using cloudpickle, which captures
    the instance's state. However, imported modules are serialized by
    reference and will be re-imported in the subprocess.
    """

    def run(
        self,
        instance: "Experiment",
        result_path: Path,
        capture: bool = True,
        stash: bool = True,
    ) -> None:
        """Run experiment in subprocess via cloudpickle serialization."""
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Create payload with serialized instance
        payload = {
            "instance": instance,
            "result_path": str(result_path),
            "stash": stash,
        }

        # Write payload to temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            payload_path = f.name
            cloudpickle.dump(payload, f)

        try:
            # Run worker subprocess with inherited sys.path via PYTHONPATH
            env = os.environ.copy()
            pythonpath = os.pathsep.join(sys.path)
            if env.get("PYTHONPATH"):
                pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
            env["PYTHONPATH"] = pythonpath

            if capture:
                proc = subprocess.run(
                    [sys.executable, "-m", "pyexp.worker", payload_path],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                log = proc.stdout + proc.stderr
            else:
                # Show output live
                proc = subprocess.run(
                    [sys.executable, "-m", "pyexp.worker", payload_path],
                    env=env,
                )
                log = ""

            # Check if result was written
            if result_path.exists():
                # Load and update with log
                with open(result_path, "rb") as f:
                    instance = pickle.load(f)
                instance._Experiment__log = log
                # Re-save with log included
                with open(result_path, "wb") as f:
                    cloudpickle.dump(instance, f)
            else:
                # Subprocess crashed before writing result
                instance._Experiment__error = f"SubprocessError: exited with code {proc.returncode}"
                instance._Experiment__log = log
                with open(result_path, "wb") as f:
                    cloudpickle.dump(instance, f)
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
    - No serialization of the instance needed

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
        instance: "Experiment",
        result_path: Path,
        capture: bool = True,
        stash: bool = True,
    ) -> None:
        """Run experiment in a forked process."""
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Create pipe for capturing output
        if capture:
            read_fd, write_fd = os.pipe()

        pid = os.fork()

        if pid == 0:
            # Child process
            try:
                if capture:
                    os.close(read_fd)
                    os.dup2(write_fd, 1)  # stdout
                    os.dup2(write_fd, 2)  # stderr
                    os.close(write_fd)

                instance.experiment()

                with open(result_path, "wb") as f:
                    cloudpickle.dump(instance, f)
                os._exit(0)
            except Exception as e:
                # Write error information
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                instance._Experiment__error = error_msg
                try:
                    with open(result_path, "wb") as f:
                        cloudpickle.dump(instance, f)
                except Exception:
                    pass
                os._exit(1)
        else:
            # Parent process
            log = ""
            if capture:
                os.close(write_fd)
                # Read all output from pipe
                log_bytes = b""
                while True:
                    chunk = os.read(read_fd, 4096)
                    if not chunk:
                        break
                    log_bytes += chunk
                os.close(read_fd)
                log = log_bytes.decode("utf-8", errors="replace")

            # Wait for child
            _, status = os.waitpid(pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)

            if result_path.exists():
                with open(result_path, "rb") as f:
                    instance = pickle.load(f)
                instance._Experiment__log = log
                with open(result_path, "wb") as f:
                    cloudpickle.dump(instance, f)
            else:
                # Child crashed before writing result
                instance._Experiment__error = f"ForkError: exited with code {exit_code}"
                instance._Experiment__log = log
                with open(result_path, "wb") as f:
                    cloudpickle.dump(instance, f)


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
        instance: "Experiment",
        result_path: Path,
        capture: bool = True,
        stash: bool = True,
    ) -> None:
        """Run experiment as a Ray task."""
        result_path.parent.mkdir(parents=True, exist_ok=True)

        @self._ray.remote
        def _run_experiment(instance, result_path_str, capture):
            """Ray remote function to execute experiment."""
            import io
            import sys
            import traceback
            from pathlib import Path
            import cloudpickle

            result_path = Path(result_path_str)
            log = ""

            # Capture output
            if capture:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

            try:
                instance.experiment()
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                instance._Experiment__error = error_msg
            finally:
                if capture:
                    log = sys.stdout.getvalue() + sys.stderr.getvalue()
                    sys.stdout, sys.stderr = old_stdout, old_stderr

            instance._Experiment__log = log
            with open(result_path, "wb") as f:
                cloudpickle.dump(instance, f)
            return instance

        # Submit task and wait for result
        future = _run_experiment.remote(instance, str(result_path), capture)
        try:
            self._ray.get(future)
        except Exception as e:
            # Task failed at Ray level
            instance._Experiment__error = f"RayError: {e}\n{traceback.format_exc()}"
            instance._Experiment__log = ""
            with open(result_path, "wb") as f:
                cloudpickle.dump(instance, f)


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
