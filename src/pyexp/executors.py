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
    from .runner import Result
    from .config import Runs

# Valid executor names
ExecutorName = Literal["inline", "subprocess", "fork", "ray"]


class Executor(ABC):
    """Abstract base class for experiment executors.

    Subclass this to implement custom execution strategies.
    """

    @abstractmethod
    def run(
        self,
        fn: Callable,
        experiment: "Result",
        deps: "Runs | None",
        result_path: Path,
        *,
        capture: bool = True,
        stash: bool = True,
        snapshot_path: Path | None = None,
        wants_out: bool = False,
        wants_deps: bool = False,
    ) -> None:
        """Run a single experiment and pickle the result.

        Args:
            fn: The experiment function to call.
            experiment: The Result dataclass (cfg/name/out already set).
            deps: Dependency experiments (Runs), or None.
            result_path: Path where the pickled experiment should be saved.
            capture: If True (default), capture output. If False, show output live.
            stash: If True, capture git commit hash at the start of the experiment.
            snapshot_path: If set, run experiment from this snapshot directory.
            wants_out: If True, pass out as second arg to fn.
            wants_deps: If True, pass deps as third arg to fn.

        The executor should:
        1. Call fn(cfg) / fn(cfg, out) / fn(cfg, out, deps)
        2. Store return value in experiment.result
        3. On error: set experiment.error
        4. Set experiment.log with captured stdout/stderr
        5. Pickle the experiment to result_path
        """
        pass


def _call_fn(fn, experiment, deps, wants_out, wants_deps, result_path=None):
    """Call the experiment function with the right arguments and store result."""
    if result_path is not None:
        experiment.out = result_path.parent
    if wants_deps:
        experiment.result = fn(experiment.cfg, experiment.out, deps)
    elif wants_out:
        experiment.result = fn(experiment.cfg, experiment.out)
    else:
        experiment.result = fn(experiment.cfg)


class InlineExecutor(Executor):
    """Executor that runs experiments in the same process.

    This provides no isolation - crashes will affect the main process.
    Useful for debugging or when isolation overhead is not desired.
    """

    def run(
        self,
        fn: Callable,
        experiment: "Result",
        deps: "Runs | None",
        result_path: Path,
        *,
        capture: bool = True,
        stash: bool = True,
        snapshot_path: Path | None = None,
        wants_out: bool = False,
        wants_deps: bool = False,
    ) -> None:
        """Run experiment inline and cache result."""
        import io
        import sys

        experiment_dir = result_path.parent
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Capture output if requested
        log = ""
        error_msg = None
        result_value = None
        if capture:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

        try:
            _call_fn(fn, experiment, deps, wants_out, wants_deps, result_path)
            result_value = experiment.result
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            experiment.error = error_msg
            if not capture:
                print(error_msg, file=sys.stderr)
        finally:
            if capture:
                log = sys.stdout.getvalue() + sys.stderr.getvalue()
                sys.stdout, sys.stderr = old_stdout, old_stderr

        if error_msg and error_msg not in log:
            log = log + error_msg if log else error_msg
        experiment.log = log
        experiment.finished = True

        # Save only the result value
        if result_value is not None:
            with open(result_path, "wb") as f:
                pickle.dump(result_value, f)

        # Save error if any
        if error_msg:
            (experiment_dir / "error.txt").write_text(error_msg)

        # Save log
        (experiment_dir / "log.out").write_text(log)

        (experiment_dir / ".finished").touch()


class SubprocessExecutor(Executor):
    """Executor that runs experiments in isolated subprocesses using cloudpickle.

    This provides strong isolation - crashes (including segfaults) won't affect
    the main process. Works cross-platform (Windows, macOS, Linux).
    """

    def run(
        self,
        fn: Callable,
        experiment: "Result",
        deps: "Runs | None",
        result_path: Path,
        *,
        capture: bool = True,
        stash: bool = True,
        snapshot_path: Path | None = None,
        wants_out: bool = False,
        wants_deps: bool = False,
    ) -> None:
        """Run experiment in subprocess via cloudpickle serialization."""
        experiment_dir = result_path.parent
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create payload (deps not included — worker resolves them from disk)
        payload = {
            "fn": fn,
            "experiment": experiment,
            "result_path": str(result_path),
            "stash": stash,
            "wants_out": wants_out,
            "wants_deps": wants_deps,
        }

        # Write payload to temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            payload_path = f.name
            cloudpickle.dump(payload, f)

        try:
            # Run worker subprocess with inherited sys.path via PYTHONPATH
            env = os.environ.copy()

            # When snapshot_path is set, remap repo-local sys.path entries
            # to their snapshot equivalents
            if snapshot_path is not None:
                try:
                    from .utils import _find_git_root

                    git_root = _find_git_root()
                    git_root_str = str(git_root.resolve())
                    snapshot_str = str(snapshot_path.resolve())

                    remapped = []
                    for p in sys.path:
                        resolved = str(Path(p).resolve()) if p else ""
                        if resolved.startswith(git_root_str):
                            relative = resolved[len(git_root_str) :]
                            remapped.append(snapshot_str + relative)
                        else:
                            remapped.append(p)
                    pythonpath = os.pathsep.join(remapped)
                except Exception:
                    # Fall back to normal sys.path if remapping fails
                    pythonpath = os.pathsep.join(sys.path)
            else:
                pythonpath = os.pathsep.join(sys.path)

            if env.get("PYTHONPATH"):
                pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
            env["PYTHONPATH"] = pythonpath

            cmd = [sys.executable, "-m", "pyexp.worker", payload_path]
            if capture:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                log = proc.stdout
                stderr = proc.stderr
            else:
                # Tee: show output live AND capture it for the log file.
                # Merge stderr into stdout so tqdm bars (which write to
                # stderr) are visible in real time.
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                log_parts = []
                fd = proc.stdout.fileno()
                while True:
                    chunk = os.read(fd, 8192)
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    log_parts.append(chunk)
                proc.wait()
                log = b"".join(log_parts).decode("utf-8", errors="replace")
                stderr = ""

            # Save log file (stdout + stderr combined)
            full_log = (log + stderr) if stderr else log
            (experiment_dir / "log.out").write_text(full_log)

            finished_marker = experiment_dir / ".finished"
            error_path = experiment_dir / "error.txt"

            # Check if worker completed (wrote .finished marker)
            if finished_marker.exists():
                # Worker completed, read any error it may have written
                if error_path.exists():
                    experiment.error = error_path.read_text()
                if result_path.exists():
                    with open(result_path, "rb") as f:
                        experiment.result = pickle.load(f)
                experiment.log = full_log
                experiment.finished = True
            else:
                # Subprocess crashed before completing — build error from
                # captured output so the traceback is always visible.
                error_msg = f"SubprocessError: exited with code {proc.returncode}"
                output = stderr.strip() or log.strip()
                if output:
                    error_msg += f"\n\n{output}"
                experiment.error = error_msg
                experiment.log = full_log
                experiment.finished = True
                # Write error file
                error_path.write_text(error_msg)
                finished_marker.touch()
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
        fn: Callable,
        experiment: "Result",
        deps: "Runs | None",
        result_path: Path,
        *,
        capture: bool = True,
        stash: bool = True,
        snapshot_path: Path | None = None,
        wants_out: bool = False,
        wants_deps: bool = False,
    ) -> None:
        """Run experiment in a forked process."""
        experiment_dir = result_path.parent
        experiment_dir.mkdir(parents=True, exist_ok=True)

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

                # Best-effort sys.path manipulation for snapshot
                if snapshot_path is not None:
                    try:
                        from .utils import _find_git_root

                        git_root = _find_git_root()
                        git_root_str = str(git_root.resolve())
                        snapshot_str = str(snapshot_path.resolve())

                        new_path = []
                        for p in sys.path:
                            resolved = str(Path(p).resolve()) if p else ""
                            if resolved.startswith(git_root_str):
                                relative = resolved[len(git_root_str):]
                                new_path.append(snapshot_str + relative)
                            else:
                                new_path.append(p)
                        sys.path[:] = new_path
                    except Exception:
                        pass  # Best-effort: continue without remapping

                _call_fn(fn, experiment, deps, wants_out, wants_deps, result_path)

                # Save only the result value
                with open(result_path, "wb") as f:
                    pickle.dump(experiment.result, f)
                (experiment_dir / ".finished").touch()
                os._exit(0)
            except Exception as e:
                # Write error information
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                if not capture:
                    print(error_msg, file=sys.stderr)
                try:
                    (experiment_dir / "error.txt").write_text(error_msg)
                    (experiment_dir / ".finished").touch()
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

            finished_marker = experiment_dir / ".finished"
            error_path = experiment_dir / "error.txt"

            # Check if child completed
            if finished_marker.exists():
                # Child completed, read any error it may have written
                if error_path.exists():
                    error_msg = error_path.read_text()
                    experiment.error = error_msg
                    if error_msg not in log:
                        log = log + error_msg if log else error_msg
                if result_path.exists():
                    with open(result_path, "rb") as f:
                        experiment.result = pickle.load(f)
                experiment.out = experiment_dir
                experiment.log = log
                experiment.finished = True
            else:
                # Child crashed before completing
                error_msg = f"ForkError: exited with code {exit_code}"
                if error_msg not in log:
                    log = log + error_msg if log else error_msg
                experiment.error = error_msg
                experiment.log = log
                experiment.finished = True
                experiment.out = experiment_dir
                # Write error file
                error_path.write_text(error_msg)
                finished_marker.touch()

            # Save log file
            (experiment_dir / "log.out").write_text(log)


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
        fn: Callable,
        experiment: "Result",
        deps: "Runs | None",
        result_path: Path,
        *,
        capture: bool = True,
        stash: bool = True,
        snapshot_path: Path | None = None,
        wants_out: bool = False,
        wants_deps: bool = False,
    ) -> None:
        """Run experiment as a Ray task."""
        experiment_dir = result_path.parent
        experiment_dir.mkdir(parents=True, exist_ok=True)

        @self._ray.remote
        def _run_experiment(fn, experiment, deps, result_path_str, capture, wants_out, wants_deps):
            """Ray remote function to execute experiment."""
            import io
            import sys
            import pickle
            import traceback
            from pathlib import Path

            result_path = Path(result_path_str)
            experiment_dir = result_path.parent
            experiment.out = experiment_dir
            log = ""
            error_msg = None
            result_value = None

            # Capture output
            if capture:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

            try:
                if wants_deps:
                    result_value = fn(experiment.cfg, experiment.out, deps)
                elif wants_out:
                    result_value = fn(experiment.cfg, experiment.out)
                else:
                    result_value = fn(experiment.cfg)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            finally:
                if capture:
                    log = sys.stdout.getvalue() + sys.stderr.getvalue()
                    sys.stdout, sys.stderr = old_stdout, old_stderr

            # Save only the result value
            if result_value is not None:
                with open(result_path, "wb") as f:
                    pickle.dump(result_value, f)

            # Save error if any
            if error_msg:
                (experiment_dir / "error.txt").write_text(error_msg)

            # Save log
            (experiment_dir / "log.out").write_text(log)

            (experiment_dir / ".finished").touch()
            return {"result": result_value, "error": error_msg, "log": log}

        # Submit task and wait for result
        future = _run_experiment.remote(fn, experiment, deps, str(result_path), capture, wants_out, wants_deps)
        try:
            result_data = self._ray.get(future)
            experiment.result = result_data["result"]
            experiment.error = result_data["error"]
            experiment.log = result_data["log"]
            experiment.finished = True
        except Exception as e:
            # Task failed at Ray level
            error_msg = f"RayError: {e}\n{traceback.format_exc()}"
            experiment.error = error_msg
            experiment.log = ""
            experiment.finished = True
            (experiment_dir / "error.txt").write_text(error_msg)
            (experiment_dir / "log.out").write_text("")
            (experiment_dir / ".finished").touch()


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
