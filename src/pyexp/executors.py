"""Executor classes for running functions in different isolation modes.

Each executor runs ``fn(*args, **kwargs)`` and returns a :class:`FnResult`
containing the return value and captured output. On error the traceback
is included in ``log`` and ``result`` is the sentinel :data:`MISSING`.

- InlineExecutor: Runs in the same process (no isolation)
- SubprocessExecutor: Runs in a subprocess using cloudpickle (cross-platform)
- ForkExecutor: Runs in a forked process (Unix only, fastest isolation)
- RayExecutor: Runs using Ray for distributed execution (requires ``pip install pyexp[ray]``)

Custom executors can be created by subclassing Executor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
import io
import os
import pickle
import subprocess
import sys
import tempfile
import traceback

import cloudpickle

# Valid executor names
ExecutorName = Literal["inline", "subprocess", "fork", "ray"]

# Sentinel to distinguish "fn returned None" from "fn raised"
MISSING = type("MISSING", (), {"__repr__": lambda self: "MISSING", "__bool__": lambda self: False})()


@dataclass
class FnResult:
    """Result of running a function via an executor.

    ``result`` is :data:`MISSING` when the function raised an exception.
    The traceback is appended to ``log``.
    """

    result: Any = field(default_factory=lambda: MISSING)
    log: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not MISSING


def _build_pythonpath(snapshot_path: Path | None = None) -> str:
    """Build PYTHONPATH for a subprocess.

    When *snapshot_path* is set, remap repo-local sys.path entries to their
    snapshot equivalents so that the subprocess imports from the snapshot.
    """
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
                    relative = resolved[len(git_root_str):]
                    remapped.append(snapshot_str + relative)
                else:
                    remapped.append(p)
            pythonpath = os.pathsep.join(remapped)
        except Exception:
            pythonpath = os.pathsep.join(sys.path)
    else:
        pythonpath = os.pathsep.join(sys.path)

    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath = pythonpath + os.pathsep + existing
    return pythonpath


class Executor(ABC):
    """Abstract base class for executors."""

    @abstractmethod
    def run(
        self,
        fn: Callable,
        *args: Any,
        capture: bool = True,
        snapshot_path: Path | None = None,
        **kwargs: Any,
    ) -> FnResult:
        """Run *fn* and return a :class:`FnResult`."""
        ...


class InlineExecutor(Executor):
    """Runs the function in the same process (no isolation)."""

    def run(
        self,
        fn: Callable,
        *args: Any,
        capture: bool = True,
        snapshot_path: Path | None = None,
        **kwargs: Any,
    ) -> FnResult:
        log = ""
        if capture:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

        try:
            result_value = fn(*args, **kwargs)
        except Exception:
            error_tb = traceback.format_exc()
            if capture:
                log = sys.stdout.getvalue() + sys.stderr.getvalue() + error_tb
                sys.stdout, sys.stderr = old_stdout, old_stderr
            else:
                log = error_tb
            return FnResult(log=log)
        else:
            if capture:
                log = sys.stdout.getvalue() + sys.stderr.getvalue()
                sys.stdout, sys.stderr = old_stdout, old_stderr
            return FnResult(result=result_value, log=log)


_WORKER_SCRIPT = """\
import pickle,sys,traceback,cloudpickle
p,r=sys.argv[1],sys.argv[2]
try:
    with open(p,"rb") as f:d=cloudpickle.load(f)
    v=d["fn"](*d["args"],**d["kwargs"])
    with open(r,"wb") as f:pickle.dump(v,f)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


class SubprocessExecutor(Executor):
    """Runs the function in an isolated subprocess using cloudpickle."""

    def run(
        self,
        fn: Callable,
        *args: Any,
        capture: bool = True,
        snapshot_path: Path | None = None,
        **kwargs: Any,
    ) -> FnResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result_path = tmp / "result.pkl"
            payload_path = tmp / "payload.pkl"

            with open(payload_path, "wb") as f:
                cloudpickle.dump({"fn": fn, "args": args, "kwargs": kwargs}, f)

            env = os.environ.copy()
            env["PYTHONPATH"] = _build_pythonpath(snapshot_path)

            cmd = [sys.executable, "-c", _WORKER_SCRIPT, str(payload_path), str(result_path)]

            if capture:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                log = proc.stdout or ""
            else:
                # Tee: show output live AND capture it.
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

            if result_path.exists():
                with open(result_path, "rb") as f:
                    result_value = pickle.load(f)
                return FnResult(result=result_value, log=log)

            # No result file → function raised (traceback is in log)
            return FnResult(log=log)


class ForkExecutor(Executor):
    """Runs the function in a forked process (Unix only, fastest isolation)."""

    def __init__(self):
        if not hasattr(os, "fork"):
            raise RuntimeError("ForkExecutor is only available on Unix systems")

    def run(
        self,
        fn: Callable,
        *args: Any,
        capture: bool = True,
        snapshot_path: Path | None = None,
        **kwargs: Any,
    ) -> FnResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.pkl"

            if capture:
                read_fd, write_fd = os.pipe()

            pid = os.fork()

            if pid == 0:
                # Child process
                try:
                    if capture:
                        os.close(read_fd)
                        os.dup2(write_fd, 1)
                        os.dup2(write_fd, 2)
                        os.close(write_fd)
                        sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", 0), line_buffering=True)
                        sys.stderr = io.TextIOWrapper(os.fdopen(2, "wb", 0), line_buffering=True)

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
                            pass

                    result_value = fn(*args, **kwargs)

                    with open(result_path, "wb") as f:
                        pickle.dump(result_value, f)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)
                except Exception:
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(1)
            else:
                # Parent process
                log = ""
                if capture:
                    os.close(write_fd)
                    log_bytes = b""
                    while True:
                        chunk = os.read(read_fd, 4096)
                        if not chunk:
                            break
                        log_bytes += chunk
                    os.close(read_fd)
                    log = log_bytes.decode("utf-8", errors="replace")

                _, status = os.waitpid(pid, 0)

                if result_path.exists():
                    with open(result_path, "rb") as f:
                        result_value = pickle.load(f)
                    return FnResult(result=result_value, log=log)

                return FnResult(log=log)


class RayExecutor(Executor):
    """Runs the function using Ray for distributed execution.

    Args:
        address: Ray cluster address. Use ``"auto"`` to connect to an existing
            cluster, or ``None`` to start a local instance.
        runtime_env: Runtime environment configuration for distributing code.
        num_cpus: Number of CPUs to use.
        num_gpus: Number of GPUs to use.
        **ray_init_kwargs: Additional arguments passed to ``ray.init()``.
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
        *args: Any,
        capture: bool = True,
        snapshot_path: Path | None = None,
        **kwargs: Any,
    ) -> FnResult:
        @self._ray.remote
        def _run_fn(fn, args, kwargs, capture):
            import io
            import sys
            import traceback

            log = ""
            if capture:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

            try:
                result_value = fn(*args, **kwargs)
            except Exception:
                error_tb = traceback.format_exc()
                if capture:
                    log = sys.stdout.getvalue() + sys.stderr.getvalue() + error_tb
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                else:
                    log = error_tb
                return {"log": log}
            else:
                if capture:
                    log = sys.stdout.getvalue() + sys.stderr.getvalue()
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                return {"result": result_value, "log": log}

        try:
            data = self._ray.get(_run_fn.remote(fn, args, kwargs, capture))
            if "result" in data:
                return FnResult(result=data["result"], log=data.get("log", ""))
            return FnResult(log=data.get("log", ""))
        except Exception as e:
            return FnResult(log=f"RayError: {e}\n{traceback.format_exc()}")


# Registry of built-in executors
EXECUTORS: dict[str, type[Executor]] = {
    "inline": InlineExecutor,
    "subprocess": SubprocessExecutor,
    "fork": ForkExecutor,
    "ray": RayExecutor,
}


def get_executor(executor: ExecutorName | Executor) -> Executor:
    """Get an executor instance from a string name or existing instance."""
    if isinstance(executor, Executor):
        return executor

    if isinstance(executor, str):
        if executor not in EXECUTORS:
            available = ", ".join(EXECUTORS.keys())
            raise ValueError(f"Unknown executor '{executor}'. Available: {available}")
        return EXECUTORS[executor]()

    raise TypeError(f"executor must be str or Executor, got {type(executor).__name__}")
