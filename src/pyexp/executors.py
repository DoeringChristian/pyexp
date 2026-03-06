"""Executor classes for running functions in different isolation modes.

Each executor runs ``fn(*args, **kwargs)`` and returns a :class:`FnFuture`
(a ``concurrent.futures.Future`` subclass) carrying the return value, any
exception, and captured output.

- InlineExecutor: Runs in the same process (no isolation)
- SubprocessExecutor: Runs in a subprocess using cloudpickle (cross-platform)
- ForkExecutor: Runs in a forked process (Unix only, fastest isolation)
- RayExecutor: Runs using Ray for distributed execution (requires ``pip install pyexp[ray]``)

Custom executors can be created by subclassing Executor.
"""

from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable
import concurrent.futures
import io
import logging
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid

log = logging.getLogger(__name__)

import cloudpickle

# Valid executor names
ExecutorName = Literal["inline", "subprocess", "fork", "ray", "ssh"]

# Sentinel to distinguish "fn returned None" from "fn raised"
MISSING = type(
    "MISSING", (), {"__repr__": lambda self: "MISSING", "__bool__": lambda self: False}
)()


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


class FnFuture(Future):
    """A ``concurrent.futures.Future`` with an additional ``log`` attribute.

    ``log`` contains captured stdout/stderr (and traceback on error).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log: str = ""

    def wait(self, timeout: float | None = None) -> FnResult:
        """Wait for the future to complete and return a :class:`FnResult`.

        Unlike :meth:`result`, this never raises — errors are captured in the
        returned ``FnResult`` (with ``ok == False`` and traceback in ``log``).
        """
        self._condition.acquire()
        try:
            self._condition.wait_for(lambda: self._state in ("FINISHED",), timeout=timeout)
        finally:
            self._condition.release()

        exc = self.exception(timeout=0)
        if exc is not None:
            return FnResult(result=MISSING, log=self.log)
        return FnResult(result=self.result(timeout=0), log=self.log)


@dataclass
class SshHost:
    """A remote host for SSH-based execution.

    Args:
        host: SSH destination (e.g. "user@gpu1").
        max_tasks: Maximum concurrent tasks on this host.
        setup: Shell command run before each task (e.g. "source ~/env/bin/activate").
        work_dir: Remote work directory override.
    """

    host: str
    max_tasks: int = 1
    setup: str | None = None
    work_dir: str | None = None


@runtime_checkable
class Provisioner(Protocol):
    """Protocol for environment provisioners."""

    def provision_commands(self, work_dir: str) -> list[str]: ...


@dataclass
class UvProvisioner:
    """Provisions a remote environment using uv."""

    python: str = "3.12"
    requirements: str | None = None
    venv_path: str = ".venv"
    extra_packages: list[str] = field(default_factory=list)

    def provision_commands(self, work_dir: str) -> list[str]:
        venv = f"{work_dir}/{self.venv_path}"
        cmds = [
            # Bootstrap uv if not already installed
            'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH")',
            f'export PATH="$HOME/.local/bin:$PATH" && cd {work_dir} && uv venv {shlex.quote(venv)} --python {shlex.quote(self.python)}',
        ]
        uv_prefix = 'export PATH="$HOME/.local/bin:$PATH" && '
        if self.requirements:
            cmds.append(
                f"{uv_prefix}cd {work_dir} && uv pip install --python {shlex.quote(venv + '/bin/python')} -r {shlex.quote(self.requirements)}"
            )
        packages = ["cloudpickle"] + self.extra_packages
        cmds.append(
            f"{uv_prefix}cd {work_dir} && uv pip install --python {shlex.quote(venv + '/bin/python')} {' '.join(shlex.quote(p) for p in packages)}"
        )
        return cmds


@dataclass
class PipProvisioner:
    """Provisions a remote environment using pip."""

    python: str = "python3"
    requirements: str | None = None
    venv_path: str = ".venv"
    extra_packages: list[str] = field(default_factory=list)

    def provision_commands(self, work_dir: str) -> list[str]:
        venv = f"{work_dir}/{self.venv_path}"
        cmds = [
            f"{shlex.quote(self.python)} -m venv {shlex.quote(venv)}",
        ]
        if self.requirements:
            cmds.append(
                f"{shlex.quote(venv + '/bin/pip')} install -r {shlex.quote(self.requirements)}"
            )
        packages = ["cloudpickle"] + self.extra_packages
        cmds.append(
            f"{shlex.quote(venv + '/bin/pip')} install {' '.join(shlex.quote(p) for p in packages)}"
        )
        return cmds


@dataclass
class PixiProvisioner:
    """Provisions a remote environment using pixi."""

    manifest: str | None = None
    environment: str | None = None
    extra_packages: list[str] = field(default_factory=list)

    def provision_commands(self, work_dir: str) -> list[str]:
        pixi_prefix = 'export PATH="$HOME/.pixi/bin:$PATH" && '
        cmds = [
            # Bootstrap pixi if not already installed
            'command -v pixi >/dev/null 2>&1 || curl -fsSL https://pixi.sh/install.sh | bash',
        ]
        install_cmd = f"{pixi_prefix}cd {work_dir} && pixi install"
        if self.manifest:
            install_cmd += f" --manifest-path {shlex.quote(self.manifest)}"
        if self.environment:
            install_cmd += f" -e {shlex.quote(self.environment)}"
        cmds.append(install_cmd)

        packages = ["cloudpickle"] + self.extra_packages
        if packages:
            add_cmd = f"{pixi_prefix}cd {work_dir} && pixi add {' '.join(shlex.quote(p) for p in packages)}"
            if self.environment:
                add_cmd += f" -e {shlex.quote(self.environment)}"
            cmds.append(add_cmd)
        return cmds


def _detect_provisioner(search_dir: Path | None = None) -> "UvProvisioner | PixiProvisioner | PipProvisioner":
    """Detect which Python package manager is used in *search_dir* (default: cwd).

    Detection order (first match wins):
    1. ``uv.lock`` → :class:`UvProvisioner` (bootstraps uv on remote if needed)
    2. ``pixi.lock`` or ``pixi.toml`` → :class:`PixiProvisioner`
    3. ``requirements.txt`` → :class:`PipProvisioner`
    4. ``pyproject.toml`` → :class:`UvProvisioner` (assume uv as modern default)
    5. Fallback → :class:`PipProvisioner` (no requirements file)
    """
    d = search_dir or Path.cwd()

    # --- uv ---
    if (d / "uv.lock").exists():
        python = "3.12"
        pv = d / ".python-version"
        if pv.exists():
            python = pv.read_text().strip()
        reqs = "requirements.txt" if (d / "requirements.txt").exists() else None
        return UvProvisioner(python=python, requirements=reqs)

    # --- pixi ---
    if (d / "pixi.lock").exists() or (d / "pixi.toml").exists():
        manifest = None
        if (d / "pixi.toml").exists():
            manifest = "pixi.toml"
        return PixiProvisioner(manifest=manifest)

    # --- pip (explicit requirements.txt) ---
    if (d / "requirements.txt").exists():
        return PipProvisioner(requirements="requirements.txt")

    # --- pyproject.toml without lock → assume uv ---
    if (d / "pyproject.toml").exists():
        python = "3.12"
        pv = d / ".python-version"
        if pv.exists():
            python = pv.read_text().strip()
        return UvProvisioner(python=python)

    # --- fallback ---
    return PipProvisioner()


@dataclass
class AutoProvisioner:
    """Automatically detects and delegates to the right provisioner.

    Looks at the current working directory for lock files and config files
    to determine which package manager (uv, pixi, pip) is in use.
    """

    search_dir: Path | None = None

    def __post_init__(self):
        self._delegate: UvProvisioner | PixiProvisioner | PipProvisioner | None = None

    @property
    def _provisioner(self) -> "UvProvisioner | PixiProvisioner | PipProvisioner":
        if self._delegate is None:
            self._delegate = _detect_provisioner(self.search_dir)
        return self._delegate

    @property
    def requirements(self) -> str | None:
        return getattr(self._provisioner, "requirements", None)

    @property
    def manifest(self) -> str | None:
        return getattr(self._provisioner, "manifest", None)

    @property
    def venv_path(self) -> str | None:
        return getattr(self._provisioner, "venv_path", None)

    @property
    def environment(self) -> str | None:
        return getattr(self._provisioner, "environment", None)

    def provision_commands(self, work_dir: str) -> list[str]:
        return self._provisioner.provision_commands(work_dir)


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
                    relative = resolved[len(git_root_str) :]
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


class Executor(concurrent.futures.Executor):
    """Abstract base class for executors."""

    def __init__(
        self, *, snapshot: bool = False, capture: bool = True, max_workers: int = 1
    ):
        self._snapshot = snapshot
        self._capture = capture
        self._max_workers = max_workers
        self._snapshot_path: Path | None = None
        self._snapshot_hash: str | None = None

        if snapshot:
            from .utils import content_hash, package_files

            h = content_hash()
            dest = Path.cwd() / ".snapshot" / h
            if not dest.exists():
                package_files(dest)
            self._snapshot_path = dest
            self._snapshot_hash = h

    @abstractmethod
    def submit(self, fn, /, *args, **kwargs) -> FnFuture:
        """Submit *fn* for execution and return a :class:`FnFuture`."""
        ...

    def wait(
        self, futures: list[FnFuture], timeout: float | None = None
    ) -> list[FnResult]:
        """Wait for all *futures* to complete and return their :class:`FnResult`\\s.

        Never raises — errors are captured in each ``FnResult``.
        """
        return [f.wait(timeout=timeout) for f in futures]

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Shutdown the executor. No-op in base class."""
        pass


class InlineExecutor(Executor):
    """Runs the function in the same process (no isolation)."""

    def __init__(
        self, *, snapshot: bool = False, capture: bool = True, max_workers: int = 1
    ):
        super().__init__(snapshot=snapshot, capture=capture, max_workers=max_workers)
        self._pool = ThreadPoolExecutor(max_workers)
        self._io_lock = threading.Lock()

    def submit(self, fn, /, *args, **kwargs) -> FnFuture:
        future = FnFuture()
        capture = self._capture
        io_lock = self._io_lock

        def _run():
            log = ""
            if capture:
                # Hold lock across entire execution to prevent concurrent
                # sys.stdout/stderr swaps from interleaving.
                with io_lock:
                    old_stdout, old_stderr = sys.stdout, sys.stderr
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    try:
                        result_value = fn(*args, **kwargs)
                    except Exception as exc:
                        error_tb = traceback.format_exc()
                        log = sys.stdout.getvalue() + sys.stderr.getvalue() + error_tb
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                        future.log = log
                        future.set_exception(exc)
                        return
                    else:
                        log = sys.stdout.getvalue() + sys.stderr.getvalue()
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                        future.log = log
                        future.set_result(result_value)
            else:
                try:
                    result_value = fn(*args, **kwargs)
                except Exception as exc:
                    future.log = traceback.format_exc()
                    future.set_exception(exc)
                else:
                    future.log = ""
                    future.set_result(result_value)

        self._pool.submit(_run)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


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

    def __init__(
        self, *, snapshot: bool = False, capture: bool = True, max_workers: int = 1
    ):
        super().__init__(snapshot=snapshot, capture=capture, max_workers=max_workers)
        self._pool = ThreadPoolExecutor(max_workers)

    def submit(self, fn, /, *args, **kwargs) -> FnFuture:
        capture = self._capture
        snapshot_path = self._snapshot_path
        future = FnFuture()

        def _run():
            tmpdir = tempfile.mkdtemp()
            try:
                tmp = Path(tmpdir)
                result_path = tmp / "result.pkl"
                payload_path = tmp / "payload.pkl"

                with open(payload_path, "wb") as f:
                    cloudpickle.dump({"fn": fn, "args": args, "kwargs": kwargs}, f)

                env = os.environ.copy()
                env["PYTHONPATH"] = _build_pythonpath(snapshot_path)

                cmd = [
                    sys.executable,
                    "-c",
                    _WORKER_SCRIPT,
                    str(payload_path),
                    str(result_path),
                ]

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
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    log = proc.stdout or ""
                    # Print output after completion to avoid interleaving
                    sys.stdout.write(log)
                    sys.stdout.flush()

                future.log = log

                if result_path.exists():
                    with open(result_path, "rb") as f:
                        result_value = pickle.load(f)
                    future.set_result(result_value)
                else:
                    future.set_exception(RuntimeError(log))
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        self._pool.submit(_run)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class ForkExecutor(Executor):
    """Runs the function in a forked process (Unix only, fastest isolation)."""

    def __init__(
        self, *, snapshot: bool = False, capture: bool = True, max_workers: int = 1
    ):
        if not hasattr(os, "fork"):
            raise RuntimeError("ForkExecutor is only available on Unix systems")
        super().__init__(snapshot=snapshot, capture=capture, max_workers=max_workers)
        self._pool = ThreadPoolExecutor(max_workers)

    def submit(self, fn, /, *args, **kwargs) -> FnFuture:
        capture = self._capture
        snapshot_path = self._snapshot_path
        future = FnFuture()

        def _run():
            tmpdir = tempfile.mkdtemp()
            try:
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
                            sys.stdout = io.TextIOWrapper(
                                os.fdopen(1, "wb", 0), line_buffering=True
                            )
                            sys.stderr = io.TextIOWrapper(
                                os.fdopen(2, "wb", 0), line_buffering=True
                            )

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
                                        relative = resolved[len(git_root_str) :]
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

                    future.log = log

                    if result_path.exists():
                        with open(result_path, "rb") as f:
                            result_value = pickle.load(f)
                        future.set_result(result_value)
                    else:
                        future.set_exception(RuntimeError(log))
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        self._pool.submit(_run)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


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
        snapshot: bool = False,
        capture: bool = True,
        max_workers: int = 1,
        **ray_init_kwargs,
    ):
        super().__init__(snapshot=snapshot, capture=capture, max_workers=max_workers)
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

    def submit(self, fn, /, *args, **kwargs) -> FnFuture:
        capture = self._capture
        ray = self._ray

        @ray.remote
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

        future = FnFuture()
        ref = _run_fn.remote(fn, args, kwargs, capture)

        def _resolve():
            try:
                data = ray.get(ref)
                future.log = data.get("log", "")
                if "result" in data:
                    future.set_result(data["result"])
                else:
                    future.set_exception(RuntimeError(future.log))
            except Exception as e:
                future.log = f"RayError: {e}\n{traceback.format_exc()}"
                future.set_exception(e)

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass  # Ray manages its own pool


class SshExecutor(Executor):
    """Runs functions on remote hosts via SSH with host-pool concurrency control.

    Tasks are distributed across a pool of SSH hosts using semaphore-gated
    round-robin scheduling. Each host can run up to ``max_tasks`` concurrent
    tasks. An optional :class:`Provisioner` sets up the remote environment
    once per host.
    """

    def __init__(
        self,
        hosts: list[SshHost | str],
        provision: Provisioner | Literal["auto"] | None = "auto",
        setup: str | None = None,
        work_dir: str | None = None,
        ssh_options: list[str] | None = None,
        snapshot: bool = False,
        capture: bool = True,
        max_workers: int = 1,  # ignored; total = sum of host max_tasks
    ):
        if not hosts:
            raise ValueError("SshExecutor requires at least one SshHost")
        super().__init__(snapshot=snapshot, capture=capture, max_workers=max_workers)
        self._hosts = [SshHost(h) if isinstance(h, str) else h for h in hosts]
        self._provision: Provisioner | None = AutoProvisioner() if provision == "auto" else provision
        self._default_setup = setup
        self._default_work_dir = work_dir or f"/tmp/pyexp-{uuid.uuid4().hex[:12]}"
        self._ssh_options = ssh_options or []

        total_slots = sum(h.max_tasks for h in self._hosts)
        self._pool = ThreadPoolExecutor(total_slots)

        # Per-host concurrency semaphores
        self._semaphores: dict[str, threading.Semaphore] = {
            h.host: threading.Semaphore(h.max_tasks) for h in self._hosts
        }
        self._host_index = 0
        self._host_lock = threading.Lock()

        # Provisioning state
        self._provisioned: dict[str, bool] = {}
        self._provision_lock = threading.Lock()

    def _work_dir_for(self, host: SshHost) -> str:
        return host.work_dir or self._default_work_dir

    def _setup_for(self, host: SshHost) -> str | None:
        return host.setup if host.setup is not None else self._default_setup

    # --- SSH command builders ---

    def _ssh_cmd(self, host: str, command: str) -> list[str]:
        return ["ssh"] + self._ssh_options + [host, command]

    def _scp_to(self, local: str, host: str, remote: str) -> list[str]:
        return ["scp"] + self._ssh_options + [local, f"{host}:{remote}"]

    def _scp_from(self, host: str, remote: str, local: str) -> list[str]:
        return ["scp"] + self._ssh_options + [f"{host}:{remote}", local]

    def _run_ssh(
        self, host: str, command: str, *, check: bool = False
    ) -> subprocess.CompletedProcess:
        log.debug("ssh %s: %s", host, command)
        result = subprocess.run(
            self._ssh_cmd(host, command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            log.warning("ssh %s failed (rc=%d): %s", host, result.returncode, result.stdout.rstrip())
            if check:
                raise RuntimeError(
                    f"SSH command failed on {host} (rc={result.returncode}):\n"
                    f"  command: {command}\n"
                    f"  output: {result.stdout}"
                )
        return result

    def _run_scp(
        self, cmd: list[str], *, check: bool = False
    ) -> subprocess.CompletedProcess:
        log.debug("scp: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            log.warning("scp failed (rc=%d): %s", result.returncode, result.stdout.rstrip())
            if check:
                raise RuntimeError(
                    f"SCP command failed (rc={result.returncode}):\n"
                    f"  command: {' '.join(cmd)}\n"
                    f"  output: {result.stdout}"
                )
        return result

    # --- Host selection ---

    def _acquire_host(self) -> SshHost:
        """Acquire a host slot using semaphore-gated round-robin."""
        n = len(self._hosts)
        # Try non-blocking first
        with self._host_lock:
            start = self._host_index
            for i in range(n):
                idx = (start + i) % n
                host = self._hosts[idx]
                if self._semaphores[host.host].acquire(blocking=False):
                    self._host_index = (idx + 1) % n
                    log.debug("acquired host %s", host.host)
                    return host
            # None available non-blocking; pick next in rotation and block
            block_idx = self._host_index
            self._host_index = (block_idx + 1) % n
            block_host = self._hosts[block_idx]

        # Block outside the lock
        log.debug("waiting for host %s", block_host.host)
        self._semaphores[block_host.host].acquire(blocking=True)
        log.debug("acquired host %s (after wait)", block_host.host)
        return block_host

    def _release_host(self, host: SshHost) -> None:
        self._semaphores[host.host].release()
        log.debug("released host %s", host.host)

    # --- Provisioning ---

    def _provision_host(self, host: SshHost) -> None:
        """Provision a host (idempotent, runs once per host)."""
        if self._provisioned.get(host.host):
            return
        with self._provision_lock:
            if self._provisioned.get(host.host):
                return
            log.info("provisioning %s", host.host)
            work_dir = self._work_dir_for(host)
            self._run_ssh(host.host, f"mkdir -p {shlex.quote(work_dir)}", check=True)

            # Copy snapshot if enabled
            if self._snapshot_path is not None:
                log.info("copying snapshot to %s:%s", host.host, work_dir)
                scp_cmd = (
                    ["scp", "-r"]
                    + self._ssh_options
                    + [str(self._snapshot_path), f"{host.host}:{work_dir}/snapshot"]
                )
                self._run_scp(scp_cmd, check=True)

            # Copy project files needed by the provisioner
            if self._provision is not None:
                # Pixi needs the whole project tree (path deps in manifest)
                if getattr(self._provision, "manifest", None) is not None:
                    log.info("copying project tree to %s:%s", host.host, work_dir)
                    scp_cmd = (
                        ["scp", "-r"]
                        + self._ssh_options
                        + [str(Path.cwd()) + "/.", f"{host.host}:{work_dir}"]
                    )
                    self._run_scp(scp_cmd, check=True)
                else:
                    # Copy specific files for non-pixi provisioners
                    for attr in ("requirements",):
                        fname = getattr(self._provision, attr, None)
                        if fname and Path(fname).exists():
                            log.debug("copying %s to %s:%s", fname, host.host, work_dir)
                            self._run_scp(
                                self._scp_to(fname, host.host, f"{work_dir}/{fname}"),
                                check=True,
                            )
                    # Common project metadata
                    for extra in ("pyproject.toml",):
                        if Path(extra).exists():
                            log.debug("copying %s to %s:%s", extra, host.host, work_dir)
                            self._run_scp(
                                self._scp_to(extra, host.host, f"{work_dir}/{extra}"),
                                check=True,
                            )

                log.info("running provision commands on %s", host.host)
                for cmd in self._provision.provision_commands(work_dir):
                    self._run_ssh(host.host, cmd, check=True)

            self._provisioned[host.host] = True
            log.info("provisioned %s", host.host)

    # --- Submit ---

    def submit(self, fn, /, *args, **kwargs) -> FnFuture:
        capture = self._capture
        future = FnFuture()

        def _run():
            task_id = uuid.uuid4().hex[:16]
            host = self._acquire_host()
            tmpdir = tempfile.mkdtemp()
            try:
                self._provision_host(host)
                work_dir = self._work_dir_for(host)
                task_dir = f"{work_dir}/tasks/{task_id}"
                log.info("task %s: submitting to %s", task_id, host.host)
                self._run_ssh(host.host, f"mkdir -p {shlex.quote(task_dir)}")

                # Serialize payload
                payload_local = os.path.join(tmpdir, "payload.pkl")
                with open(payload_local, "wb") as f:
                    cloudpickle.dump({"fn": fn, "args": args, "kwargs": kwargs}, f)

                # Upload payload
                self._run_scp(
                    self._scp_to(payload_local, host.host, f"{task_dir}/payload.pkl")
                )

                # Build remote command
                setup = self._setup_for(host)
                payload_path = f"{task_dir}/payload.pkl"
                result_path = f"{task_dir}/result.pkl"

                # Determine python executable
                provision = self._provision
                venv_path = getattr(provision, "venv_path", None) if provision else None
                pixi_env = getattr(provision, "environment", None) if provision else None
                if venv_path:
                    python_bin = f"{work_dir}/{venv_path}/bin/python"
                elif provision is not None and getattr(provision, "manifest", None) is not None:
                    env_flag = f" -e {shlex.quote(pixi_env)}" if pixi_env else ""
                    python_bin = f"pixi run{env_flag} python"
                else:
                    python_bin = "python3"

                script = shlex.quote(
                    f"import pickle,sys,traceback,cloudpickle\n"
                    f"p,r=sys.argv[1],sys.argv[2]\n"
                    f"try:\n"
                    f"    with open(p,'rb') as f:d=cloudpickle.load(f)\n"
                    f"    v=d['fn'](*d['args'],**d['kwargs'])\n"
                    f"    with open(r,'wb') as f:pickle.dump(v,f)\n"
                    f"except Exception:\n"
                    f"    traceback.print_exc()\n"
                    f"    sys.exit(1)\n"
                )

                parts = []
                parts.append(f"cd {shlex.quote(work_dir)}")
                if setup:
                    parts.append(setup)
                parts.append(
                    f"{python_bin} -c {script} {shlex.quote(payload_path)} {shlex.quote(result_path)}"
                )
                remote_cmd = " && ".join(parts)

                log.debug("task %s: executing on %s", task_id, host.host)
                proc = self._run_ssh(host.host, remote_cmd)
                output = proc.stdout or ""
                future.log = output

                # Download result
                result_local = os.path.join(tmpdir, "result.pkl")
                scp_result = self._run_scp(
                    self._scp_from(host.host, result_path, result_local)
                )

                if scp_result.returncode == 0 and os.path.exists(result_local):
                    with open(result_local, "rb") as f:
                        result_value = pickle.load(f)
                    future.set_result(result_value)
                    log.info("task %s: completed on %s", task_id, host.host)
                else:
                    future.set_exception(RuntimeError(output))
                    log.warning("task %s: failed on %s", task_id, host.host)

                # Cleanup remote task dir
                self._run_ssh(host.host, f"rm -rf {shlex.quote(task_dir)}")
            except Exception as exc:
                if not future.done():
                    future.log = traceback.format_exc()
                    future.set_exception(exc)
                log.error("task %s: exception: %s", task_id, exc)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
                self._release_host(host)

        self._pool.submit(_run)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


# Registry of built-in executors
EXECUTORS: dict[str, type[Executor]] = {
    "inline": InlineExecutor,
    "subprocess": SubprocessExecutor,
    "fork": ForkExecutor,
    "ray": RayExecutor,
    "ssh": SshExecutor,
}


def get_executor(
    executor: ExecutorName | Executor,
    *,
    snapshot: bool = False,
    capture: bool = True,
    max_workers: int = 1,
) -> Executor:
    """Get an executor instance from a string name or existing instance."""
    if isinstance(executor, Executor):
        return executor

    if isinstance(executor, str):
        if executor not in EXECUTORS:
            available = ", ".join(EXECUTORS.keys())
            raise ValueError(f"Unknown executor '{executor}'. Available: {available}")
        return EXECUTORS[executor](
            snapshot=snapshot, capture=capture, max_workers=max_workers
        )

    raise TypeError(f"executor must be str or Executor, got {type(executor).__name__}")
