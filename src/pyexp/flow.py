"""Flow — CLI-driven DAG execution with spin.

A flow wraps a function that builds a task DAG.  Tasks are collected from
the registry after the function runs — they don't need to be returned.
Auto-generates CLI flags from the function's kwargs, runs the DAG, and
supports ``--spin task_name`` to re-run only one task while loading cached
results for the rest.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .database import get_default_database
from .executors import Executor, get_default_executor, get_executor
from .task import Task, _collect_dag, _eval_tasks, _resolve_args, _save_task_result, _task_registry, _topo_sort, clear_task_registry

_NOT_SET = object()  # sentinel to distinguish "not provided on CLI" from argparse defaults


@dataclass
class _FlowEntry:
    """Lightweight stand-in for a :class:`Task` when loading historical results."""

    name: str
    key: str
    _result: Any
    _evaluated: bool = True
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

    @property
    def result(self) -> Any:
        return self._result

    @property
    def _hash(self) -> str:
        return self.key


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


class _FlowProgress:
    """Simple ANSI progress bar for flow execution (writes to stderr)."""

    def __init__(self, total: int = 0, width: int = 40):
        self.total = total
        self.width = width
        self.current = 0
        self.passed = 0
        self.failed = 0
        self.cached = 0
        self._render()

    def start(self, name: str = ""):
        self._render(name, running=True)

    def update(self, status: str, name: str = ""):
        self.current += 1
        if status == "passed":
            self.passed += 1
        elif status == "failed":
            self.failed += 1
        elif status == "cached":
            self.cached += 1
        self._render(name)

    def _render(self, name: str = "", running: bool = False):
        pct = self.current / self.total if self.total > 0 else 1
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        parts = []
        if self.passed:
            parts.append(f"\033[32m{self.passed} passed\033[0m")
        if self.failed:
            parts.append(f"\033[31m{self.failed} failed\033[0m")
        if self.cached:
            parts.append(f"\033[33m{self.cached} cached\033[0m")
        status = ", ".join(parts) if parts else ""

        max_name_len = 30
        display_name = name[:max_name_len] + "..." if len(name) > max_name_len else name

        line = f"\r{bar} {self.current}/{self.total}"
        if status:
            line += f" {status}"
        if display_name:
            if running:
                line += f" \033[36m[running: {display_name}]\033[0m"
            else:
                line += f" [{display_name}]"

        sys.stdout.flush()
        sys.stderr.write(f"{line}\033[K")
        sys.stderr.flush()

    def finish(self):
        sys.stderr.write("\n")
        sys.stderr.flush()


def _resolved_kwargs(t: Task) -> dict[str, Any]:
    """Map a task's positional args and kwargs to a flat dict using inspect.signature."""
    sig = inspect.signature(t._fn)
    params = list(sig.parameters.keys())
    result = {}
    for i, arg in enumerate(t._args):
        if i < len(params):
            result[params[i]] = arg.result if isinstance(arg, Task) and arg._evaluated else arg
    for k, v in t._kwargs.items():
        result[k] = v.result if isinstance(v, Task) and v._evaluated else v
    # Fill in defaults for parameters not covered by args or kwargs
    for name, param in sig.parameters.items():
        if name not in result and param.default is not inspect.Parameter.empty:
            result[name] = param.default
    return result


def _task_label(t: Task | _FlowEntry) -> str:
    """Return the task's explicit name if set, otherwise the function name."""
    if isinstance(t, _FlowEntry):
        return t.name
    return t._name or t._fn.__name__


class FlowResult:
    """Indexable collection of evaluated tasks from a flow run."""

    def __init__(self, tasks: list[Task | _FlowEntry]) -> None:
        self._tasks = tasks

    def __getitem__(self, key: int | str) -> Task | _FlowEntry | FlowResult:
        if isinstance(key, int):
            return self._tasks[key]
        if isinstance(key, str):
            matches = [t for t in self._tasks if re.search(key, _task_label(t))]
            if not matches:
                available = sorted({_task_label(t) for t in self._tasks})
                raise KeyError(f"No task matching '{key}'. Available: {available}")
            if len(matches) == 1:
                return matches[0]
            return FlowResult(matches)
        if callable(key):
            sig = inspect.signature(key)
            param_names = list(sig.parameters.keys())
            matches = []
            for t in self._tasks:
                kw = t.kwargs if isinstance(t, _FlowEntry) else _resolved_kwargs(t)
                if not all(p in kw for p in param_names):
                    continue
                if key(**{p: kw[p] for p in param_names}):
                    matches.append(t)
            if not matches:
                raise KeyError("No tasks match the callable filter")
            if len(matches) == 1:
                return matches[0]
            return FlowResult(matches)
        raise TypeError(f"FlowResult indices must be int, str, or callable, not {type(key).__name__}")

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task | _FlowEntry]:
        return iter(self._tasks)

    def __repr__(self) -> str:
        lines = [f"FlowResult({len(self._tasks)} tasks):"]
        for t in self._tasks:
            label = _task_label(t)
            status = "evaluated" if t._evaluated else "pending"
            lines.append(f"  {label} [{t._hash}] {status}")
        return "\n".join(lines)


class Flow:
    """A CLI-driven wrapper around a task-DAG–building function."""

    def __init__(self, fn: Callable, *, name: str | None = None) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        self._sig = inspect.signature(fn)

    def __call__(self, **overrides) -> Any:
        return self.run(**overrides)

    def __getitem__(self, key: int) -> FlowResult:
        """Load historical flow results from the database.

        ``flow[-1]`` returns the most recent run, ``flow[0]`` the first, etc.
        """
        db = get_default_database()
        runs = db.load(self.name)
        if not runs:
            raise RuntimeError(f"No previous runs for flow '{self.name}'")
        manifest = runs[key].result  # list of (label, storage_key, result[, kwargs])
        entries = []
        for entry in manifest:
            label, storage_key, result = entry[0], entry[1], entry[2]
            kw = entry[3] if len(entry) > 3 else {}
            entries.append(_FlowEntry(name=label, key=storage_key, _result=result, kwargs=kw))
        return FlowResult(entries)

    def results(self, **kwargs) -> FlowResult:
        """Re-collect tasks and load each one's latest cached result.

        Calls the flow function to rebuild the DAG (without executing it),
        then loads the most recent result for each task from the database.
        """
        resolved = {
            pname: kwargs.get(pname, param.default)
            for pname, param in self._sig.parameters.items()
            if pname in kwargs or param.default is not inspect.Parameter.empty
        }
        before = set(_task_registry.keys())
        self._fn(**resolved)
        tasks = [t for h, t in _task_registry.items() if h not in before]
        if not tasks:
            raise RuntimeError(f"Flow '{self.name}' produced no tasks")
        for t in tasks:
            runs = t.runs
            if not runs:
                raise RuntimeError(
                    f"No cached results for task '{_task_label(t)}' ({t._hash}). "
                    f"Run the flow first."
                )
            t._result = runs[-1].result
            t._evaluated = True

        # Clean up: remove tasks created by this flow from the global registry
        for t in tasks:
            _task_registry.pop(t._hash, None)

        return FlowResult(tasks)

    def run(self, **overrides) -> Any:
        """Parse CLI, build DAG, evaluate (or spin), return results."""
        args = self._parse_args()
        kwargs = self._resolve_kwargs(args, overrides)

        if args.executor:
            executor = get_executor(args.executor, capture=not args.no_capture)
        elif args.no_capture:
            executor = get_executor("inline", capture=False)
        else:
            executor = get_default_executor()
        spin_name = args.spin

        # Snapshot registry keys before the flow builds its DAG, so we only
        # pick up tasks that *this* flow created (not leftovers from earlier code).
        before = set(_task_registry.keys())
        returned = self._fn(**kwargs)

        # Collect only tasks added by this flow invocation
        all_tasks = [t for h, t in _task_registry.items() if h not in before]
        if not all_tasks:
            return returned

        # Find root tasks: tasks that are not a dependency of any other task
        dep_hashes = set()
        for t in all_tasks:
            for dep in t.dependencies:
                dep_hashes.add(dep._hash)
        root_list = [t for t in all_tasks if t._hash not in dep_hashes]

        progress = _FlowProgress()

        if spin_name:
            _spin_eval(root_list, spin_name, executor, progress=progress)
        else:
            _eval_tasks(root_list, executor, progress=progress)

        progress.finish()

        # Save manifest so flow[i] can reconstruct results without re-running
        db = get_default_database()
        manifest = [(_task_label(t), t._storage_key, t._result, _resolved_kwargs(t)) for t in all_tasks]
        db.save(self.name, manifest)

        # Clean up: remove tasks created by this flow from the global registry
        for t in all_tasks:
            _task_registry.pop(t._hash, None)

        return FlowResult(all_tasks)

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=f"Flow: {self.name}")
        parser.add_argument("--executor", type=str, default=None, help="Executor name")
        parser.add_argument("--spin", type=str, default=None, metavar="TASK_NAME", help="Re-run only this task, load others from cache")
        parser.add_argument("-s", "--no-capture", action="store_true", default=False, help="Disable output capture (like pytest -s)")

        for pname, param in self._sig.parameters.items():
            flag = f"--{pname.replace('_', '-')}"
            ann = param.annotation

            if ann is bool or isinstance(param.default, bool):
                parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=_NOT_SET)
            elif ann is int:
                parser.add_argument(flag, type=int, default=_NOT_SET)
            elif ann is float:
                parser.add_argument(flag, type=float, default=_NOT_SET)
            else:
                parser.add_argument(flag, type=str if ann is inspect.Parameter.empty else ann, default=_NOT_SET)

        return parser.parse_args()

    def _resolve_kwargs(self, args: argparse.Namespace, overrides: dict) -> dict:
        """Priority: CLI (explicitly provided) > overrides > fn defaults."""
        kwargs: dict[str, Any] = {}
        for pname, param in self._sig.parameters.items():
            cli_key = pname.replace("-", "_")
            cli_val = getattr(args, cli_key, _NOT_SET)
            if cli_val is not _NOT_SET:
                kwargs[pname] = cli_val
            elif pname in overrides:
                kwargs[pname] = overrides[pname]
            elif param.default is not inspect.Parameter.empty:
                kwargs[pname] = param.default
        return kwargs


def flow(fn: Callable | None = None, *, name: str | None = None) -> Flow | Callable:
    """Decorator that wraps a DAG-building function as a :class:`Flow`.

    Supports ``@flow`` and ``@flow(name="...")``::

        @flow
        def my_flow():
            ...

        @flow(name="custom")
        def my_flow():
            ...
    """
    def _wrap(fn: Callable) -> Flow:
        return Flow(fn, name=name)

    if fn is not None:
        return _wrap(fn)
    return _wrap


def _spin_eval(
    roots: list[Task], spin_name: str, executor: Executor | None = None, progress: Any = None
) -> None:
    """Re-run only tasks matching *spin_name*, load others from cache."""
    dag = _collect_dag(roots)
    order = _topo_sort(dag)
    ex = executor or get_default_executor()
    results: dict[str, Any] = {}

    if progress is not None:
        progress.total = len(order)

    # Find spin targets — match on explicit .name() first, then fn.__name__
    spin_targets = {t._hash for t in order if _task_label(t) == spin_name}
    if not spin_targets:
        available = sorted({_task_label(t) for t in order})
        raise ValueError(f"No task named '{spin_name}'. Available: {available}")

    for t in order:
        label = _task_label(t)
        if t._hash in spin_targets:
            if progress is not None:
                progress.start(label)

            # Execute with retry
            res = None
            for _attempt in range(t._retry):
                args, kwargs = _resolve_args(t, results)
                future = ex.submit(t._fn, *args, **kwargs)
                res = future.wait()
                if res.ok:
                    break

            if not res.ok:
                if progress is not None:
                    progress.update("failed", label)
                    progress.finish()
                raise RuntimeError(f"Task {label} ({t._hash}) failed:\n{res.log}")
            t._result = res.result
            t._evaluated = True
            _save_task_result(t, log=res.log, snapshot=ex.snapshot_hash)
            results[t._hash] = t._result
            if progress is not None:
                progress.update("passed", label)
        else:
            # Load from cache
            runs = t.runs
            if not runs:
                raise RuntimeError(
                    f"Spin requires cached result for '{label}' ({t._hash}), "
                    f"but no previous runs found. Run the full flow first."
                )
            t._result = runs[-1].result
            t._evaluated = True
            results[t._hash] = t._result
            if progress is not None:
                progress.update("cached", label)
