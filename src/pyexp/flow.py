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
import sys
from typing import Any, Callable

from .executors import Executor, get_default_executor, get_executor
from .task import Task, _collect_dag, _eval_tasks, _resolve_args, _save_task_result, _task_registry, _topo_sort, clear_task_registry

_NOT_SET = object()  # sentinel to distinguish "not provided on CLI" from argparse defaults


class Flow:
    """A CLI-driven wrapper around a task-DAG–building function."""

    def __init__(self, fn: Callable, *, name: str | None = None) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        self._sig = inspect.signature(fn)

    def __call__(self, **overrides) -> Any:
        return self.run(**overrides)

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

        if spin_name:
            _spin_eval(root_list, spin_name, executor)
        else:
            _eval_tasks(root_list, executor)

        results = tuple(t._result for t in root_list)
        return results[0] if len(results) == 1 else results

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


def _task_label(t: Task) -> str:
    """Return the task's explicit name if set, otherwise the function name."""
    return t._name or t._fn.__name__


def _spin_eval(roots: list[Task], spin_name: str, executor: Executor | None = None) -> None:
    """Re-run only tasks matching *spin_name*, load others from cache."""
    dag = _collect_dag(roots)
    order = _topo_sort(dag)
    ex = executor or get_default_executor()
    results: dict[str, Any] = {}

    # Find spin targets — match on explicit .name() first, then fn.__name__
    spin_targets = {t._hash for t in order if _task_label(t) == spin_name}
    if not spin_targets:
        available = sorted({_task_label(t) for t in order})
        raise ValueError(f"No task named '{spin_name}'. Available: {available}")

    for t in order:
        if t._hash in spin_targets:
            # Execute normally
            args, kwargs = _resolve_args(t, results)
            future = ex.submit(t._fn, *args, **kwargs)
            res = future.wait()
            if not res.ok:
                raise RuntimeError(f"Task {_task_label(t)} ({t._hash}) failed:\n{res.log}")
            t._result = res.result
            t._evaluated = True
            _save_task_result(t, log=res.log, snapshot=ex.snapshot_hash)
            results[t._hash] = t._result
        else:
            # Load from cache
            runs = t.runs
            if not runs:
                raise RuntimeError(
                    f"Spin requires cached result for '{_task_label(t)}' ({t._hash}), "
                    f"but no previous runs found. Run the full flow first."
                )
            t._result = runs[-1].result
            t._evaluated = True
            results[t._hash] = t._result
