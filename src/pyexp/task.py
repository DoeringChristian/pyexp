"""Lazy task DAG system.

Decorated functions return :class:`Task` nodes when called; passing a ``Task``
as an argument to another task creates a dependency edge.  Tasks are
deduplicated by content hash and executed via the existing executor system.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import json
from typing import Any, Callable

from .config import Runs
from .database import Entry, get_default_database
from .executors import Executor, get_default_executor

# ---------------------------------------------------------------------------
# Sentinel & registry
# ---------------------------------------------------------------------------

_UNSET = type(
    "_UNSET", (), {"__repr__": lambda self: "_UNSET", "__bool__": lambda self: False}
)()

_task_registry: dict[str, Task] = {}


def clear_task_registry() -> None:
    """Remove all cached :class:`Task` objects (useful in tests)."""
    _task_registry.clear()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _task_hash(fn: Callable, args: tuple, kwargs: dict) -> str:
    """Content-hash from function identity + args.

    Task arguments are replaced by their hash (recursive dedup).
    """

    def _normalise(v: Any) -> Any:
        if isinstance(v, Task):
            return {"__task__": v._hash}
        return v

    key = json.dumps(
        {
            "fn": f"{fn.__module__}.{fn.__qualname__}",
            "args": [_normalise(a) for a in args],
            "kwargs": {k: _normalise(v) for k, v in sorted(kwargs.items())},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


class Task:
    """A lazy node in a computation DAG."""

    __slots__ = (
        "_fn",
        "_args",
        "_kwargs",
        "_hash",
        "_result",
        "_evaluated",
        "_db",
        "_name",
        "_retry",
    )

    def __init__(
        self,
        fn: Callable,
        args: tuple,
        kwargs: dict,
        hash: str,
    ) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._hash = hash
        self._result: Any = _UNSET
        self._evaluated: bool = False
        self._db = None
        self._name: str | None = None
        self._retry: int = 1

    @classmethod
    def _from_manifest(cls, *, label: str, storage_key: str, result: Any) -> Task:
        """Reconstruct a lightweight Task from a stored manifest entry.

        Sets ``_name`` to None so that ``_storage_key`` returns *storage_key*
        directly (it's already the full key).  The *label* is stored on a
        separate attribute for display purposes.
        """
        t = object.__new__(cls)
        t._fn = None
        t._args = ()
        t._kwargs = {}
        t._hash = storage_key
        t._result = result
        t._evaluated = True
        t._db = None
        t._name = label
        t._retry = 1
        return t

    # -- dependency introspection --

    @property
    def dependencies(self) -> list[Task]:
        """Extract :class:`Task` objects from args and kwargs."""
        deps: list[Task] = []
        for a in self._args:
            if isinstance(a, Task):
                deps.append(a)
        for v in self._kwargs.values():
            if isinstance(v, Task):
                deps.append(v)
        return deps

    @property
    def _storage_key(self) -> str:
        """Database key: ``name_hash`` if named, otherwise just the hash.

        For manifest-loaded tasks (``_fn is None``), ``_hash`` already holds
        the full storage key.
        """
        if self._fn is None:
            return self._hash
        if self._name:
            return f"{self._name}_{self._hash}"
        return self._hash

    @property
    def runs(self) -> Runs[Entry]:
        """Load previous results from the database."""
        db = self._db or get_default_database()
        return Runs(db.load(self._storage_key))

    @property
    def snapshot(self):
        """Return a :class:`Snapshot` for the latest run, or ``None``."""
        from .utils import Snapshot

        runs = self.runs
        if not runs:
            return None
        h = runs[-1].metadata.get("snapshot")
        return Snapshot(h) if h else None

    @property
    def result(self) -> Any:
        """Access the task's result after evaluation."""
        if not self._evaluated:
            raise RuntimeError("Task has not been evaluated yet")
        return self._result

    def name(self, name: str) -> Task:
        """Assign a human-readable name to this task (returns self for chaining)."""
        self._name = name
        return self

    # -- repr / str --

    def __repr__(self) -> str:
        status = "evaluated" if self._evaluated else "pending"
        ndeps = len(self.dependencies)
        label = self._name or self._fn.__name__
        return f"Task({label}, hash={self._hash}, deps={ndeps}, {status})"

    def __str__(self) -> str:
        return str(self.eval())

    # -- evaluation --

    def eval(self, *, executor: Executor | None = None) -> Any:
        """Evaluate the full DAG rooted at this task."""
        if self._evaluated:
            return self._result
        _eval_tasks([self], executor)
        return self._result


# ---------------------------------------------------------------------------
# DAG engine (internal)
# ---------------------------------------------------------------------------


def _collect_dag(roots: list[Task]) -> dict[str, Task]:
    """BFS from *roots* to collect all reachable tasks."""
    visited: dict[str, Task] = {}
    queue = collections.deque(roots)
    while queue:
        t = queue.popleft()
        if t._hash in visited:
            continue
        visited[t._hash] = t
        for dep in t.dependencies:
            if dep._hash not in visited:
                queue.append(dep)
    return visited


def _topo_sort(tasks: dict[str, Task]) -> list[Task]:
    """Kahn's algorithm — returns tasks in dependency order."""
    in_degree: dict[str, int] = {h: 0 for h in tasks}
    dependents: dict[str, list[str]] = {h: [] for h in tasks}

    for h, t in tasks.items():
        for dep in t.dependencies:
            if dep._hash in tasks:
                in_degree[h] += 1
                dependents[dep._hash].append(h)

    queue = collections.deque(h for h, d in in_degree.items() if d == 0)
    order: list[Task] = []

    while queue:
        h = queue.popleft()
        order.append(tasks[h])
        for child_h in dependents[h]:
            in_degree[child_h] -= 1
            if in_degree[child_h] == 0:
                queue.append(child_h)

    if len(order) != len(tasks):
        raise RuntimeError("Cycle detected in task DAG")

    return order


def _resolve_args(task: Task, results: dict[str, Any]) -> tuple[tuple, dict]:
    """Replace Task refs in args/kwargs with concrete values."""
    args = tuple(results[a._hash] if isinstance(a, Task) else a for a in task._args)
    kwargs = {
        k: results[v._hash] if isinstance(v, Task) else v
        for k, v in task._kwargs.items()
    }
    return args, kwargs


def _save_task_result(t: Task, log: str = "", snapshot: str | None = None) -> None:
    """Persist a completed task's result to the database."""
    db = t._db or get_default_database()
    meta: dict[str, Any] = {"fn": f"{t._fn.__module__}.{t._fn.__qualname__}"}
    if snapshot is not None:
        meta["snapshot"] = snapshot
    db.save(t._storage_key, t._result, log=log, metadata=meta)


def _eval_tasks(
    roots: list[Task], executor: Executor | None = None, progress: Any = None
) -> None:
    """Evaluate tasks in topological order (blocking)."""
    dag = _collect_dag(roots)
    order = _topo_sort(dag)
    ex = executor or get_default_executor()
    results: dict[str, Any] = {}
    snapshot = ex.snapshot_hash

    if progress is not None:
        progress.total = len(order)

    for t in order:
        label = t._name or t._fn.__name__
        if t._evaluated:
            results[t._hash] = t._result
            if progress is not None:
                progress.update("cached", label)
            continue

        if progress is not None:
            progress.start(label)

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
            raise RuntimeError(f"Task {t._fn.__name__} ({t._hash}) failed:\n{res.log}")
        t._result = res.result
        t._evaluated = True
        _save_task_result(t, log=res.log, snapshot=snapshot)
        results[t._hash] = t._result
        if progress is not None:
            progress.update("passed", label)


# ---------------------------------------------------------------------------
# task decorator
# ---------------------------------------------------------------------------


def task(
    fn: Callable | None = None, *, executor: Executor | None = None, retry: int = 1
) -> Callable:
    """Decorator that makes a function return a :class:`Task` when called.

    Supports ``@task``, ``@task(executor=...)``, and ``@task(retry=N)``::

        @task
        def compute(x):
            return x ** 2

        @task(retry=3)
        def flaky(x):
            return x ** 2
    """

    def _wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Task:
            h = _task_hash(fn, args, kwargs)
            if h in _task_registry:
                return _task_registry[h]
            t = Task(fn, args, kwargs, h)
            t._retry = retry
            _task_registry[h] = t
            return t

        wrapper._original = fn
        wrapper._task_executor = executor
        return wrapper

    if fn is not None:
        return _wrap(fn)
    return _wrap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def eval(*tasks: Any, executor: Executor | None = None) -> Any:
    """Evaluate tasks and return results.

    Returns a single result for one task, or a tuple for multiple.
    Filters out ``Ellipsis`` arguments.
    """
    filtered = [t for t in tasks if t is not ... and isinstance(t, Task)]
    if not filtered:
        return None
    _eval_tasks(filtered, executor)
    results = tuple(t._result for t in filtered)
    return results[0] if len(results) == 1 else results
