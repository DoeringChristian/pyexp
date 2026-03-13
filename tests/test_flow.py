"""Tests for the Flow CLI-driven DAG execution system."""

import pytest

import pyexp
from pyexp.database import FileDatabase, set_default_database
from pyexp.executors import InlineExecutor, set_default_executor
from pyexp.flow import Flow
from pyexp.task import Task, clear_task_registry


@pytest.fixture(autouse=True)
def _reset(tmp_path):
    """Use InlineExecutor, temp database, and clear registry for each test."""
    set_default_executor(InlineExecutor(capture=False))
    set_default_database(FileDatabase(tmp_path / "db"))
    yield
    clear_task_registry()
    set_default_executor(None)
    set_default_database(None)


# ---------------------------------------------------------------------------
# Decorator basics
# ---------------------------------------------------------------------------


def test_decorator_returns_flow():
    @pyexp.flow
    def my_flow():
        pass

    assert isinstance(my_flow, Flow)


def test_decorator_with_name():
    @pyexp.flow(name="custom")
    def my_flow():
        pass

    assert isinstance(my_flow, Flow)
    assert my_flow.name == "custom"


def test_decorator_default_name():
    @pyexp.flow
    def my_flow():
        pass

    assert my_flow.name == "my_flow"


# ---------------------------------------------------------------------------
# run() basics
# ---------------------------------------------------------------------------


def test_run_builds_and_evaluates_dag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def source():
        return 10

    @pyexp.task
    def double(x):
        return x * 2

    @pyexp.flow
    def my_flow():
        double(source())

    result = my_flow.run()
    assert result == 20


def test_run_no_return_evaluates_all(monkeypatch):
    """Flow function doesn't need to return tasks — DAG is inferred from registry."""
    monkeypatch.setattr("sys.argv", ["test"])

    call_counts = {"a": 0, "b": 0}

    @pyexp.task
    def a():
        call_counts["a"] += 1
        return 1

    @pyexp.task
    def b(x):
        call_counts["b"] += 1
        return x + 1

    @pyexp.flow
    def my_flow():
        b(a())
        # no return

    my_flow.run()
    assert call_counts == {"a": 1, "b": 1}


def test_run_ignores_tasks_created_before_flow(monkeypatch):
    """Tasks registered before the flow call must not be picked up."""
    monkeypatch.setattr("sys.argv", ["test"])

    stray_count = 0

    @pyexp.task
    def stray():
        nonlocal stray_count
        stray_count += 1
        return "stray"

    @pyexp.task
    def real():
        return "real"

    # Create a task in the registry *before* the flow runs
    stray()

    @pyexp.flow
    def my_flow():
        real()

    result = my_flow.run()
    assert result == "real"
    assert stray_count == 0  # stray was never executed


def test_run_with_kwargs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def greet(name):
        return f"hello {name}"

    @pyexp.flow
    def my_flow(name="world"):
        return greet(name)

    result = my_flow.run(name="alice")
    assert result == "hello alice"


def test_run_cli_overrides_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--name", "bob"])

    @pyexp.task
    def greet(name):
        return f"hello {name}"

    @pyexp.flow
    def my_flow(name="world"):
        return greet(name)

    result = my_flow.run()
    assert result == "hello bob"


def test_run_cli_overrides_run_kwargs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--name", "cli"])

    @pyexp.task
    def greet(name):
        return f"hello {name}"

    @pyexp.flow
    def my_flow(name="default"):
        return greet(name)

    result = my_flow.run(name="override")
    assert result == "hello cli"


# ---------------------------------------------------------------------------
# Multiple roots
# ---------------------------------------------------------------------------


def test_multiple_roots_returns_tuple(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def a():
        return 1

    @pyexp.task
    def b():
        return 2

    @pyexp.flow
    def my_flow():
        return a(), b()

    result = my_flow.run()
    assert result == (1, 2)


# ---------------------------------------------------------------------------
# Spin
# ---------------------------------------------------------------------------


def test_spin_reruns_target_loads_others(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["test"])

    call_counts = {"source": 0, "double": 0}

    @pyexp.task
    def source():
        call_counts["source"] += 1
        return 10

    @pyexp.task
    def double(x):
        call_counts["double"] += 1
        return x * 2

    @pyexp.flow
    def my_flow():
        return double(source())

    # First: full run to populate cache
    my_flow.run()
    assert call_counts == {"source": 1, "double": 1}

    # Reset counts and registry for spin run
    call_counts["source"] = 0
    call_counts["double"] = 0
    clear_task_registry()

    # Spin: only re-run double, load source from cache
    monkeypatch.setattr("sys.argv", ["test", "--spin", "double"])
    result = my_flow.run()
    assert result == 20
    assert call_counts["source"] == 0  # loaded from cache
    assert call_counts["double"] == 1  # re-executed


def test_spin_by_explicit_name(monkeypatch, tmp_path):
    """Spin matches on .name() labels, not just function names."""
    monkeypatch.setattr("sys.argv", ["test"])

    call_counts = {"pretrain": 0, "finetune": 0}

    @pyexp.task
    def train(tag):
        call_counts[tag] += 1
        return f"result_{tag}"

    @pyexp.flow
    def my_flow():
        pre = train("pretrain").name("pretrain")
        train("finetune").name("finetune")

    # Full run
    my_flow.run()
    assert call_counts == {"pretrain": 1, "finetune": 1}

    # Spin finetune only
    call_counts["pretrain"] = 0
    call_counts["finetune"] = 0
    clear_task_registry()
    monkeypatch.setattr("sys.argv", ["test", "--spin", "finetune"])
    my_flow.run()
    assert call_counts["pretrain"] == 0
    assert call_counts["finetune"] == 1


def test_task_name_returns_self():
    @pyexp.task
    def val():
        return 1

    t = val()
    assert t.name("my_name") is t
    assert t._name == "my_name"


def test_task_name_in_repr():
    @pyexp.task
    def val():
        return 1

    t = val().name("custom_label")
    assert "custom_label" in repr(t)


def test_spin_unknown_task_name(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--spin", "nonexistent"])

    @pyexp.task
    def source():
        return 1

    @pyexp.flow
    def my_flow():
        return source()

    with pytest.raises(ValueError, match="No task named 'nonexistent'"):
        my_flow.run()


def test_spin_no_previous_runs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--spin", "double"])

    @pyexp.task
    def source():
        return 10

    @pyexp.task
    def double(x):
        return x * 2

    @pyexp.flow
    def my_flow():
        return double(source())

    # No full run first — source has no cached result
    with pytest.raises(RuntimeError, match="no previous runs"):
        my_flow.run()


# ---------------------------------------------------------------------------
# CLI type coercion
# ---------------------------------------------------------------------------


def test_cli_int_param(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--count", "5"])

    @pyexp.task
    def repeat(n):
        return "x" * n

    @pyexp.flow
    def my_flow(count: int = 1):
        return repeat(count)

    result = my_flow.run()
    assert result == "xxxxx"


def test_cli_bool_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--verbose"])

    @pyexp.task
    def log(verbose):
        return "verbose" if verbose else "quiet"

    @pyexp.flow
    def my_flow(verbose: bool = False):
        return log(verbose)

    result = my_flow.run()
    assert result == "verbose"


def test_cli_no_bool_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--no-verbose"])

    @pyexp.task
    def log(verbose):
        return "verbose" if verbose else "quiet"

    @pyexp.flow
    def my_flow(verbose: bool = True):
        return log(verbose)

    result = my_flow.run()
    assert result == "quiet"
