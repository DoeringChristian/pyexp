"""Tests for the lazy task DAG system."""

import pytest

import pyexp
from pyexp.database import FileDatabase, set_default_database
from pyexp.executors import InlineExecutor, set_default_executor
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


def test_decorator_returns_task():
    @pyexp.task
    def greet():
        return "hi"

    result = greet()
    assert isinstance(result, Task)


def test_deduplication_same_fn_same_args():
    @pyexp.task
    def add(a, b):
        return a + b

    t1 = add(1, 2)
    t2 = add(1, 2)
    assert t1 is t2


def test_different_args_different_task():
    @pyexp.task
    def add(a, b):
        return a + b

    t1 = add(1, 2)
    t2 = add(3, 4)
    assert t1 is not t2


def test_decorator_with_parens():
    @pyexp.task()
    def greet():
        return "hi"

    result = greet()
    assert isinstance(result, Task)


def test_preserves_function_name():
    @pyexp.task
    def my_function():
        return 1

    assert my_function.__name__ == "my_function"


# ---------------------------------------------------------------------------
# repr / str
# ---------------------------------------------------------------------------


def test_repr_does_not_evaluate():
    call_count = 0

    @pyexp.task
    def side_effect():
        nonlocal call_count
        call_count += 1
        return "done"

    t = side_effect()
    r = repr(t)
    assert "side_effect" in r
    assert "pending" in r
    assert call_count == 0


def test_str_triggers_evaluation():
    @pyexp.task
    def greet():
        return "hello"

    t = greet()
    assert str(t) == "hello"
    assert t._evaluated


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_simple_eval():
    @pyexp.task
    def compute():
        return 42

    t = compute()
    assert t.eval() == 42


def test_chain_eval():
    @pyexp.task
    def source():
        return 10

    @pyexp.task
    def double(x):
        return x * 2

    t1 = source()
    t2 = double(t1)
    assert t2.eval() == 20


def test_diamond_dag_shared_dep_runs_once():
    """Diamond: A -> B, A -> C, B+C -> D. A should only execute once."""
    call_count = 0

    @pyexp.task
    def root():
        nonlocal call_count
        call_count += 1
        return 1

    @pyexp.task
    def left(x):
        return x + 10

    @pyexp.task
    def right(x):
        return x + 100

    @pyexp.task
    def merge(l, r):
        return l + r

    a = root()
    b = left(a)
    c = right(a)
    d = merge(b, c)

    assert d.eval() == 112  # (1+10) + (1+100)
    assert call_count == 1


def test_already_evaluated_skips():
    call_count = 0

    @pyexp.task
    def counted():
        nonlocal call_count
        call_count += 1
        return "ok"

    t = counted()
    t.eval()
    assert call_count == 1
    t.eval()
    assert call_count == 1  # not re-executed


def test_eval_returns_none():
    @pyexp.task
    def nothing():
        return None

    t = nothing()
    assert t.eval() is None
    assert t._evaluated


# ---------------------------------------------------------------------------
# pyexp.eval (module-level)
# ---------------------------------------------------------------------------


def test_pyexp_eval_single():
    @pyexp.task
    def val():
        return 7

    t = val()
    assert pyexp.eval(t) == 7


def test_pyexp_eval_multiple_returns_tuple():
    @pyexp.task
    def a():
        return 1

    @pyexp.task
    def b():
        return 2

    t1 = a()
    t2 = b()
    result = pyexp.eval(t1, t2)
    assert result == (1, 2)


def test_pyexp_eval_filters_ellipsis():
    @pyexp.task
    def val():
        return 99

    t = val()
    assert pyexp.eval(t, ...) == 99


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_error_propagation():
    @pyexp.task
    def bad():
        raise ValueError("boom")

    @pyexp.task
    def downstream(x):
        return x + 1

    t1 = bad()
    t2 = downstream(t1)
    with pytest.raises(RuntimeError, match="bad"):
        t2.eval()


# ---------------------------------------------------------------------------
# kwargs
# ---------------------------------------------------------------------------


def test_kwargs_forwarding():
    @pyexp.task
    def greet(name, greeting="hello"):
        return f"{greeting}, {name}"

    t = greet("world", greeting="hi")
    assert t.eval() == "hi, world"


def test_task_as_kwarg():
    @pyexp.task
    def source():
        return "val"

    @pyexp.task
    def use(data=None):
        return f"got {data}"

    s = source()
    t = use(data=s)
    assert t.eval() == "got val"


# ---------------------------------------------------------------------------
# runs / persistence
# ---------------------------------------------------------------------------


def test_runs_empty_before_eval():
    @pyexp.task
    def val():
        return 1

    t = val()
    assert len(t.runs) == 0


def test_runs_after_eval():
    @pyexp.task
    def val():
        return 42

    t = val()
    t.eval()
    runs = t.runs
    assert len(runs) == 1
    assert runs[-1].result == 42


def test_runs_metadata_has_snapshot_when_executor_provides_it(tmp_path):
    set_default_executor(InlineExecutor(snapshot=True, capture=False))
    set_default_database(FileDatabase(tmp_path / "snap_db"))

    @pyexp.task
    def val():
        return 1

    t = val()
    t.eval()
    assert "snapshot" in t.runs[-1].metadata
    assert isinstance(t.runs[-1].metadata["snapshot"], str)


def test_runs_metadata_no_snapshot_without_executor_flag():
    @pyexp.task
    def val():
        return 1

    t = val()
    t.eval()
    assert "snapshot" not in t.runs[-1].metadata


def test_chain_eval_persists_all():
    @pyexp.task
    def source():
        return 10

    @pyexp.task
    def double(x):
        return x * 2

    t1 = source()
    t2 = double(t1)
    t2.eval()

    assert len(t1.runs) == 1
    assert t1.runs[-1].result == 10
    assert len(t2.runs) == 1
    assert t2.runs[-1].result == 20
