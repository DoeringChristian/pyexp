"""Tests for the Flow CLI-driven DAG execution system."""

import pytest

import pyexp
from pyexp.database import FileDatabase, set_default_database
from pyexp.executors import InlineExecutor, set_default_executor
from pyexp.flow import Flow, FlowResult
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
    assert isinstance(result, FlowResult)
    assert result[-1].result == 20


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
    assert result[0].result == "real"
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
    assert result[-1].result == "hello alice"


def test_run_cli_overrides_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--name", "bob"])

    @pyexp.task
    def greet(name):
        return f"hello {name}"

    @pyexp.flow
    def my_flow(name="world"):
        return greet(name)

    result = my_flow.run()
    assert result[-1].result == "hello bob"


def test_run_cli_overrides_run_kwargs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--name", "cli"])

    @pyexp.task
    def greet(name):
        return f"hello {name}"

    @pyexp.flow
    def my_flow(name="default"):
        return greet(name)

    result = my_flow.run(name="override")
    assert result[-1].result == "hello cli"


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
    assert result[0].result == 1
    assert result[1].result == 2


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
    assert result[-1].result == 20
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
    assert result[-1].result == "xxxxx"


def test_cli_bool_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--verbose"])

    @pyexp.task
    def log(verbose):
        return "verbose" if verbose else "quiet"

    @pyexp.flow
    def my_flow(verbose: bool = False):
        return log(verbose)

    result = my_flow.run()
    assert result[-1].result == "verbose"


def test_cli_no_bool_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test", "--no-verbose"])

    @pyexp.task
    def log(verbose):
        return "verbose" if verbose else "quiet"

    @pyexp.flow
    def my_flow(verbose: bool = True):
        return log(verbose)

    result = my_flow.run()
    assert result[-1].result == "quiet"


# ---------------------------------------------------------------------------
# FlowResult
# ---------------------------------------------------------------------------


def test_flow_result_int_index(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def first():
        return "a"

    @pyexp.task
    def second(x):
        return "b"

    @pyexp.flow
    def my_flow():
        second(first())

    result = my_flow.run()
    assert isinstance(result, FlowResult)
    assert result[0].result == "a"
    assert result[1].result == "b"
    assert result[-1].result == "b"


def test_flow_result_str_index_by_name(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def compute(x):
        return x * 2

    @pyexp.flow
    def my_flow():
        compute(1).name("pretrain")
        compute(2).name("finetune")

    result = my_flow.run()
    assert result["pretrain"].result == 2
    assert result["finetune"].result == 4


def test_flow_result_str_index_by_fn_name(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def alpha():
        return 10

    @pyexp.task
    def beta(x):
        return x + 1

    @pyexp.flow
    def my_flow():
        beta(alpha())

    result = my_flow.run()
    assert result["alpha"].result == 10
    assert result["beta"].result == 11


def test_flow_result_str_regex_multiple_matches(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def step(tag):
        return tag

    @pyexp.flow
    def my_flow():
        step("a").name("train_phase1")
        step("b").name("train_phase2")
        step("c").name("eval_final")

    result = my_flow.run()
    sub = result["train.*"]
    assert isinstance(sub, FlowResult)
    assert len(sub) == 2
    assert sub[0].result == "a"
    assert sub[1].result == "b"


def test_flow_result_str_no_match(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def compute():
        return 1

    @pyexp.flow
    def my_flow():
        compute()

    result = my_flow.run()
    with pytest.raises(KeyError, match="No task matching"):
        result["nonexistent"]


def test_flow_result_len_and_iter(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def val(x):
        return x

    @pyexp.flow
    def my_flow():
        val(1).name("one")
        val(2).name("two")
        val(3).name("three")

    result = my_flow.run()
    assert len(result) == 3
    results = [t.result for t in result]
    assert results == [1, 2, 3]


def test_flow_result_repr(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def val():
        return 1

    @pyexp.flow
    def my_flow():
        val().name("my_task")

    result = my_flow.run()
    r = repr(result)
    assert "FlowResult(1 tasks)" in r
    assert "my_task" in r


def test_task_result_property_before_eval():
    @pyexp.task
    def val():
        return 1

    t = val()
    with pytest.raises(RuntimeError, match="not been evaluated"):
        t.result


def test_task_result_property_after_eval(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def val():
        return 42

    @pyexp.flow
    def my_flow():
        val()

    result = my_flow.run()
    assert result[0].result == 42


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_failures(monkeypatch):
    """@task(retry=3): fails twice then succeeds on third attempt."""
    monkeypatch.setattr("sys.argv", ["test"])

    call_count = 0

    @pyexp.task(retry=3)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return "ok"

    @pyexp.flow
    def my_flow():
        flaky()

    result = my_flow.run()
    assert result[-1].result == "ok"
    assert call_count == 3


def test_retry_default_no_retry(monkeypatch):
    """@task (default retry=1) with failure raises immediately."""
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def fail_once():
        raise ValueError("boom")

    @pyexp.flow
    def my_flow():
        fail_once()

    with pytest.raises(RuntimeError, match="failed"):
        my_flow.run()


def test_retry_all_attempts_fail(monkeypatch):
    """@task(retry=2): all attempts fail → raises last error."""
    monkeypatch.setattr("sys.argv", ["test"])

    call_count = 0

    @pyexp.task(retry=2)
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ValueError("always fails")

    @pyexp.flow
    def my_flow():
        always_fail()

    with pytest.raises(RuntimeError, match="failed"):
        my_flow.run()
    assert call_count == 2


def test_retry_in_eval(monkeypatch):
    """retry works through pyexp.eval() (no flow)."""
    call_count = 0

    @pyexp.task(retry=3)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return "ok"

    result = pyexp.eval(flaky())
    assert result == "ok"
    assert call_count == 3


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


def test_flow_progress_bar_runs(monkeypatch, capsys):
    """Flow with progress bar completes without errors; output goes to stderr."""
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def step(x):
        return x * 2

    @pyexp.flow
    def my_flow():
        step(1).name("double")
        step(2).name("triple")

    result = my_flow.run()
    assert len(result) == 2
    # Progress output goes to stderr
    captured = capsys.readouterr()
    assert "passed" in captured.err or "█" in captured.err


# ---------------------------------------------------------------------------
# Flow indexing (historical results)
# ---------------------------------------------------------------------------


def test_flow_getitem_loads_last_result(monkeypatch, tmp_path):
    """flow[-1] loads the most recent cached result for each task."""
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

    # Run to populate the database
    my_flow.run()
    clear_task_registry()

    # Load from cache — no run() needed
    result = my_flow[-1]
    assert isinstance(result, FlowResult)
    assert result[-1].result == 20


def test_flow_getitem_first_run(monkeypatch, tmp_path):
    """flow[0] loads the first cached result."""
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def val():
        return 42

    @pyexp.flow
    def my_flow():
        val()

    my_flow.run()
    clear_task_registry()

    result = my_flow[0]
    assert result[0].result == 42


def test_flow_getitem_no_runs_raises():
    """flow[-1] raises if no previous runs exist."""

    @pyexp.flow
    def my_flow():
        pass

    with pytest.raises(RuntimeError, match="No previous runs"):
        my_flow[-1]


def test_flow_getitem_multiple_runs(monkeypatch, tmp_path):
    """flow[0] returns first run results, flow[-1] returns latest."""
    monkeypatch.setattr("sys.argv", ["test"])

    counter = {"n": 0}

    @pyexp.task
    def counting():
        counter["n"] += 1
        return counter["n"]

    @pyexp.flow
    def my_flow():
        counting()

    # First run
    my_flow.run()
    clear_task_registry()

    # Second run
    my_flow.run()
    clear_task_registry()

    assert my_flow[0][0].result == 1
    assert my_flow[-1][0].result == 2
    assert my_flow[1][0].result == 2


# ---------------------------------------------------------------------------
# Flow.results() — re-collect tasks and load cached results
# ---------------------------------------------------------------------------


def test_flow_results_loads_cached(monkeypatch):
    """flow.results() re-collects tasks and loads the latest cached result."""
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

    my_flow.run()
    clear_task_registry()

    result = my_flow.results()
    assert isinstance(result, FlowResult)
    assert result[-1].result == 20


def test_flow_results_with_kwargs(monkeypatch):
    """flow.results(name=...) rebuilds the DAG with the given kwargs."""
    monkeypatch.setattr("sys.argv", ["test"])

    @pyexp.task
    def greet(name):
        return f"hello {name}"

    @pyexp.flow
    def my_flow(name="world"):
        greet(name)

    my_flow.run(name="alice")
    clear_task_registry()

    result = my_flow.results(name="alice")
    assert result[0].result == "hello alice"


def test_flow_results_no_cache_raises():
    """flow.results() raises if no cached results exist."""

    @pyexp.task
    def val():
        return 1

    @pyexp.flow
    def my_flow():
        val()

    with pytest.raises(RuntimeError, match="No cached results"):
        my_flow.results()


def test_flow_results_returns_real_tasks(monkeypatch):
    """flow.results() returns actual Task objects, not _FlowEntry."""
    monkeypatch.setattr("sys.argv", ["test"])
    from pyexp.task import Task

    @pyexp.task
    def val():
        return 42

    @pyexp.flow
    def my_flow():
        val()

    my_flow.run()
    clear_task_registry()

    result = my_flow.results()
    assert isinstance(result[0], Task)
    assert result[0].result == 42
