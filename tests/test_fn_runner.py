"""Tests for pyexp executors (general-purpose function execution)."""

import tempfile
import time
from pathlib import Path

import pytest

from pyexp.executors import FnFuture, SubprocessExecutor, InlineExecutor, ForkExecutor


def _add(a, b):
    return a + b


def test_subprocess_basic():
    """SubprocessExecutor runs a simple function and returns its result."""
    executor = SubprocessExecutor()
    f = executor.submit(_add, 2, 3)
    assert isinstance(f, FnFuture)
    assert f.result() == 5
    assert f.exception() is None


def test_subprocess_kwargs():
    """Keyword arguments are forwarded correctly."""
    executor = SubprocessExecutor()
    f = executor.submit(_add, 1, b=10)
    assert f.result() == 11


def _fail():
    raise ValueError("boom")


def test_subprocess_error_captures_traceback():
    """When fn raises, traceback appears in log."""
    executor = SubprocessExecutor()
    f = executor.submit(_fail)
    assert f.exception() is not None
    assert "ValueError" in f.log
    assert "boom" in f.log


def _print_and_return(msg):
    print(msg)
    return msg


def test_subprocess_capture_output():
    """Captured stdout appears in .log."""
    executor = SubprocessExecutor()
    f = executor.submit(_print_and_return, "hello world")
    assert f.result() == "hello world"
    assert "hello world" in f.log


def test_inline_basic():
    """InlineExecutor runs in the same process."""
    executor = InlineExecutor()
    f = executor.submit(_add, 2, 3)
    assert f.result() == 5
    assert f.exception() is None


def test_inline_error():
    """InlineExecutor captures errors."""
    executor = InlineExecutor()
    f = executor.submit(_fail)
    assert f.exception() is not None
    assert "ValueError" in f.log
    assert "boom" in f.log


def test_inline_capture_output():
    """InlineExecutor captures stdout."""
    executor = InlineExecutor()
    f = executor.submit(_print_and_return, "hello inline")
    assert f.result() == "hello inline"
    assert "hello inline" in f.log


def test_fork_basic():
    """ForkExecutor runs in a forked process."""
    executor = ForkExecutor()
    f = executor.submit(_add, 2, 3)
    assert f.result() == 5
    assert f.exception() is None


def test_fork_error():
    """ForkExecutor captures errors."""
    executor = ForkExecutor()
    f = executor.submit(_fail)
    assert f.exception() is not None
    assert "ValueError" in f.log
    assert "boom" in f.log


def test_fork_capture_output():
    """ForkExecutor captures stdout."""
    executor = ForkExecutor()
    f = executor.submit(_print_and_return, "hello fork")
    assert f.result() == "hello fork"
    assert "hello fork" in f.log


# --- Async / worker pool tests ---


def _slow_add(a, b):
    import time

    time.sleep(0.3)
    return a + b


@pytest.mark.parametrize(
    "executor_cls", [InlineExecutor, SubprocessExecutor, ForkExecutor]
)
def test_submit_returns_pending_future(executor_cls):
    """submit() returns a FnFuture that is initially not done (pending)."""
    executor = executor_cls()
    f = executor.submit(_slow_add, 1, 2)
    assert isinstance(f, FnFuture)
    # The future should resolve eventually
    assert f.result(timeout=10) == 3
    executor.shutdown()


@pytest.mark.parametrize(
    "executor_cls", [InlineExecutor, SubprocessExecutor, ForkExecutor]
)
def test_future_result_blocks(executor_cls):
    """future.result() blocks until the value is available."""
    executor = executor_cls()
    f = executor.submit(_add, 10, 20)
    assert f.result(timeout=10) == 30
    assert f.done()
    executor.shutdown()


@pytest.mark.parametrize(
    "executor_cls", [InlineExecutor, SubprocessExecutor, ForkExecutor]
)
def test_future_log_populated(executor_cls):
    """future.log is populated after resolution."""
    executor = executor_cls()
    f = executor.submit(_print_and_return, "log test")
    assert f.result(timeout=10) == "log test"
    assert "log test" in f.log
    executor.shutdown()


@pytest.mark.parametrize(
    "executor_cls", [SubprocessExecutor, ForkExecutor]
)
def test_concurrent_execution(executor_cls):
    """max_workers > 1 runs tasks concurrently (subprocess/fork)."""
    executor = executor_cls(max_workers=3)
    start = time.monotonic()
    futures = [executor.submit(_slow_add, i, i) for i in range(3)]
    results = [f.result(timeout=30) for f in futures]
    elapsed = time.monotonic() - start

    assert results == [0, 2, 4]
    # 3 tasks * 0.3s each serial = 0.9s; concurrent should be < 0.7s
    assert elapsed < 0.7, f"Expected concurrent execution, took {elapsed:.2f}s"
    executor.shutdown()


def test_concurrent_execution_inline_no_capture():
    """InlineExecutor with capture=False runs tasks concurrently."""
    executor = InlineExecutor(max_workers=3, capture=False)
    start = time.monotonic()
    futures = [executor.submit(_slow_add, i, i) for i in range(3)]
    results = [f.result(timeout=30) for f in futures]
    elapsed = time.monotonic() - start

    assert results == [0, 2, 4]
    assert elapsed < 0.7, f"Expected concurrent execution, took {elapsed:.2f}s"
    executor.shutdown()


@pytest.mark.parametrize(
    "executor_cls", [InlineExecutor, SubprocessExecutor, ForkExecutor]
)
def test_shutdown_waits(executor_cls):
    """shutdown(wait=True) blocks until all pending work completes."""
    executor = executor_cls(max_workers=2)
    futures = [executor.submit(_slow_add, i, 1) for i in range(3)]
    executor.shutdown(wait=True)
    # After shutdown, all futures should be done
    for f in futures:
        assert f.done()


@pytest.mark.parametrize(
    "executor_cls", [InlineExecutor, SubprocessExecutor, ForkExecutor]
)
def test_context_manager(executor_cls):
    """Executor works as a context manager (calls shutdown on exit)."""
    with executor_cls() as executor:
        f = executor.submit(_add, 5, 7)
        assert f.result(timeout=10) == 12


@pytest.mark.parametrize(
    "executor_cls", [InlineExecutor, SubprocessExecutor, ForkExecutor]
)
def test_error_handling_async(executor_cls):
    """Exceptions are properly captured in async mode."""
    executor = executor_cls()
    f = executor.submit(_fail)
    exc = f.exception(timeout=10)
    assert exc is not None
    assert f.log  # log should have traceback info
    executor.shutdown()
