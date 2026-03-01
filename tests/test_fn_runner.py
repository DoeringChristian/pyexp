"""Tests for pyexp executors (general-purpose function execution)."""

import tempfile
from pathlib import Path

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
