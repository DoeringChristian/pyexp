"""Tests for pyexp executors (general-purpose function execution)."""

import tempfile
from pathlib import Path

from pyexp.executors import FnResult, SubprocessExecutor, InlineExecutor, ForkExecutor


def _add(a, b):
    return a + b


def test_subprocess_basic():
    """SubprocessExecutor runs a simple function and returns its result."""
    executor = SubprocessExecutor()
    r = executor.run(_add, 2, 3)
    assert isinstance(r, FnResult)
    assert r.result == 5
    assert r.ok


def test_subprocess_kwargs():
    """Keyword arguments are forwarded correctly."""
    executor = SubprocessExecutor()
    r = executor.run(_add, 1, b=10)
    assert r.result == 11


def _fail():
    raise ValueError("boom")


def test_subprocess_error_captures_traceback():
    """When fn raises, traceback appears in log."""
    executor = SubprocessExecutor()
    r = executor.run(_fail)
    assert not r.ok
    assert "ValueError" in r.log
    assert "boom" in r.log


def _print_and_return(msg):
    print(msg)
    return msg


def test_subprocess_capture_output():
    """Captured stdout appears in .log."""
    executor = SubprocessExecutor()
    r = executor.run(_print_and_return, "hello world")
    assert r.result == "hello world"
    assert "hello world" in r.log


def test_inline_basic():
    """InlineExecutor runs in the same process."""
    executor = InlineExecutor()
    r = executor.run(_add, 2, 3)
    assert r.result == 5
    assert r.ok


def test_inline_error():
    """InlineExecutor captures errors."""
    executor = InlineExecutor()
    r = executor.run(_fail)
    assert not r.ok
    assert "ValueError" in r.log
    assert "boom" in r.log


def test_inline_capture_output():
    """InlineExecutor captures stdout."""
    executor = InlineExecutor()
    r = executor.run(_print_and_return, "hello inline")
    assert r.result == "hello inline"
    assert "hello inline" in r.log


def test_fork_basic():
    """ForkExecutor runs in a forked process."""
    executor = ForkExecutor()
    r = executor.run(_add, 2, 3)
    assert r.result == 5
    assert r.ok


def test_fork_error():
    """ForkExecutor captures errors."""
    executor = ForkExecutor()
    r = executor.run(_fail)
    assert not r.ok
    assert "ValueError" in r.log
    assert "boom" in r.log


def test_fork_capture_output():
    """ForkExecutor captures stdout."""
    executor = ForkExecutor()
    r = executor.run(_print_and_return, "hello fork")
    assert r.result == "hello fork"
    assert "hello fork" in r.log
