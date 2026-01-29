"""Tests for the Logger class."""

import json

import cloudpickle
import pytest
from pyexp import Logger


def _read_jsonl(path, strip_ts=True):
    """Read a JSONL file and return list of parsed entries.

    Args:
        path: Path to the JSONL file.
        strip_ts: If True, remove 'ts' (timestamp) field from entries for comparison.
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if strip_ts:
                    entry.pop("ts", None)
                entries.append(entry)
    return entries


class TestLogger:
    """Tests for Logger functionality."""

    def test_logger_creates_directory(self, tmp_path):
        """Logger should create log_dir if it doesn't exist."""
        log_dir = tmp_path / "logs" / "nested"
        assert not log_dir.exists()

        Logger(log_dir)

        assert log_dir.exists()

    def test_set_global_it(self, tmp_path):
        """set_global_it should update the iteration counter."""
        logger = Logger(tmp_path)

        assert logger._global_it == 0

        logger.set_global_it(100)
        assert logger._global_it == 100

        logger.set_global_it(500)
        assert logger._global_it == 500

    def test_add_scalar(self, tmp_path):
        """add_scalar should append to scalars.jsonl."""
        logger = Logger(tmp_path)
        logger.set_global_it(10)
        logger.add_scalar("loss", 0.5)
        logger.flush()

        scalars_path = tmp_path / "scalars.jsonl"
        assert scalars_path.exists()

        entries = _read_jsonl(scalars_path)
        assert entries == [{"it": 10, "tag": "loss", "value": 0.5}]

    def test_add_scalar_multiple_tags_same_iteration(self, tmp_path):
        """Multiple scalars at same iteration should be in same file."""
        logger = Logger(tmp_path)
        logger.set_global_it(1)

        logger.add_scalar("loss", 0.5)
        logger.add_scalar("accuracy", 0.9)
        logger.flush()

        entries = _read_jsonl(tmp_path / "scalars.jsonl")
        assert entries == [
            {"it": 1, "tag": "loss", "value": 0.5},
            {"it": 1, "tag": "accuracy", "value": 0.9},
        ]

    def test_add_scalar_different_iterations(self, tmp_path):
        """Scalars at different iterations should all be in scalars.jsonl."""
        logger = Logger(tmp_path)

        logger.set_global_it(1)
        logger.add_scalar("loss", 1.0)

        logger.set_global_it(2)
        logger.add_scalar("loss", 0.8)

        logger.set_global_it(3)
        logger.add_scalar("loss", 0.5)
        logger.flush()

        entries = _read_jsonl(tmp_path / "scalars.jsonl")
        assert entries == [
            {"it": 1, "tag": "loss", "value": 1.0},
            {"it": 2, "tag": "loss", "value": 0.8},
            {"it": 3, "tag": "loss", "value": 0.5},
        ]

    def test_add_text(self, tmp_path):
        """add_text should append to text.jsonl."""
        logger = Logger(tmp_path)
        logger.set_global_it(10)
        logger.add_text("info", "Training started")
        logger.flush()

        text_path = tmp_path / "text.jsonl"
        assert text_path.exists()

        entries = _read_jsonl(text_path)
        assert entries == [{"it": 10, "tag": "info", "text": "Training started"}]

    def test_add_text_multiple_tags_same_iteration(self, tmp_path):
        """Multiple texts at same iteration should be in same file."""
        logger = Logger(tmp_path)
        logger.set_global_it(1)

        logger.add_text("info", "Info message")
        logger.add_text("debug", "Debug message")
        logger.flush()

        entries = _read_jsonl(tmp_path / "text.jsonl")
        assert entries == [
            {"it": 1, "tag": "info", "text": "Info message"},
            {"it": 1, "tag": "debug", "text": "Debug message"},
        ]

    def test_add_text_different_iterations(self, tmp_path):
        """Texts at different iterations should all be in text.jsonl."""
        logger = Logger(tmp_path)

        logger.set_global_it(1)
        logger.add_text("log", "First message")

        logger.set_global_it(5)
        logger.add_text("log", "Second message")
        logger.flush()

        entries = _read_jsonl(tmp_path / "text.jsonl")
        assert entries == [
            {"it": 1, "tag": "log", "text": "First message"},
            {"it": 5, "tag": "log", "text": "Second message"},
        ]

    def test_add_figure(self, tmp_path):
        """add_figure should save figure to iteration folder."""
        logger = Logger(tmp_path)
        logger.set_global_it(100)

        figure = {"type": "figure", "data": [1, 2, 3]}
        logger.add_figure("plot", figure)
        logger.flush()

        fig_path = tmp_path / "100" / "figures" / "plot.cpkl"
        assert fig_path.exists()

        with open(fig_path, "rb") as f:
            loaded = cloudpickle.load(f)
        assert loaded == figure

    def test_add_figure_multiple_tags_same_iteration(self, tmp_path):
        """Multiple figures at same iteration should be separate files in same folder."""
        logger = Logger(tmp_path)
        logger.set_global_it(1)

        logger.add_figure("loss_curve", {"type": "loss"})
        logger.add_figure("accuracy_curve", {"type": "acc"})
        logger.flush()

        assert (tmp_path / "1" / "figures" / "loss_curve.cpkl").exists()
        assert (tmp_path / "1" / "figures" / "accuracy_curve.cpkl").exists()

    def test_add_figure_different_iterations(self, tmp_path):
        """Figures at different iterations should be in separate folders."""
        logger = Logger(tmp_path)

        logger.set_global_it(1)
        logger.add_figure("plot", {"value": 1})

        logger.set_global_it(10)
        logger.add_figure("plot", {"value": 10})
        logger.flush()

        with open(tmp_path / "1" / "figures" / "plot.cpkl", "rb") as f:
            assert cloudpickle.load(f) == {"value": 1}

        with open(tmp_path / "10" / "figures" / "plot.cpkl", "rb") as f:
            assert cloudpickle.load(f) == {"value": 10}

    def test_logger_default_iteration_zero(self, tmp_path):
        """Logger should start with global_it = 0."""
        logger = Logger(tmp_path)

        logger.add_scalar("test", 1.0)
        logger.flush()

        entries = _read_jsonl(tmp_path / "scalars.jsonl")
        assert entries == [{"it": 0, "tag": "test", "value": 1.0}]

    def test_mixed_types_same_iteration(self, tmp_path):
        """Scalars, text, and figures at same iteration should coexist."""
        logger = Logger(tmp_path)
        logger.set_global_it(42)

        logger.add_scalar("loss", 0.5)
        logger.add_text("status", "running")
        logger.add_figure("chart", {"data": [1, 2, 3]})
        logger.flush()

        scalar_entries = _read_jsonl(tmp_path / "scalars.jsonl")
        assert scalar_entries == [{"it": 42, "tag": "loss", "value": 0.5}]

        text_entries = _read_jsonl(tmp_path / "text.jsonl")
        assert text_entries == [{"it": 42, "tag": "status", "text": "running"}]

        with open(tmp_path / "42" / "figures" / "chart.cpkl", "rb") as f:
            assert cloudpickle.load(f) == {"data": [1, 2, 3]}

    def test_flush_waits_for_pending_writes(self, tmp_path):
        """flush() should block until all writes are complete."""
        logger = Logger(tmp_path)

        # Queue many operations across different iterations
        for i in range(100):
            logger.set_global_it(i)
            logger.add_scalar("test", float(i))

        logger.flush()

        # All entries should be in scalars.jsonl
        entries = _read_jsonl(tmp_path / "scalars.jsonl")
        assert len(entries) == 100
        for i, entry in enumerate(entries):
            assert entry == {"it": i, "tag": "test", "value": float(i)}

    def test_async_saving_does_not_block(self, tmp_path):
        """add_* methods should return immediately without blocking."""
        import time

        logger = Logger(tmp_path)

        start = time.perf_counter()
        for i in range(100):
            logger.add_scalar("test", float(i))
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms (I/O would take longer)
        assert elapsed < 0.1

        logger.flush()

    def test_marker_file_created(self, tmp_path):
        """Logger should create .pyexp marker file."""
        Logger(tmp_path)
        assert (tmp_path / ".pyexp").exists()
