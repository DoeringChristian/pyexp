"""Logging utilities for pyexp experiments."""

import atexit
import json
import queue
import threading
from pathlib import Path
from typing import Any

import cloudpickle


MARKER_FILE = ".pyexp"


class LogReader:
    """Reader for exploring and loading pyexp logs.

    Args:
        log_dir: Path to a log directory (single run or parent of multiple runs).

    Example:
        # Load a single run
        reader = LogReader("/path/to/logs/run1")
        print(reader.iterations)  # [0, 1, 2, ...]
        print(reader.scalar_tags)  # ['loss', 'accuracy', ...]

        # Load scalars as time series
        loss = reader.load_scalars("loss")  # [(0, 0.5), (1, 0.4), ...]

        # Load a figure
        fig = reader.load_figure("plot", iteration=100)

        # Discover and load multiple runs
        reader = LogReader("/path/to/logs")
        print(reader.runs)  # ['run1', 'run2', ...]
        run1 = reader.get_run("run1")
    """

    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        if not self._log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")

    @property
    def path(self) -> Path:
        """Return the log directory path."""
        return self._log_dir

    @property
    def is_run(self) -> bool:
        """Check if this directory is a pyexp run (has marker file)."""
        return (self._log_dir / MARKER_FILE).exists()

    @property
    def runs(self) -> list[str]:
        """Discover all run names under this directory."""
        if self.is_run:
            return ["."]
        runs = []
        for item in self._log_dir.rglob(MARKER_FILE):
            run_path = item.parent
            runs.append(str(run_path.relative_to(self._log_dir)))
        return sorted(runs)

    def get_run(self, name: str) -> "LogReader":
        """Get a LogReader for a specific run."""
        if name == ".":
            return self
        return LogReader(self._log_dir / name)

    @property
    def iterations(self) -> list[int]:
        """Get all iteration numbers in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        iterations = []
        for d in self._log_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                iterations.append(int(d.name))
        return sorted(iterations)

    @property
    def scalar_tags(self) -> set[str]:
        """Get all scalar tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        tags = set()
        for it in self.iterations:
            scalars_path = self._log_dir / str(it) / "scalars.json"
            if scalars_path.exists():
                data = json.loads(scalars_path.read_text())
                tags.update(data.keys())
        return tags

    @property
    def text_tags(self) -> set[str]:
        """Get all text tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        tags = set()
        for it in self.iterations:
            text_path = self._log_dir / str(it) / "text.json"
            if text_path.exists():
                data = json.loads(text_path.read_text())
                tags.update(data.keys())
        return tags

    @property
    def figure_tags(self) -> set[str]:
        """Get all figure tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        tags = set()
        for it in self.iterations:
            figures_dir = self._log_dir / str(it) / "figures"
            if figures_dir.exists():
                for fig_path in figures_dir.glob("*.cpkl"):
                    tags.add(fig_path.stem)
        return tags

    def load_scalars(self, tag: str) -> list[tuple[int, float]]:
        """Load scalar values for a tag as (iteration, value) pairs."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        values = []
        for it in self.iterations:
            scalars_path = self._log_dir / str(it) / "scalars.json"
            if scalars_path.exists():
                data = json.loads(scalars_path.read_text())
                if tag in data:
                    values.append((it, data[tag]))
        return values

    def load_text(self, tag: str) -> list[tuple[int, str]]:
        """Load text values for a tag as (iteration, text) pairs."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        values = []
        for it in self.iterations:
            text_path = self._log_dir / str(it) / "text.json"
            if text_path.exists():
                data = json.loads(text_path.read_text())
                if tag in data:
                    values.append((it, data[tag]))
        return values

    def load_figure(self, tag: str, iteration: int) -> Any:
        """Load a figure object for a specific tag and iteration."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        fig_path = self._log_dir / str(iteration) / "figures" / f"{tag}.cpkl"
        if not fig_path.exists():
            raise FileNotFoundError(f"Figure not found: {tag} at iteration {iteration}")
        with open(fig_path, "rb") as f:
            return cloudpickle.load(f)

    def figure_iterations(self, tag: str) -> list[int]:
        """Get all iterations where a figure tag was logged."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        iterations = []
        for it in self.iterations:
            fig_path = self._log_dir / str(it) / "figures" / f"{tag}.cpkl"
            if fig_path.exists():
                iterations.append(it)
        return iterations

    def __repr__(self) -> str:
        if self.is_run:
            return f"LogReader('{self._log_dir}', iterations={len(self.iterations)})"
        return f"LogReader('{self._log_dir}', runs={self.runs})"


class Logger:
    """Logger for tracking scalars, text, and figures during experiments.

    Saving is performed asynchronously in a background thread to avoid
    blocking the main training loop. Use flush() to wait for pending writes.

    Storage structure (organized by iteration):
        log_dir/
        ├── .pyexp              # Marker file identifying this as a pyexp log
        └── <iteration>/
            ├── scalars.json    # {tag: value, ...}
            ├── text.json       # {tag: text, ...}
            └── figures/
                ├── <tag>.cpkl  # Pickled figure
                └── <tag>.meta  # Metadata (interactive flag)

    Args:
        log_dir: Directory to store log files.

    Example:
        logger = Logger("/path/to/logs")
        logger.set_global_it(100)
        logger.add_scalar("loss", 0.5)
        logger.add_text("info", "Training started")
        logger.add_figure("plot", fig, interactive=True)
        logger.flush()  # Wait for all writes to complete
    """

    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._global_it = 0

        # Create marker file to identify this as a pyexp log directory
        marker_path = self._log_dir / MARKER_FILE
        marker_path.touch()

        # Async saving infrastructure
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # Register auto-flush on exit
        atexit.register(self.flush)

    def _get_it_dir(self, it: int) -> Path:
        """Get the directory for a specific iteration."""
        return self._log_dir / str(it)

    def _worker_loop(self) -> None:
        """Background worker that processes save operations."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            op, args = task
            try:
                if op == "scalar":
                    self._write_scalar(*args)
                elif op == "text":
                    self._write_text(*args)
                elif op == "figure":
                    self._write_figure(*args)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Wait for all pending writes to complete."""
        self._queue.join()

    def set_global_it(self, it: int) -> None:
        """Set the global iteration counter."""
        self._global_it = it

    def add_scalar(self, tag: str, scalar_value: float) -> None:
        """Log a scalar value at the current iteration.

        Scalars are stored in <iteration>/scalars.json as {tag: value, ...}.
        """
        self._queue.put(("scalar", (tag, scalar_value, self._global_it)))

    def add_text(self, tag: str, text_string: str) -> None:
        """Log a text string at the current iteration.

        Text is stored in <iteration>/text.json as {tag: text, ...}.
        """
        self._queue.put(("text", (tag, text_string, self._global_it)))

    def add_figure(self, tag: str, figure: Any, interactive: bool = True) -> None:
        """Log a figure object at the current iteration.

        Figures are saved as <iteration>/figures/<tag>.cpkl,
        preserving the full object for later loading and modification.

        Args:
            tag: Name/tag for the figure.
            figure: The figure object (e.g., matplotlib figure).
            interactive: If True, render as interactive widget in viewer.
                        If False, render as static image (faster loading).
        """
        self._queue.put(("figure", (tag, figure, self._global_it, interactive)))

    def _write_scalar(self, tag: str, scalar_value: float, it: int) -> None:
        """Write a scalar value to disk."""
        it_dir = self._get_it_dir(it)
        it_dir.mkdir(parents=True, exist_ok=True)

        scalars_path = it_dir / "scalars.json"

        # Load existing or create new
        if scalars_path.exists():
            data = json.loads(scalars_path.read_text())
        else:
            data = {}

        data[tag] = scalar_value
        scalars_path.write_text(json.dumps(data, indent=2))

    def _write_text(self, tag: str, text_string: str, it: int) -> None:
        """Write a text string to disk."""
        it_dir = self._get_it_dir(it)
        it_dir.mkdir(parents=True, exist_ok=True)

        text_path = it_dir / "text.json"

        # Load existing or create new
        if text_path.exists():
            data = json.loads(text_path.read_text())
        else:
            data = {}

        data[tag] = text_string
        text_path.write_text(json.dumps(data, indent=2))

    def _write_figure(self, tag: str, figure: Any, it: int, interactive: bool) -> None:
        """Write a figure to disk."""
        it_dir = self._get_it_dir(it)
        fig_dir = it_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Save the pickled figure
        fig_path = fig_dir / f"{tag}.cpkl"
        with open(fig_path, "wb") as f:
            cloudpickle.dump(figure, f)

        # Save metadata
        meta_path = fig_dir / f"{tag}.meta"
        meta_path.write_text(json.dumps({"interactive": interactive}))
