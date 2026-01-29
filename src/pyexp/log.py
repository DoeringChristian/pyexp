"""Logging utilities for pyexp experiments."""

import atexit
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cloudpickle

# File names for storage
SCALARS_FILE = "scalars.jsonl"
TEXT_FILE = "text.jsonl"
MARKER_FILE = ".pyexp"  # Marker file identifying a pyexp run directory


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
        # Cache for parsed JSONL data
        self._scalars_cache: dict[str, list[tuple[int, float]]] | None = None
        self._text_cache: dict[str, list[tuple[int, str]]] | None = None

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

    def _load_scalars_jsonl(self) -> dict[str, list[tuple[int, float]]]:
        """Load and cache all scalars from JSONL file."""
        if self._scalars_cache is not None:
            return self._scalars_cache

        self._scalars_cache = {}
        scalars_path = self._log_dir / SCALARS_FILE
        if scalars_path.exists():
            with open(scalars_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        tag = entry["tag"]
                        if tag not in self._scalars_cache:
                            self._scalars_cache[tag] = []
                        self._scalars_cache[tag].append((entry["it"], entry["value"]))
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines

        # Sort each tag's values by iteration
        for tag in self._scalars_cache:
            self._scalars_cache[tag].sort(key=lambda x: x[0])

        return self._scalars_cache

    def _load_text_jsonl(self) -> dict[str, list[tuple[int, str]]]:
        """Load and cache all text from JSONL file."""
        if self._text_cache is not None:
            return self._text_cache

        self._text_cache = {}
        text_path = self._log_dir / TEXT_FILE
        if text_path.exists():
            with open(text_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        tag = entry["tag"]
                        if tag not in self._text_cache:
                            self._text_cache[tag] = []
                        self._text_cache[tag].append((entry["it"], entry["text"]))
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines

        # Sort each tag's values by iteration
        for tag in self._text_cache:
            self._text_cache[tag].sort(key=lambda x: x[0])

        return self._text_cache

    @property
    def iterations(self) -> list[int]:
        """Get all iteration numbers in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")

        iterations = set()

        # Get iterations from scalars
        for values in self._load_scalars_jsonl().values():
            for it, _ in values:
                iterations.add(it)

        # Get iterations from text
        for values in self._load_text_jsonl().values():
            for it, _ in values:
                iterations.add(it)

        # Get iterations from iteration directories (figures, checkpoints)
        for d in self._log_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                iterations.add(int(d.name))

        return sorted(iterations)

    @property
    def scalar_tags(self) -> set[str]:
        """Get all scalar tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        return set(self._load_scalars_jsonl().keys())

    @property
    def text_tags(self) -> set[str]:
        """Get all text tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        return set(self._load_text_jsonl().keys())

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
        return self._load_scalars_jsonl().get(tag, [])

    def load_text(self, tag: str) -> list[tuple[int, str]]:
        """Load text values for a tag as (iteration, text) pairs."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        return self._load_text_jsonl().get(tag, [])

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

    def __getitem__(self, tag: str) -> tuple[int, Any]:
        """Get the last logged value for a tag.

        Searches scalars, text, figures, and checkpoints in that order.
        Returns (iteration, value) tuple for the most recent entry.

        Example:
            it, loss = reader["loss"]
            it, fig = reader["loss_landscape"]
        """
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")

        # Check scalars
        if tag in self.scalar_tags:
            data = self.load_scalars(tag)
            if data:
                return data[-1]

        # Check text
        if tag in self.text_tags:
            data = self.load_text(tag)
            if data:
                return data[-1]

        # Check figures
        if tag in self.figure_tags:
            iterations = self.figure_iterations(tag)
            if iterations:
                last_it = iterations[-1]
                return (last_it, self.load_figure(tag, last_it))

        # Check checkpoints
        if tag in self.checkpoint_tags:
            data = self.load_checkpoints(tag)
            if data:
                return data[-1]

        raise KeyError(f"Tag not found: {tag}")

    @property
    def checkpoint_tags(self) -> set[str]:
        """Get all checkpoint tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        tags = set()
        for it in self.iterations:
            checkpoints_dir = self._log_dir / str(it) / "checkpoints"
            if checkpoints_dir.exists():
                for ckpt_path in checkpoints_dir.glob("*.cpkl"):
                    tags.add(ckpt_path.stem)
        return tags

    def load_checkpoint(self, tag: str, iteration: int) -> Any:
        """Load a checkpoint object for a specific tag and iteration."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        ckpt_path = self._log_dir / str(iteration) / "checkpoints" / f"{tag}.cpkl"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {tag} at iteration {iteration}"
            )
        with open(ckpt_path, "rb") as f:
            return cloudpickle.load(f)

    def load_checkpoints(self, tag: str) -> list[tuple[int, Any]]:
        """Load all checkpoint values for a tag as (iteration, value) pairs."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        values = []
        for it in self.iterations:
            ckpt_path = self._log_dir / str(it) / "checkpoints" / f"{tag}.cpkl"
            if ckpt_path.exists():
                with open(ckpt_path, "rb") as f:
                    values.append((it, cloudpickle.load(f)))
        return values

    def checkpoint_iterations(self, tag: str) -> list[int]:
        """Get all iterations where a checkpoint tag was logged."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        iterations = []
        for it in self.iterations:
            ckpt_path = self._log_dir / str(it) / "checkpoints" / f"{tag}.cpkl"
            if ckpt_path.exists():
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

    Storage structure:
        log_dir/
        ├── .pyexp              # Marker file identifying this as a pyexp log
        ├── scalars.jsonl       # {"it": N, "tag": "...", "value": V}
        ├── text.jsonl          # {"it": N, "tag": "...", "text": "..."}
        └── <iteration>/
            ├── figures/
            │   ├── <tag>.cpkl  # Pickled figure
            │   └── <tag>.meta  # Metadata (interactive flag)
            └── checkpoints/
                └── <tag>.cpkl  # Pickled checkpoint

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

        # Create marker file to identify this as a pyexp run directory
        marker_path = self._log_dir / MARKER_FILE
        marker_path.touch(exist_ok=True)

        # File locks for thread-safe JSONL appending
        self._scalars_lock = threading.Lock()
        self._text_lock = threading.Lock()

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
                elif op == "checkpoint":
                    self._write_checkpoint(*args)
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

        Scalars are appended to scalars.jsonl as {"it": N, "tag": "...", "value": V, "ts": T}.
        """
        ts = time.time()
        self._queue.put(("scalar", (tag, scalar_value, self._global_it, ts)))

    def add_text(self, tag: str, text_string: str) -> None:
        """Log a text string at the current iteration.

        Text is appended to text.jsonl as {"it": N, "tag": "...", "text": "...", "ts": T}.
        """
        ts = time.time()
        self._queue.put(("text", (tag, text_string, self._global_it, ts)))

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
        ts = time.time()
        self._queue.put(("figure", (tag, figure, self._global_it, interactive, ts)))

    def add_checkpoint(self, tag: str, obj: Any) -> None:
        """Log an arbitrary object as a checkpoint at the current iteration.

        Checkpoints are saved as <iteration>/checkpoints/<tag>.cpkl using cloudpickle.
        Use this to save model weights, optimizer state, or any serializable object.

        Args:
            tag: Name/tag for the checkpoint.
            obj: The object to save (must be picklable).
        """
        ts = time.time()
        self._queue.put(("checkpoint", (tag, obj, self._global_it, ts)))

    def _write_scalar(self, tag: str, scalar_value: float, it: int, ts: float) -> None:
        """Append a scalar value to scalars.jsonl."""
        entry = {"it": it, "tag": tag, "value": scalar_value, "ts": ts}
        scalars_path = self._log_dir / SCALARS_FILE
        with self._scalars_lock:
            with open(scalars_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _write_text(self, tag: str, text_string: str, it: int, ts: float) -> None:
        """Append a text string to text.jsonl."""
        entry = {"it": it, "tag": tag, "text": text_string, "ts": ts}
        text_path = self._log_dir / TEXT_FILE
        with self._text_lock:
            with open(text_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _write_figure(
        self, tag: str, figure: Any, it: int, interactive: bool, ts: float
    ) -> None:
        """Write a figure to disk."""
        it_dir = self._get_it_dir(it)
        fig_dir = it_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Save the pickled figure
        fig_path = fig_dir / f"{tag}.cpkl"
        with open(fig_path, "wb") as f:
            cloudpickle.dump(figure, f)

        # Save metadata (including timestamp)
        meta_path = fig_dir / f"{tag}.meta"
        meta_path.write_text(json.dumps({"interactive": interactive, "ts": ts}))

    def _write_checkpoint(self, tag: str, obj: Any, it: int, ts: float) -> None:
        """Write a checkpoint object to disk."""
        it_dir = self._get_it_dir(it)
        ckpt_dir = it_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / f"{tag}.cpkl"
        with open(ckpt_path, "wb") as f:
            cloudpickle.dump(obj, f)

        # Save metadata with timestamp
        meta_path = ckpt_dir / f"{tag}.meta"
        meta_path.write_text(json.dumps({"ts": ts}))
