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
EVENTS_FILE = "events.pb"  # Protobuf event file


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
        # Cache for parsed data
        self._scalars_cache: dict[str, list[tuple[int, float]]] | None = None
        self._text_cache: dict[str, list[tuple[int, str]]] | None = None
        self._figures_cache: dict[str, list[tuple[int, bytes]]] | None = None
        self._checkpoints_cache: dict[str, list[tuple[int, bytes]]] | None = None
        self._protobuf_loaded = False

    def _has_protobuf(self) -> bool:
        """Check if this run uses protobuf format."""
        return (self._log_dir / EVENTS_FILE).exists()

    def _load_protobuf_events(self) -> None:
        """Load all events from protobuf file into caches."""
        if self._protobuf_loaded:
            return

        from .event_file import EventFileReader

        events_path = self._log_dir / EVENTS_FILE
        if not events_path.exists():
            self._protobuf_loaded = True
            return

        self._scalars_cache = {}
        self._text_cache = {}
        self._figures_cache = {}
        self._checkpoints_cache = {}

        reader = EventFileReader(events_path)
        for event in reader:
            it = event.iteration
            data_type = event.WhichOneof("data")

            if data_type == "scalar":
                tag = event.scalar.tag
                if tag not in self._scalars_cache:
                    self._scalars_cache[tag] = []
                self._scalars_cache[tag].append((it, event.scalar.value))

            elif data_type == "text":
                tag = event.text.tag
                if tag not in self._text_cache:
                    self._text_cache[tag] = []
                self._text_cache[tag].append((it, event.text.value))

            elif data_type == "figure":
                tag = event.figure.tag
                if tag not in self._figures_cache:
                    self._figures_cache[tag] = []
                self._figures_cache[tag].append((it, event.figure.data))

            elif data_type == "checkpoint":
                tag = event.checkpoint.tag
                if tag not in self._checkpoints_cache:
                    self._checkpoints_cache[tag] = []
                self._checkpoints_cache[tag].append((it, event.checkpoint.data))

        # Sort by iteration
        for cache in [
            self._scalars_cache,
            self._text_cache,
            self._figures_cache,
            self._checkpoints_cache,
        ]:
            for tag in cache:
                cache[tag].sort(key=lambda x: x[0])

        self._protobuf_loaded = True

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

    def _load_scalars(self) -> dict[str, list[tuple[int, float]]]:
        """Load and cache all scalars."""
        # Try protobuf first
        if self._has_protobuf():
            self._load_protobuf_events()
            return self._scalars_cache or {}

        # Fall back to JSONL
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

    def _load_text(self) -> dict[str, list[tuple[int, str]]]:
        """Load and cache all text."""
        # Try protobuf first
        if self._has_protobuf():
            self._load_protobuf_events()
            return self._text_cache or {}

        # Fall back to JSONL
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
        for values in self._load_scalars().values():
            for it, _ in values:
                iterations.add(it)

        # Get iterations from text
        for values in self._load_text().values():
            for it, _ in values:
                iterations.add(it)

        # If using protobuf, also get from figures/checkpoints cache
        if self._has_protobuf():
            self._load_protobuf_events()
            if self._figures_cache:
                for values in self._figures_cache.values():
                    for it, _ in values:
                        iterations.add(it)
            if self._checkpoints_cache:
                for values in self._checkpoints_cache.values():
                    for it, _ in values:
                        iterations.add(it)
        else:
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
        return set(self._load_scalars().keys())

    @property
    def text_tags(self) -> set[str]:
        """Get all text tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        return set(self._load_text().keys())

    @property
    def figure_tags(self) -> set[str]:
        """Get all figure tags logged in this run."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")

        if self._has_protobuf():
            self._load_protobuf_events()
            return set(self._figures_cache.keys()) if self._figures_cache else set()

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
        return self._load_scalars().get(tag, [])

    def load_text(self, tag: str) -> list[tuple[int, str]]:
        """Load text values for a tag as (iteration, text) pairs."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")
        return self._load_text().get(tag, [])

    def load_figure(self, tag: str, iteration: int) -> Any:
        """Load a figure object for a specific tag and iteration."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")

        if self._has_protobuf():
            self._load_protobuf_events()
            if self._figures_cache and tag in self._figures_cache:
                for it, data in self._figures_cache[tag]:
                    if it == iteration:
                        return cloudpickle.loads(data)
            raise FileNotFoundError(f"Figure not found: {tag} at iteration {iteration}")

        fig_path = self._log_dir / str(iteration) / "figures" / f"{tag}.cpkl"
        if not fig_path.exists():
            raise FileNotFoundError(f"Figure not found: {tag} at iteration {iteration}")
        with open(fig_path, "rb") as f:
            return cloudpickle.load(f)

    def figure_iterations(self, tag: str) -> list[int]:
        """Get all iterations where a figure tag was logged."""
        if not self.is_run:
            raise ValueError("Not a run directory. Use get_run() first.")

        if self._has_protobuf():
            self._load_protobuf_events()
            if self._figures_cache and tag in self._figures_cache:
                return [it for it, _ in self._figures_cache[tag]]
            return []

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

        if self._has_protobuf():
            self._load_protobuf_events()
            return (
                set(self._checkpoints_cache.keys())
                if self._checkpoints_cache
                else set()
            )

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

        if self._has_protobuf():
            self._load_protobuf_events()
            if self._checkpoints_cache and tag in self._checkpoints_cache:
                for it, data in self._checkpoints_cache[tag]:
                    if it == iteration:
                        return cloudpickle.loads(data)
            raise FileNotFoundError(
                f"Checkpoint not found: {tag} at iteration {iteration}"
            )

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

        if self._has_protobuf():
            self._load_protobuf_events()
            if self._checkpoints_cache and tag in self._checkpoints_cache:
                return [
                    (it, cloudpickle.loads(data))
                    for it, data in self._checkpoints_cache[tag]
                ]
            return []

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

        if self._has_protobuf():
            self._load_protobuf_events()
            if self._checkpoints_cache and tag in self._checkpoints_cache:
                return [it for it, _ in self._checkpoints_cache[tag]]
            return []

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

    Storage structure (JSONL format, use_protobuf=False):
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

    Storage structure (Protobuf format, use_protobuf=True):
        log_dir/
        ├── .pyexp              # Marker file identifying this as a pyexp log
        └── events.pb           # All events in a single protobuf file

    Args:
        log_dir: Directory to store log files.
        use_protobuf: If True, use protobuf format (single file, faster).
                     If False, use JSONL format (human-readable, multiple files).

    Example:
        logger = Logger("/path/to/logs", use_protobuf=True)
        logger.set_global_it(100)
        logger.add_scalar("loss", 0.5)
        logger.add_text("info", "Training started")
        logger.add_figure("plot", fig, interactive=True)
        logger.flush()  # Wait for all writes to complete
    """

    def __init__(self, log_dir: str | Path, use_protobuf: bool = True):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._global_it = 0
        self._use_protobuf = use_protobuf

        # Create marker file to identify this as a pyexp run directory
        marker_path = self._log_dir / MARKER_FILE
        marker_path.touch(exist_ok=True)

        # File locks for thread-safe writing
        self._scalars_lock = threading.Lock()
        self._text_lock = threading.Lock()

        # Protobuf event file writer
        self._event_writer = None
        if use_protobuf:
            from .event_file import EventFileWriter

            self._event_writer = EventFileWriter(self._log_dir / EVENTS_FILE)

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
        if self._event_writer:
            self._event_writer.flush()

    def set_global_it(self, it: int) -> None:
        """Set the global iteration counter."""
        self._global_it = it

    def _get_timestamp(self) -> float:
        """Get current timestamp in seconds with nanosecond precision."""
        return time.time_ns() / 1e9

    def add_scalar(self, tag: str, scalar_value: float) -> None:
        """Log a scalar value at the current iteration.

        Scalars are appended to scalars.jsonl as {"it": N, "tag": "...", "value": V, "ts": T}.
        """
        ts = self._get_timestamp()
        self._queue.put(("scalar", (tag, scalar_value, self._global_it, ts)))

    def add_text(self, tag: str, text_string: str) -> None:
        """Log a text string at the current iteration.

        Text is appended to text.jsonl as {"it": N, "tag": "...", "text": "...", "ts": T}.
        """
        ts = self._get_timestamp()
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
        ts = self._get_timestamp()
        self._queue.put(("figure", (tag, figure, self._global_it, interactive, ts)))

    def add_checkpoint(self, tag: str, obj: Any) -> None:
        """Log an arbitrary object as a checkpoint at the current iteration.

        Checkpoints are saved as <iteration>/checkpoints/<tag>.cpkl using cloudpickle.
        Use this to save model weights, optimizer state, or any serializable object.

        Args:
            tag: Name/tag for the checkpoint.
            obj: The object to save (must be picklable).
        """
        ts = self._get_timestamp()
        self._queue.put(("checkpoint", (tag, obj, self._global_it, ts)))

    def _write_scalar(self, tag: str, scalar_value: float, it: int, ts: float) -> None:
        """Write a scalar value."""
        if self._use_protobuf:
            from .events_pb2 import Event, Scalar

            event = Event()
            event.timestamp = ts
            event.iteration = it
            event.scalar.CopyFrom(Scalar(tag=tag, value=scalar_value))
            self._event_writer.write(event)
        else:
            entry = {"it": it, "tag": tag, "value": scalar_value, "ts": ts}
            scalars_path = self._log_dir / SCALARS_FILE
            with self._scalars_lock:
                with open(scalars_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

    def _write_text(self, tag: str, text_string: str, it: int, ts: float) -> None:
        """Write a text string."""
        if self._use_protobuf:
            from .events_pb2 import Event, Text

            event = Event()
            event.timestamp = ts
            event.iteration = it
            event.text.CopyFrom(Text(tag=tag, value=text_string))
            self._event_writer.write(event)
        else:
            entry = {"it": it, "tag": tag, "text": text_string, "ts": ts}
            text_path = self._log_dir / TEXT_FILE
            with self._text_lock:
                with open(text_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

    def _write_figure(
        self, tag: str, figure: Any, it: int, interactive: bool, ts: float
    ) -> None:
        """Write a figure."""
        if self._use_protobuf:
            from .events_pb2 import Event, Figure

            data = cloudpickle.dumps(figure)
            event = Event()
            event.timestamp = ts
            event.iteration = it
            event.figure.CopyFrom(Figure(tag=tag, data=data, interactive=interactive))
            self._event_writer.write(event)
        else:
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
        """Write a checkpoint object."""
        if self._use_protobuf:
            from .events_pb2 import Checkpoint, Event

            data = cloudpickle.dumps(obj)
            event = Event()
            event.timestamp = ts
            event.iteration = it
            event.checkpoint.CopyFrom(Checkpoint(tag=tag, data=data))
            self._event_writer.write(event)
        else:
            it_dir = self._get_it_dir(it)
            ckpt_dir = it_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            ckpt_path = ckpt_dir / f"{tag}.cpkl"
            with open(ckpt_path, "wb") as f:
                cloudpickle.dump(obj, f)

            # Save metadata with timestamp
            meta_path = ckpt_dir / f"{tag}.meta"
            meta_path.write_text(json.dumps({"ts": ts}))
