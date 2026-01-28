"""Solara-based viewer for pyexp logs."""

import json
import pickle
from pathlib import Path
from typing import Any

from pyexp.log import MARKER_FILE, SCALARS_FILE, TEXT_FILE


def is_run_directory(path: Path) -> bool:
    """Check if a directory is a pyexp run (contains marker file)."""
    if not path.is_dir():
        return False
    return (path / MARKER_FILE).exists()


def discover_runs(root_path: Path) -> list[Path]:
    """Recursively discover all run directories under the given root."""
    runs = []
    if not root_path.exists():
        return runs

    # Check if root itself is a run
    if is_run_directory(root_path):
        runs.append(root_path)

    # Recursively search subdirectories
    for item in root_path.iterdir():
        if item.is_dir() and not item.name.isdigit():  # Skip iteration directories
            if is_run_directory(item):
                runs.append(item)
            else:
                # Recurse into non-run directories
                runs.extend(discover_runs(item))

    return sorted(runs)


def load_scalars_timeseries(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Load all scalars from JSONL file as time series."""
    timeseries: dict[str, list[tuple[int, float]]] = {}
    scalars_path = log_path / SCALARS_FILE

    if scalars_path.exists():
        try:
            with open(scalars_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        tag = entry["tag"]
                        if tag not in timeseries:
                            timeseries[tag] = []
                        timeseries[tag].append((entry["it"], entry["value"]))
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines
        except (IOError, OSError):
            pass  # File is being written to

    # Sort by iteration
    for tag in timeseries:
        timeseries[tag].sort(key=lambda x: x[0])

    return timeseries


def load_text_timeseries(log_path: Path) -> dict[str, list[tuple[int, str]]]:
    """Load all text from JSONL file as time series."""
    timeseries: dict[str, list[tuple[int, str]]] = {}
    text_path = log_path / TEXT_FILE

    if text_path.exists():
        try:
            with open(text_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        tag = entry["tag"]
                        if tag not in timeseries:
                            timeseries[tag] = []
                        timeseries[tag].append((entry["it"], entry["text"]))
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines
        except (IOError, OSError):
            pass  # File is being written to

    # Sort by iteration
    for tag in timeseries:
        timeseries[tag].sort(key=lambda x: x[0])

    return timeseries


def load_iterations(log_path: Path) -> list[int]:
    """Load all iteration numbers from log directory.

    Combines iterations from JSONL files and iteration directories.
    """
    if not log_path.exists():
        return []

    iterations = set()

    # Get iterations from scalars JSONL
    for values in load_scalars_timeseries(log_path).values():
        for it, _ in values:
            iterations.add(it)

    # Get iterations from text JSONL
    for values in load_text_timeseries(log_path).values():
        for it, _ in values:
            iterations.add(it)

    # Get iterations from directories (figures, checkpoints)
    for d in log_path.iterdir():
        if d.is_dir() and d.name.isdigit():
            iterations.add(int(d.name))

    return sorted(iterations)


def load_figure(fig_path: Path) -> Any:
    """Load a figure from cloudpickle file."""
    import cloudpickle

    with open(fig_path, "rb") as f:
        try:
            return cloudpickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            # File is being written to
            return None


def load_figure_meta(fig_path: Path) -> dict:
    """Load metadata for a figure."""
    meta_path = fig_path.with_suffix(".meta")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass  # File is being written to
    # Default to interactive for backwards compatibility
    return {"interactive": True}


def load_figures_info(log_path: Path) -> dict[str, list[tuple[int, Path, bool]]]:
    """Load all figure paths and metadata across iterations.

    Returns:
        Dict mapping tag to list of (iteration, path, interactive) tuples.
    """
    figures: dict[str, list[tuple[int, Path, bool]]] = {}
    for it in load_iterations(log_path):
        figures_dir = log_path / str(it) / "figures"
        if figures_dir.exists():
            for fig_path in figures_dir.glob("*.cpkl"):
                tag = fig_path.stem
                meta = load_figure_meta(fig_path)
                if tag not in figures:
                    figures[tag] = []
                figures[tag].append((it, fig_path, meta.get("interactive", True)))
    # Sort by iteration
    for tag in figures:
        figures[tag].sort(key=lambda x: x[0])
    return figures


def _find_free_port(start: int, max_attempts: int = 100) -> int:
    """Find a free port starting from the given port number."""
    import socket

    for offset in range(max_attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + max_attempts - 1}")


def run(log_path: str | Path | None = None, port: int = 8765):
    """Run the viewer server.

    Args:
        log_path: Optional log directory to open on start.
        port: Port to run the server on. If busy, the next free port is used.

    Example:
        from pyexp.viewer import run
        run("/path/to/logs")
    """
    import os
    import subprocess
    import sys

    if log_path:
        # Set environment variable so the Page component can read it
        os.environ["PYEXP_LOG_DIR"] = str(Path(log_path).absolute())

    port = _find_free_port(port)
    print(f"Starting pyexp viewer on http://localhost:{port}")

    # Run solara with this module
    subprocess.run(
        [
            sys.executable,
            "-m",
            "solara",
            "run",
            "pyexp._viewer_app:Page",
            "--host",
            "localhost",
            "--port",
            str(port),
        ],
        check=True,
    )


def main():
    """CLI entry point for the viewer."""
    import argparse

    parser = argparse.ArgumentParser(description="pyexp Log Viewer")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=None,
        help="Log directory to view",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to run the server on (default: 8765)",
    )
    args = parser.parse_args()

    run(args.log_dir, args.port)
