"""Solara-based viewer for pyexp logs."""

import json
from pathlib import Path
from typing import Any


def is_run_directory(path: Path) -> bool:
    """Check if a directory is a pyexp run (contains iteration subdirectories)."""
    if not path.is_dir():
        return False
    # A run directory contains numeric subdirectories (iterations)
    for item in path.iterdir():
        if item.is_dir() and item.name.isdigit():
            return True
    return False


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


def load_iterations(log_path: Path) -> list[int]:
    """Load all iteration numbers from log directory."""
    if not log_path.exists():
        return []
    iterations = []
    for d in log_path.iterdir():
        if d.is_dir() and d.name.isdigit():
            iterations.append(int(d.name))
    return sorted(iterations)


def load_scalars_timeseries(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Load all scalars across iterations as time series."""
    timeseries: dict[str, list[tuple[int, float]]] = {}
    for it in load_iterations(log_path):
        scalars_path = log_path / str(it) / "scalars.json"
        if scalars_path.exists():
            data = json.loads(scalars_path.read_text())
            for tag, value in data.items():
                if tag not in timeseries:
                    timeseries[tag] = []
                timeseries[tag].append((it, value))
    # Sort by iteration
    for tag in timeseries:
        timeseries[tag].sort(key=lambda x: x[0])
    return timeseries


def load_iteration_data(log_path: Path, iteration: int) -> dict[str, Any]:
    """Load all data for a specific iteration."""
    it_dir = log_path / str(iteration)
    data: dict[str, Any] = {"scalars": {}, "text": {}, "figures": {}}

    scalars_path = it_dir / "scalars.json"
    if scalars_path.exists():
        data["scalars"] = json.loads(scalars_path.read_text())

    text_path = it_dir / "text.json"
    if text_path.exists():
        data["text"] = json.loads(text_path.read_text())

    figures_dir = it_dir / "figures"
    if figures_dir.exists():
        for fig_path in figures_dir.glob("*.cpkl"):
            tag = fig_path.stem
            data["figures"][tag] = fig_path  # Store path, load on demand

    return data


def load_figure(fig_path: Path) -> Any:
    """Load a figure from cloudpickle file."""
    import cloudpickle
    with open(fig_path, "rb") as f:
        return cloudpickle.load(f)


def load_text_timeseries(log_path: Path) -> dict[str, list[tuple[int, str]]]:
    """Load all text across iterations as time series."""
    timeseries: dict[str, list[tuple[int, str]]] = {}
    for it in load_iterations(log_path):
        text_path = log_path / str(it) / "text.json"
        if text_path.exists():
            data = json.loads(text_path.read_text())
            for tag, text in data.items():
                if tag not in timeseries:
                    timeseries[tag] = []
                timeseries[tag].append((it, text))
    # Sort by iteration
    for tag in timeseries:
        timeseries[tag].sort(key=lambda x: x[0])
    return timeseries


def load_figures_info(log_path: Path) -> dict[str, list[tuple[int, Path]]]:
    """Load all figure paths across iterations."""
    figures: dict[str, list[tuple[int, Path]]] = {}
    for it in load_iterations(log_path):
        figures_dir = log_path / str(it) / "figures"
        if figures_dir.exists():
            for fig_path in figures_dir.glob("*.cpkl"):
                tag = fig_path.stem
                if tag not in figures:
                    figures[tag] = []
                figures[tag].append((it, fig_path))
    # Sort by iteration
    for tag in figures:
        figures[tag].sort(key=lambda x: x[0])
    return figures


def get_all_scalar_tags(runs: list[Path]) -> set[str]:
    """Get all unique scalar tags across all runs."""
    tags = set()
    for run in runs:
        timeseries = load_scalars_timeseries(run)
        tags.update(timeseries.keys())
    return tags


def get_all_text_tags(runs: list[Path]) -> set[str]:
    """Get all unique text tags across all runs."""
    tags = set()
    for run in runs:
        timeseries = load_text_timeseries(run)
        tags.update(timeseries.keys())
    return tags


def get_all_figure_tags(runs: list[Path]) -> set[str]:
    """Get all unique figure tags across all runs."""
    tags = set()
    for run in runs:
        figures = load_figures_info(run)
        tags.update(figures.keys())
    return tags


def run(log_path: str | Path | None = None, port: int = 8765):
    """Run the viewer server.

    Args:
        log_path: Optional log directory to open on start.
        port: Port to run the server on.

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

    # Run solara with this module
    subprocess.run(
        [sys.executable, "-m", "solara", "run", "pyexp._viewer_app:Page", "--port", str(port)],
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

    print(f"Starting pyexp viewer on http://localhost:{args.port}")
    run(args.log_dir, args.port)
