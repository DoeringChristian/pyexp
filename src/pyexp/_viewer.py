"""Solara-based viewer for pyexp logs."""

import json
from pathlib import Path
from typing import Any


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
