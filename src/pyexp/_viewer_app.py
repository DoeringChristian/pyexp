"""Solara app components for pyexp viewer.

This module is loaded by solara run, so it can import solara at the top level.
"""

import os
from pathlib import Path

import cloudpickle
import solara

from pyexp._viewer import (
    load_figure,
    load_iteration_data,
    load_iterations,
    load_scalars_timeseries,
)

# Reactive state
log_dir = solara.reactive("")
selected_iteration = solara.reactive(None)
refresh_counter = solara.reactive(0)


@solara.component
def InteractiveFigure(fig_path: Path):
    """Display a matplotlib figure interactively using ipympl."""
    import matplotlib
    matplotlib.use('module://ipympl.backend_nbagg')
    import matplotlib.pyplot as plt
    from ipympl.backend_nbagg import Canvas, FigureManager

    # Load the pickled figure
    mpl_fig = load_figure(fig_path)

    # Create an interactive canvas for the figure
    canvas = Canvas(mpl_fig)
    manager = FigureManager(canvas, 0)

    # Display the canvas widget
    solara.display(canvas)


@solara.component
def ScalarPlots():
    """Display scalar time series plots."""
    _ = refresh_counter.value  # Trigger refresh

    if not log_dir.value:
        return

    log_path = Path(log_dir.value)
    timeseries = load_scalars_timeseries(log_path)

    if not timeseries:
        solara.Text("No scalars logged yet.")
        return

    import matplotlib
    matplotlib.use('module://ipympl.backend_nbagg')
    import matplotlib.pyplot as plt
    from ipympl.backend_nbagg import Canvas, FigureManager

    for tag, values in timeseries.items():
        iterations = [v[0] for v in values]
        vals = [v[1] for v in values]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(iterations, vals, marker=".", markersize=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)

        canvas = Canvas(fig)
        manager = FigureManager(canvas, 0)
        solara.display(canvas)


@solara.component
def IterationBrowser():
    """Browse and view individual iterations."""
    _ = refresh_counter.value  # Trigger refresh

    if not log_dir.value:
        return

    log_path = Path(log_dir.value)
    iterations = load_iterations(log_path)

    if not iterations:
        solara.Text("No iterations logged yet.")
        return

    # Iteration selector
    solara.Select(
        label="Iteration",
        value=selected_iteration,
        values=iterations,
    )

    if selected_iteration.value is None:
        return

    data = load_iteration_data(log_path, selected_iteration.value)

    # Display scalars
    if data["scalars"]:
        solara.Markdown("### Scalars")
        with solara.Columns([1, 1, 1]):
            for tag, value in data["scalars"].items():
                solara.Info(f"**{tag}**: {value}")

    # Display text
    if data["text"]:
        solara.Markdown("### Text")
        for tag, text in data["text"].items():
            with solara.Card(title=tag):
                solara.Markdown(f"```\n{text}\n```")

    # Display figures
    if data["figures"]:
        solara.Markdown("### Figures")
        for tag, fig_path in data["figures"].items():
            solara.Markdown(f"**{tag}**")
            try:
                InteractiveFigure(fig_path)
            except Exception as e:
                import traceback
                solara.Error(f"Failed: {e}\n{traceback.format_exc()}")


@solara.component
def RefreshButton():
    """Manual refresh button."""
    def do_refresh():
        refresh_counter.set(refresh_counter.value + 1)

    solara.Button("Refresh", on_click=do_refresh, icon_name="mdi-refresh")


@solara.component
def Page():
    """Main viewer page."""
    # Initialize from environment variable if set
    if not log_dir.value and os.environ.get("PYEXP_LOG_DIR"):
        log_dir.set(os.environ["PYEXP_LOG_DIR"])

    solara.Title("pyexp Log Viewer")

    with solara.AppBar():
        solara.Text("pyexp Log Viewer")

    with solara.Sidebar():
        solara.Markdown("## Settings")
        solara.InputText(
            label="Log Directory",
            value=log_dir,
        )

        if log_dir.value:
            log_path = Path(log_dir.value)
            if log_path.exists():
                iterations = load_iterations(log_path)
                solara.Info(f"{len(iterations)} iterations")
            else:
                solara.Warning("Directory not found")

        RefreshButton()

    if not log_dir.value:
        solara.Markdown("## Welcome to pyexp Log Viewer")
        solara.Markdown("Enter a log directory path in the sidebar to get started.")
        return

    log_path = Path(log_dir.value)
    if not log_path.exists():
        solara.Warning(f"Directory not found: {log_dir.value}")
        return

    with solara.Card(title="Scalar Plots"):
        ScalarPlots()

    with solara.Card(title="Iteration Browser"):
        IterationBrowser()
