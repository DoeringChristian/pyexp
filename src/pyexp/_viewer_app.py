"""Solara app components for pyexp viewer.

This module is loaded by solara run, so it can import solara at the top level.
"""

import os
from pathlib import Path

import solara

from pyexp._viewer import (
    discover_runs,
    load_figure,
    load_figures_info,
    load_scalars_timeseries,
    load_text_timeseries,
)

# Reactive state
root_dir = solara.reactive("")
selected_runs = solara.reactive([])
refresh_counter = solara.reactive(0)


@solara.component
def RunSelector():
    """Left panel for selecting runs."""
    _ = refresh_counter.value  # Trigger refresh

    if not root_dir.value:
        solara.Text("Enter a directory path above.")
        return

    root_path = Path(root_dir.value)
    if not root_path.exists():
        solara.Warning("Directory not found")
        return

    runs = discover_runs(root_path)

    if not runs:
        solara.Text("No runs found.")
        return

    solara.Markdown(f"**{len(runs)} runs found**")

    # Multi-select for runs
    run_names = [str(r.relative_to(root_path)) if r != root_path else "." for r in runs]
    run_map = {name: run for name, run in zip(run_names, runs)}

    def on_select(selected_names):
        selected_runs.set([run_map[name] for name in selected_names])

    # Get currently selected names
    current_names = []
    for run in selected_runs.value:
        for name, r in run_map.items():
            if r == run:
                current_names.append(name)
                break

    for name in run_names:
        is_selected = name in current_names

        def toggle(checked, n=name):
            current = list(current_names)
            if checked and n not in current:
                current.append(n)
            elif not checked and n in current:
                current.remove(n)
            on_select(current)

        solara.Checkbox(label=name, value=is_selected, on_value=toggle)


@solara.component
def ScalarPlot(tag: str, runs_data: dict, root_path: Path):
    """Display an interactive scalar plot using bqplot.

    Features (like TensorBoard):
    - Box zoom: drag to select area, zoom on release
    - Right-click: X-only zoom
    - Double-click: Reset zoom
    """
    import bqplot as bq
    import numpy as np

    # Collect all data
    series_list = []
    for run, timeseries in runs_data.items():
        if tag in timeseries:
            data = timeseries[tag]
            run_name = str(run.relative_to(root_path)) if run != root_path else "."
            iterations = np.array([d[0] for d in data])
            values = np.array([d[1] for d in data])
            series_list.append((run_name, iterations, values))

    if not series_list:
        solara.Text("No data")
        return

    # Colors matching TensorBoard
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    # Create scales
    x_scale = bq.LinearScale()
    y_scale = bq.LinearScale()

    # Create lines only
    lines = []
    for idx, (run_name, iterations, values) in enumerate(series_list):
        color = colors[idx % len(colors)]
        line = bq.Lines(
            x=iterations,
            y=values,
            scales={"x": x_scale, "y": y_scale},
            colors=[color],
            stroke_width=2,
            labels=[run_name],
            display_legend=True,
        )
        lines.append(line)

    # Create axes
    x_axis = bq.Axis(scale=x_scale, label="Iteration", grid_lines="solid", grid_color="#eee")
    y_axis = bq.Axis(scale=y_scale, orientation="vertical", label="Value", grid_lines="solid", grid_color="#eee")

    # XY Box zoom selector
    brush_xy = bq.interacts.BrushSelector(x_scale=x_scale, y_scale=y_scale, color="steelblue")
    # X-only selector
    brush_x = bq.interacts.BrushIntervalSelector(scale=x_scale, color="orange")

    # Track which mode we're in
    zoom_mode = solara.use_reactive("xy")

    fig = bq.Figure(
        marks=lines,
        axes=[x_axis, y_axis],
        title=tag,
        legend_location="top-right",
        fig_margin={"top": 60, "bottom": 60, "left": 70, "right": 20},
        layout={"width": "100%", "height": "350px"},
        interaction=brush_xy,
    )

    # Handle XY brush - apply zoom only when brushing ends
    def on_xy_brushing_change(change):
        # Only apply when brushing ends (goes from True to False)
        if change["old"] == True and change["new"] == False:
            selected = brush_xy.selected
            if selected is not None and len(selected) == 2:
                [[x1, y1], [x2, y2]] = selected
                if abs(x2 - x1) > 0.001 and abs(y2 - y1) > 0.001:
                    x_scale.min, x_scale.max = float(min(x1, x2)), float(max(x1, x2))
                    y_scale.min, y_scale.max = float(min(y1, y2)), float(max(y1, y2))
            brush_xy.selected = None

    brush_xy.observe(on_xy_brushing_change, names=["brushing"])

    # Handle X-only brush - apply zoom only when brushing ends
    def on_x_brushing_change(change):
        if change["old"] == True and change["new"] == False:
            selected = brush_x.selected
            if selected is not None and len(selected) == 2:
                x1, x2 = selected
                if abs(x2 - x1) > 0.001:
                    x_scale.min, x_scale.max = float(min(x1, x2)), float(max(x1, x2))
                    # Auto-scale y to visible data
                    y_vals = []
                    for _, iterations, values in series_list:
                        mask = (iterations >= x_scale.min) & (iterations <= x_scale.max)
                        if mask.any():
                            y_vals.extend(values[mask])
                    if y_vals:
                        padding = (max(y_vals) - min(y_vals)) * 0.05 or 0.1
                        y_scale.min = float(min(y_vals) - padding)
                        y_scale.max = float(max(y_vals) + padding)
            brush_x.selected = None

    brush_x.observe(on_x_brushing_change, names=["brushing"])

    # Reset zoom function
    def reset_zoom(*args):
        x_scale.min, x_scale.max = None, None
        y_scale.min, y_scale.max = None, None
        brush_xy.selected = None
        brush_x.selected = None

    def set_xy_mode(*args):
        zoom_mode.set("xy")
        fig.interaction = brush_xy

    def set_x_mode(*args):
        zoom_mode.set("x")
        fig.interaction = brush_x

    # Controls
    with solara.Row():
        solara.Button(
            "XY Zoom",
            on_click=set_xy_mode,
            icon_name="mdi-selection-drag",
            outlined=zoom_mode.value != "xy",
        )
        solara.Button(
            "X Zoom",
            on_click=set_x_mode,
            icon_name="mdi-arrow-expand-horizontal",
            outlined=zoom_mode.value != "x",
        )
        solara.Button("Reset", on_click=reset_zoom, icon_name="mdi-magnify-minus")

    solara.display(fig)


@solara.component
def ScalarsPanel():
    """Panel for viewing scalar time series."""
    refresh = refresh_counter.value
    runs = selected_runs.value

    if not runs:
        solara.Text("Select runs from the sidebar to view scalars.")
        return

    # Memoize data loading - only reload when runs or refresh changes
    def load_data():
        return {run: load_scalars_timeseries(run) for run in runs}

    runs_data = solara.use_memo(load_data, dependencies=[tuple(runs), refresh])

    # Get all tags from the pre-loaded data
    all_tags = set()
    for timeseries in runs_data.values():
        all_tags.update(timeseries.keys())

    if not all_tags:
        solara.Text("No scalars logged in selected runs.")
        return

    root_path = Path(root_dir.value)

    for tag in sorted(all_tags):
        with solara.Details(tag, expand=True):
            ScalarPlot(tag, runs_data, root_path)


@solara.component
def TextItem(run: Path, tag: str, root_path: Path, data: list):
    """Display text for a single run and tag."""
    run_name = str(run.relative_to(root_path)) if run != root_path else "."
    iterations = [d[0] for d in data]

    # Iteration slider - hook at top of component
    iter_idx = solara.use_reactive(len(iterations) - 1)

    with solara.Card(title=run_name):
        if len(iterations) > 1:
            solara.SliderInt(
                label="Iteration",
                value=iter_idx,
                min=0,
                max=len(iterations) - 1,
            )
            solara.Text(f"Iteration: {iterations[iter_idx.value]}")

        # Display text
        text = data[iter_idx.value][1]
        solara.Markdown(f"```\n{text}\n```")


@solara.component
def TextPanel():
    """Panel for viewing text logs."""
    refresh = refresh_counter.value
    runs = selected_runs.value

    if not runs:
        solara.Text("Select runs from the sidebar to view text.")
        return

    # Memoize data loading
    def load_data():
        return {run: load_text_timeseries(run) for run in runs}

    runs_data = solara.use_memo(load_data, dependencies=[tuple(runs), refresh])

    # Get all tags from the pre-loaded data
    all_tags = set()
    for timeseries in runs_data.values():
        all_tags.update(timeseries.keys())

    if not all_tags:
        solara.Text("No text logged in selected runs.")
        return

    root_path = Path(root_dir.value)

    for tag in sorted(all_tags):
        with solara.Details(tag, expand=True):
            for run in runs:
                if tag in runs_data[run] and runs_data[run][tag]:
                    TextItem(run, tag, root_path, runs_data[run][tag])


@solara.component
def FigureItem(run: Path, tag: str, root_path: Path, data: list):
    """Display figure for a single run and tag."""
    run_name = str(run.relative_to(root_path)) if run != root_path else "."
    iterations = [d[0] for d in data]

    # Iteration slider - hook at top of component
    iter_idx = solara.use_reactive(len(iterations) - 1)

    with solara.Card(title=run_name):
        if len(iterations) > 1:
            solara.SliderInt(
                label="Iteration",
                value=iter_idx,
                min=0,
                max=len(iterations) - 1,
            )
            solara.Text(f"Iteration: {iterations[iter_idx.value]}")

        # Display figure - check if interactive
        fig_path = data[iter_idx.value][1]
        interactive = data[iter_idx.value][2]
        try:
            if interactive:
                InteractiveFigure(fig_path)
            else:
                StaticFigure(fig_path)
        except Exception as e:
            import traceback

            solara.Error(f"Failed: {e}\n{traceback.format_exc()}")


@solara.component
def FiguresPanel():
    """Panel for viewing figures."""
    refresh = refresh_counter.value
    runs = selected_runs.value

    if not runs:
        solara.Text("Select runs from the sidebar to view figures.")
        return

    # Memoize data loading
    def load_data():
        return {run: load_figures_info(run) for run in runs}

    runs_data = solara.use_memo(load_data, dependencies=[tuple(runs), refresh])

    # Get all tags from the pre-loaded data
    all_tags = set()
    for figures_info in runs_data.values():
        all_tags.update(figures_info.keys())

    if not all_tags:
        solara.Text("No figures logged in selected runs.")
        return

    root_path = Path(root_dir.value)

    for tag in sorted(all_tags):
        with solara.Details(tag, expand=True):
            for run in runs:
                if tag in runs_data[run] and runs_data[run][tag]:
                    FigureItem(run, tag, root_path, runs_data[run][tag])


@solara.component
def InteractiveFigure(fig_path: Path):
    """Display a matplotlib figure interactively using ipympl."""
    import matplotlib

    matplotlib.use("module://ipympl.backend_nbagg")
    from ipympl.backend_nbagg import Canvas, FigureManager

    # Load the pickled figure
    mpl_fig = load_figure(fig_path)
    if mpl_fig is None:
        solara.Text("Loading figure...")
        return

    # Create an interactive canvas for the figure
    canvas = Canvas(mpl_fig)
    manager = FigureManager(canvas, 0)

    # Display the canvas widget
    solara.display(canvas)


@solara.component
def StaticFigure(fig_path: Path):
    """Display a matplotlib figure as a static image."""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load the pickled figure
    mpl_fig = load_figure(fig_path)
    if mpl_fig is None:
        solara.Text("Loading figure...")
        return

    # Render to PNG
    buf = io.BytesIO()
    mpl_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(mpl_fig)

    # Display as HTML img tag
    solara.HTML(
        tag="img",
        attributes={
            "src": f"data:image/png;base64,{img_data}",
            "style": "max-width: 100%;",
        },
    )


@solara.component
def RefreshButton():
    """Manual refresh button."""

    def do_refresh():
        refresh_counter.set(refresh_counter.value + 1)

    solara.Button("Refresh", on_click=do_refresh, icon_name="mdi-refresh")


@solara.component
def Page():
    """Main viewer page."""
    # Hooks must be called before any early returns
    tab_index = solara.use_reactive(0)

    # Initialize from environment variable if set
    if not root_dir.value and os.environ.get("PYEXP_LOG_DIR"):
        root_dir.set(os.environ["PYEXP_LOG_DIR"])

    solara.Title("pyexp Log Viewer")

    with solara.AppBar():
        solara.Text("pyexp Log Viewer")

    with solara.Sidebar():
        solara.Markdown("## Runs")
        solara.InputText(
            label="Root Directory",
            value=root_dir,
        )
        RefreshButton()
        solara.Markdown("---")
        RunSelector()

    if not root_dir.value:
        solara.Markdown("## Welcome to pyexp Log Viewer")
        solara.Markdown("Enter a root directory path in the sidebar to discover runs.")
        return

    # Tabs for different views
    tab_names = ["Scalars", "Text", "Figures"]

    with solara.lab.Tabs(value=tab_index):
        for name in tab_names:
            solara.lab.Tab(name)

    # Content based on active tab
    if tab_index.value == 0:
        ScalarsPanel()
    elif tab_index.value == 1:
        TextPanel()
    elif tab_index.value == 2:
        FiguresPanel()
