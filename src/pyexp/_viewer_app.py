"""Solara app components for pyexp viewer.

This module is loaded by solara run, so it can import solara at the top level.
"""

import os
from pathlib import Path

import solara

from pyexp._viewer import (
    discover_runs,
    get_all_figure_tags,
    get_all_scalar_tags,
    get_all_text_tags,
    load_figure,
    load_figures_info,
    load_iterations,
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
def ScalarPlot(tag: str, runs: list, root_path: Path):
    """Display a scalar plot for a single tag across multiple runs."""
    import plotly.graph_objects as go

    # Per-tag log scale toggle
    log_scale = solara.use_reactive(False)

    fig = go.Figure()

    for run in runs:
        timeseries = load_scalars_timeseries(run)
        if tag in timeseries:
            data = timeseries[tag]
            iterations = [d[0] for d in data]
            values = [d[1] for d in data]

            run_name = str(run.relative_to(root_path)) if run != root_path else "."
            fig.add_trace(go.Scatter(
                x=iterations,
                y=values,
                mode='lines+markers',
                marker=dict(size=4),
                name=run_name,
                hovertemplate=f'{run_name}<br>Iteration: %{{x}}<br>Value: %{{y:.6g}}<extra></extra>',
            ))

    fig.update_layout(
        title=tag,
        xaxis_title='Iteration',
        yaxis_title=tag,
        yaxis_type='log' if log_scale.value else 'linear',
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    solara.Checkbox(label="Log scale (Y-axis)", value=log_scale)
    solara.FigurePlotly(fig)


@solara.component
def ScalarsPanel():
    """Panel for viewing scalar time series."""
    _ = refresh_counter.value

    if not selected_runs.value:
        solara.Text("Select runs from the sidebar to view scalars.")
        return

    # Get all scalar tags across selected runs
    all_tags = get_all_scalar_tags(selected_runs.value)

    if not all_tags:
        solara.Text("No scalars logged in selected runs.")
        return

    root_path = Path(root_dir.value)

    for tag in sorted(all_tags):
        with solara.Details(tag, expand=True):
            ScalarPlot(tag, selected_runs.value, root_path)


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
    _ = refresh_counter.value

    if not selected_runs.value:
        solara.Text("Select runs from the sidebar to view text.")
        return

    # Get all text tags across selected runs
    all_tags = get_all_text_tags(selected_runs.value)

    if not all_tags:
        solara.Text("No text logged in selected runs.")
        return

    root_path = Path(root_dir.value)

    for tag in sorted(all_tags):
        with solara.Details(tag, expand=True):
            for run in selected_runs.value:
                timeseries = load_text_timeseries(run)
                if tag in timeseries and timeseries[tag]:
                    TextItem(run, tag, root_path, timeseries[tag])


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

        # Display figure
        fig_path = data[iter_idx.value][1]
        try:
            InteractiveFigure(fig_path)
        except Exception as e:
            import traceback
            solara.Error(f"Failed: {e}\n{traceback.format_exc()}")


@solara.component
def FiguresPanel():
    """Panel for viewing figures."""
    _ = refresh_counter.value

    if not selected_runs.value:
        solara.Text("Select runs from the sidebar to view figures.")
        return

    # Get all figure tags across selected runs
    all_tags = get_all_figure_tags(selected_runs.value)

    if not all_tags:
        solara.Text("No figures logged in selected runs.")
        return

    root_path = Path(root_dir.value)

    for tag in sorted(all_tags):
        with solara.Details(tag, expand=True):
            for run in selected_runs.value:
                figures_info = load_figures_info(run)
                if tag in figures_info and figures_info[tag]:
                    FigureItem(run, tag, root_path, figures_info[tag])


@solara.component
def InteractiveFigure(fig_path: Path):
    """Display a matplotlib figure interactively using ipympl."""
    import matplotlib
    matplotlib.use('module://ipympl.backend_nbagg')
    from ipympl.backend_nbagg import Canvas, FigureManager

    # Load the pickled figure
    mpl_fig = load_figure(fig_path)

    # Create an interactive canvas for the figure
    canvas = Canvas(mpl_fig)
    manager = FigureManager(canvas, 0)

    # Display the canvas widget
    solara.display(canvas)


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
