"""Solara-based viewer for pyexp logs.

Install with: pip install pyexp[viewer]
Run with: python -m pyexp.viewer <log_dir>
"""

def __getattr__(name):
    """Lazy import to avoid loading solara until needed."""
    from pyexp import _viewer
    return getattr(_viewer, name)
