"""Inline Python visualizations for notebooks (Colab/Jupyter).

These read the same SQLite store the engine writes, so you get the full
dashboard — equity curve, economics, positions, trades, activity — rendered
directly in notebook output cells, with no web server or port proxy.

Typical use in a notebook::

    import lmtrade.viz as viz
    viz.show()            # one-shot dashboard figure + tables
    viz.live()            # auto-refreshing dashboard (Ctrl-C / stop to end)
"""
from __future__ import annotations

from .panels import (
    activity_df,
    dashboard_figure,
    live,
    positions_df,
    show,
    summary,
    trades_df,
)

__all__ = [
    "show",
    "live",
    "summary",
    "dashboard_figure",
    "trades_df",
    "activity_df",
    "positions_df",
]
