"""Matplotlib/pandas dashboard panels rendered inline in notebooks.

Everything here reads from the Store (read path only) so it can run in a
different process/kernel from the engine. Designed to degrade gracefully when
the bot has produced no data yet.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from ..config import Settings, load_settings
from ..core.state import Store

# Dark theme to match the web dashboard.
_COLORS = {
    "bg": "#0b0f14", "panel": "#131a22", "text": "#e6edf3", "muted": "#8b98a5",
    "green": "#3fb950", "red": "#f85149", "blue": "#58a6ff", "amber": "#d29922",
    "line": "#243040",
}


def _store(db_path: str | Path | None = None) -> tuple[Store, Settings]:
    settings = load_settings()
    path = Path(db_path) if db_path else settings.db_path
    return Store(path), settings


def _fmt(n, d=2):
    return "—" if n is None else f"{n:.{d}f}"


# --------------------------------------------------------------------------- #
# Tables (return DataFrames so a notebook can just display them)
# --------------------------------------------------------------------------- #
def trades_df(limit: int = 50, db_path=None) -> pd.DataFrame:
    store, _ = _store(db_path)
    rows = store.recent_trades(limit)
    store.close()
    if not rows:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "fee", "confidence"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%H:%M:%S")
    return df[["time", "symbol", "side", "qty", "price", "fee", "confidence"]]


def activity_df(limit: int = 50, db_path=None) -> pd.DataFrame:
    store, _ = _store(db_path)
    rows = store.recent_activity(limit)
    store.close()
    if not rows:
        return pd.DataFrame(columns=["time", "kind", "symbol", "summary"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%H:%M:%S")
    return df[["time", "kind", "symbol", "summary"]]


def positions_df(db_path=None) -> pd.DataFrame:
    store, _ = _store(db_path)
    pos = store.positions()
    store.close()
    if not pos:
        return pd.DataFrame(columns=["symbol", "qty", "avg_price"])
    return pd.DataFrame(
        [{"symbol": p.symbol, "qty": round(p.qty, 6), "avg_price": round(p.avg_price, 4)}
         for p in pos]
    )


def summary(db_path=None) -> dict:
    """Return the headline numbers as a plain dict."""
    store, settings = _store(db_path)
    econ = store.get_meta("economics", {}) or {}
    cash = float(store.get_meta("cash", settings.budget))
    mode = store.get_meta("mode", settings.mode)
    curve = store.equity_curve(500)
    store.close()
    equity = curve[-1]["equity"] if curve else cash
    return {
        "mode": mode,
        "currency": settings.currency,
        "cash": cash,
        "equity": equity,
        "net_worth": econ.get("net_worth_eur", equity),
        "pnl": econ.get("pnl_eur", equity - settings.budget),
        "runway_hours": econ.get("runway_hours"),
        "self_sustaining": econ.get("self_sustaining", False),
        "compute_usd": (econ.get("gpu_cost_accrued_usd", 0) or 0)
                       + (econ.get("inference_cost_usd", 0) or 0),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def dashboard_figure(db_path=None, figsize=(13, 8)):
    """Build the multi-panel dashboard figure and return it (does not show)."""
    store, settings = _store(db_path)
    cur = settings.currency
    econ = store.get_meta("economics", {}) or {}
    cash = float(store.get_meta("cash", settings.budget))
    starting = float(store.get_meta("starting_cash", settings.budget))
    curve = store.equity_curve(500)
    bench = store.benchmark_curve(500)
    alpha = store.get_meta("alpha")
    open_opts = store.open_options()
    costs = store.total_costs()
    positions = store.positions()
    mode = store.get_meta("mode", settings.mode)
    store.close()

    plt.rcParams.update({
        "figure.facecolor": _COLORS["bg"], "axes.facecolor": _COLORS["panel"],
        "text.color": _COLORS["text"], "axes.labelcolor": _COLORS["muted"],
        "xtick.color": _COLORS["muted"], "ytick.color": _COLORS["muted"],
        "axes.edgecolor": _COLORS["line"], "font.size": 10,
    })

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35,
                  height_ratios=[1.4, 1.0, 1.0])

    # -- Equity curve (top, full width) --------------------------------------
    ax_eq = fig.add_subplot(gs[0, :])
    if len(curve) >= 2:
        xs = [pd.to_datetime(p["ts"], unit="s") for p in curve]
        ys = [p["equity"] for p in curve]
        up = ys[-1] >= ys[0]
        color = _COLORS["green"] if up else _COLORS["red"]
        ax_eq.plot(xs, ys, color=color, lw=2, label="bot")
        ax_eq.fill_between(xs, ys, min(ys), color=color, alpha=0.12)
        ax_eq.axhline(starting, color=_COLORS["muted"], ls="--", lw=1, alpha=0.6)
    else:
        ax_eq.text(0.5, 0.5, "Waiting for equity data…", ha="center", va="center",
                   color=_COLORS["muted"], transform=ax_eq.transAxes)
    if bench:
        bx = [pd.to_datetime(p["ts"], unit="s") for p in bench]
        by = [p["equity"] for p in bench]
        ax_eq.plot(bx, by, color=_COLORS["blue"], lw=1.5, ls=":",
                   label="benchmark (buy&hold)")
    if (len(curve) >= 2) or bench:
        ax_eq.legend(loc="upper left", frameon=False, fontsize=8,
                     labelcolor=_COLORS["muted"])
    alpha_txt = f"  ·  alpha {alpha:+.3f} {cur}" if alpha is not None else ""
    ax_eq.set_title(f"Equity Curve ({cur}) vs Retail Benchmark{alpha_txt}",
                    color=_COLORS["text"], loc="left", fontweight="bold")
    ax_eq.grid(True, color=_COLORS["line"], alpha=0.3)

    # -- KPI / economics panel (mid-left, text) ------------------------------
    ax_kpi = fig.add_subplot(gs[1, 0]); ax_kpi.axis("off")
    net = econ.get("net_worth_eur", cash)
    pnl = econ.get("pnl_eur", net - starting)
    runway = econ.get("runway_hours")
    ss = econ.get("self_sustaining", False)
    pnl_color = _COLORS["green"] if pnl >= 0 else _COLORS["red"]
    lines = [
        ("Net worth", f"{_fmt(net)} {cur}", _COLORS["text"]),
        ("P&L", f"{'+' if pnl >= 0 else ''}{_fmt(pnl)} {cur}", pnl_color),
        ("Cash", f"{_fmt(cash)} {cur}", _COLORS["text"]),
        ("Runway", f"{_fmt(runway, 1)} h" if runway is not None else "—", _COLORS["blue"]),
        ("Self-sustaining", "YES" if ss else "not yet",
         _COLORS["green"] if ss else _COLORS["amber"]),
    ]
    ax_kpi.set_title("Portfolio", color=_COLORS["text"], loc="left", fontweight="bold")
    for i, (k, v, c) in enumerate(lines):
        y = 0.86 - i * 0.2
        ax_kpi.text(0.02, y, k, color=_COLORS["muted"], fontsize=9, transform=ax_kpi.transAxes)
        ax_kpi.text(0.98, y, v, color=c, fontsize=12, fontweight="bold",
                    ha="right", transform=ax_kpi.transAxes)

    # -- Compute cost breakdown (mid-mid) ------------------------------------
    ax_cost = fig.add_subplot(gs[1, 1])
    gpu = econ.get("gpu_cost_accrued_usd", 0) or 0
    inf = econ.get("inference_cost_usd", 0) or 0
    fees_usd = (costs.get("fee", 0) or 0) * 1.08
    parts = [("GPU", gpu), ("Inference", inf), ("Fees", fees_usd)]
    parts = [(n, v) for n, v in parts if v > 0]
    if parts:
        ax_cost.pie([v for _, v in parts], labels=[n for n, _ in parts],
                    colors=[_COLORS["blue"], _COLORS["amber"], _COLORS["red"]],
                    autopct=lambda p: f"${p*sum(v for _, v in parts)/100:.3f}",
                    textprops={"color": _COLORS["text"], "fontsize": 8})
    else:
        ax_cost.text(0.5, 0.5, "no cost yet", ha="center", va="center",
                     color=_COLORS["muted"], transform=ax_cost.transAxes)
        ax_cost.axis("off")
    ax_cost.set_title("Compute Spend (USD)", color=_COLORS["text"], loc="left", fontweight="bold")

    # -- Positions incl. options (mid-right) -----------------------------------
    ax_pos = fig.add_subplot(gs[1, 2])
    labels: list[str] = [p.symbol for p in positions]
    notion: list[float] = [p.qty * p.avg_price for p in positions]
    for o in open_opts:
        labels.append(f"{o['underlying']} {o['kind'][0].upper()}{o['strike']:g}")
        notion.append(o["contracts"] * o["entry_premium"])
    if labels:
        ax_pos.barh(labels, notion, color=_COLORS["blue"])
        ax_pos.set_xlabel(cur)
    else:
        ax_pos.text(0.5, 0.5, "no open positions", ha="center", va="center",
                    color=_COLORS["muted"], transform=ax_pos.transAxes)
        ax_pos.axis("off")
    ax_pos.set_title("Positions & Options", color=_COLORS["text"], loc="left",
                     fontweight="bold")

    # -- Recent trades table (bottom, full width) ----------------------------
    ax_tr = fig.add_subplot(gs[2, :]); ax_tr.axis("off")
    ax_tr.set_title("Recent Trades", color=_COLORS["text"], loc="left", fontweight="bold")
    tdf = trades_df(8, db_path)
    if len(tdf):
        tbl = ax_tr.table(cellText=tdf.round(4).values, colLabels=tdf.columns,
                          loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
        for (r, _), cell in tbl.get_celld().items():
            cell.set_edgecolor(_COLORS["line"])
            cell.set_facecolor(_COLORS["panel"] if r else _COLORS["line"])
            cell.set_text_props(color=_COLORS["text"])
    else:
        ax_tr.text(0.5, 0.4, "no trades yet", ha="center", va="center",
                   color=_COLORS["muted"], transform=ax_tr.transAxes)

    fig.suptitle(f"LMTrade  |  {mode.upper()}  |  {time.strftime('%H:%M:%S')}",
                 color=_COLORS["text"], fontsize=14, fontweight="bold", x=0.02, ha="left")
    return fig


def show(db_path=None, figsize=(13, 8)):
    """Render the dashboard once (inline in a notebook)."""
    fig = dashboard_figure(db_path, figsize)
    plt.show()
    return fig


def live(interval: float = 5.0, iterations: int | None = None, db_path=None, figsize=(13, 8)):
    """Auto-refreshing dashboard for notebooks: redraws every `interval` seconds.

    Runs `iterations` times (None = until interrupted). Uses IPython display
    clearing when available, so a single cell shows a live-updating dashboard.
    """
    try:
        from IPython.display import clear_output, display
        in_notebook = True
    except Exception:  # noqa: BLE001
        in_notebook = False

    n = 0
    try:
        while iterations is None or n < iterations:
            fig = dashboard_figure(db_path, figsize)
            if in_notebook:
                clear_output(wait=True)
                display(fig)
            else:
                plt.show()
            plt.close(fig)
            n += 1
            if iterations is not None and n >= iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("live dashboard stopped")
