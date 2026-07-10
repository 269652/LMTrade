"""FastAPI dashboard.

Read-only view over the Store: portfolio, economics/self-sustaining status,
trades, positions, activity feed, logs and the equity curve. The engine writes;
the web only reads, so it can run in the same or a separate process.

Run with:  lmtrade web
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import DEFAULT_TOML_PATH, Settings, load_settings
from ..core.control import LOW_BALANCE_EUR, ControlState
from ..core.state import Store
from . import settings_editor

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"


def _json_safe(obj: Any) -> Any:
    """Replace any inf/-inf/nan float anywhere in a JSON-able structure with
    None. Store round-trips those fine (plain json.dumps/loads allow them —
    e.g. a historical activity/meta row written before a producer started
    sanitizing its own floats, such as an infinite runway_hours when
    gpu_usd_per_hour=0), but strict JSON (RFC 8259, no Infinity/NaN) does
    not, and that's what every response here must produce."""
    if isinstance(obj, float):
        return None if math.isinf(obj) or math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(_json_safe(content))


def _schedule_restart() -> None:
    """Replace the running process with a fresh copy 0.5 s after being called.

    The short delay lets FastAPI finish sending the HTTP response before the
    process image is replaced. Works for both ``lmtrade run`` (engine + web in
    one process) and ``lmtrade web`` (web-only); in both cases ``os.execv``
    re-executes the same command with the same arguments, picking up the newly
    saved ``config.toml``.
    """
    import os
    import sys
    import threading

    def _exec() -> None:
        import time
        time.sleep(0.5)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_exec, daemon=True).start()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    # Two books, one connection each. Which one every read serves is chosen
    # per request by the control state's mode, so flipping the dashboard
    # paper/live toggle swaps the whole view to the matching ledger/history.
    _books = {
        "paper": Store(settings.db_path),
        "live": Store(settings.live_db_path),
    }
    app = FastAPI(title="LMTrade", version="0.1.0")

    def control() -> ControlState:
        return ControlState.load(settings.control_path)

    def store_for(mode: str) -> Store:
        return _books["live"] if mode == "live" else _books["paper"]

    def store() -> Store:
        # The live tab always shows the REAL account (live book), regardless
        # of the arm guard — arming only gates whether NEW orders are placed
        # for real, it must not change what the dashboard displays. The live
        # book stays current even while unarmed (the engine trades paper
        # meanwhile) because sync_tr_portfolio imports real TR positions into
        # it and the engine cross-values the inactive book every cycle
        # (Engine._value_fallback_book) using shared price data.
        return store_for(control().mode)

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    def _asset_version() -> str:
        """Short content hash of app.js, appended as ?v= to its URL so a
        changed file bypasses the browser cache (otherwise a pulled app.js is
        rendered against a freshly-served HTML shell — column mismatch)."""
        js = STATIC / "app.js"
        if not js.exists():
            return "0"
        return hashlib.sha1(js.read_bytes()).hexdigest()[:8]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Explicit utf-8: Path.read_text() uses the platform default encoding,
        # which on Windows is cp1252 and mangles the template's ⚡/· glyphs
        # into mojibake (âš¡ / Â·) before they're ever served.
        html = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")
        return html.replace("/static/app.js", f"/static/app.js?v={_asset_version()}")

    def _tr_meta(key: str):
        """TR account meta (cash/baseline). The engine mirrors this into BOTH
        books every cycle (Engine._value_fallback_book), so the live book
        normally has it directly; the paper-book fallback is defensive only
        (e.g. a brand new live book before the engine's first cycle)."""
        v = store_for("live").get_meta(key)
        return v if v is not None else store_for("paper").get_meta(key)

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        is_live = control().mode == "live"
        equity_positions = store().positions()
        open_opts = store().open_options()
        tr_cash = _tr_meta("tr_account_cash")
        # LIVE view: cash is the REAL TR balance (None until fetched) — never
        # default an empty live book to the paper budget, which fabricated a
        # "100.00 EUR" live net worth. Paper view: the simulated book's cash.
        if is_live:
            cash = tr_cash
        else:
            cash = float(store().get_meta("cash", settings.budget))
        econ = store().get_meta("economics", {})
        
        # Calculate equity: for live mode, use real cash + position values;
        # for paper mode, use the recorded equity curve (which includes cash + all holdings).
        if is_live and cash is not None:
            # Real live: equity = TR cash + unrealized P&L on open positions
            positions_value = sum(
                (m.get("value") or 0.0) for m in store().get_meta("open_option_marks", {}).values()
            )
            equity = cash + positions_value
        else:
            curve = store().equity_curve(limit=300)
            equity = curve[-1]["equity"] if curve else (cash or 0.0)
        # Show the FULL book the engine counts toward max_positions: equity
        # positions AND open options/knockouts. Options were previously
        # omitted, so an options-only book (the common case) showed "0
        # positions" even when full. For an option, qty=contracts and
        # avg_price=entry premium; the real TR ISIN is surfaced when present.
        marks = store().get_meta("open_option_marks", {}) or {}
        rows = [
            {"symbol": p.symbol, "qty": round(p.qty, 6),
             "avg_price": round(p.avg_price, 4), "kind": "equity", "isin": None,
             "value": None, "unrealized_pnl": None}
            for p in equity_positions
        ]
        for o in open_opts:
            kind = o.get("instrument_type") or "option"
            label = f"{o['underlying']} {o.get('kind', '')}".strip()
            # Live mark persisted by the engine each cycle; None until the
            # first cycle marks this option (don't fabricate a P&L).
            mark = marks.get(str(o.get("id")))
            rows.append({
                "symbol": label,
                "qty": round(o.get("contracts", 0.0), 6),
                "avg_price": round(o.get("entry_premium", 0.0), 4),
                "kind": kind,
                "isin": o.get("isin"),
                "value": mark.get("value") if mark else None,
                "unrealized_pnl": mark.get("unrealized_pnl") if mark else None,
            })
        return SafeJSONResponse({
            "mode": store().get_meta("mode", settings.mode),
            "currency": settings.currency,
            "universe": store().get_meta("universe", settings.universe),
            "cash": round(cash, 4) if cash is not None else None,
            "equity": round(equity, 4),
            "reserve": round(store().reserve_balance(), 4),
            "tr_account_cash": tr_cash,
            "tr_baseline_net_worth": _tr_meta("tr_baseline_net_worth"),
            "last_realized_net_worth": store().get_meta("last_realized_net_worth"),
            "starting_cash": store().get_meta("starting_cash", settings.budget),
            "economics": econ,
            "positions": rows,
            "num_positions": len(rows),
            "provider_warnings": store().get_meta("provider_warnings"),
            "alpha": store().get_meta("alpha"),
        })

    @app.get("/api/trades")
    def trades(limit: int = 100) -> JSONResponse:
        return SafeJSONResponse(store().recent_trades(limit))

    @app.get("/api/activity")
    def activity(limit: int = 100) -> JSONResponse:
        return SafeJSONResponse(store().recent_activity(limit))

    @app.get("/api/logs")
    def logs(limit: int = 200) -> JSONResponse:
        return SafeJSONResponse(store().recent_logs(limit))

    @app.get("/api/equity")
    def equity() -> JSONResponse:
        return SafeJSONResponse(store().equity_curve(limit=500))

    @app.get("/api/costs")
    def costs() -> JSONResponse:
        return SafeJSONResponse(store().total_costs())

    @app.get("/api/realized")
    def realized(limit: int = 100) -> JSONResponse:
        """Closed options/knockouts with realized P&L, plus running totals —
        the counterpart to the open positions' unrealized P&L."""
        closed = store().closed_options(limit)
        rows = []
        total = wins = losses = 0.0
        for o in closed:
            pnl = float(o.get("pnl") or 0.0)
            total += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            rows.append({
                "symbol": f"{o['underlying']} {o.get('kind', '')}".strip(),
                "kind": o.get("instrument_type") or "option",
                "isin": o.get("isin"),
                "contracts": round(o.get("contracts", 0.0), 6),
                "entry_premium": round(o.get("entry_premium", 0.0), 4),
                "exit_premium": round((o.get("exit_premium") or 0.0), 4),
                "pnl": round(pnl, 4),
                "closed_ts": o.get("closed_ts"),
            })
        return SafeJSONResponse({
            "total_pnl": round(total, 4),
            "wins": int(wins),
            "losses": int(losses),
            "rows": rows,
        })

    @app.get("/api/options")
    def options() -> JSONResponse:
        return SafeJSONResponse({
            "open": store().open_options(),
            "closed": store().closed_options(50),
        })

    @app.get("/api/leaderboard")
    def leaderboard() -> JSONResponse:
        genomes = store().get_meta("genomes", []) or []
        rows = [
            {"id": g["id"], "strategy": g["strategy"], "params": g["params"],
             "trades": g["trades"], "pnl": round(g["pnl"], 4),
             # Same shrunk fitness as the optimizer (see FITNESS_SHRINK_K).
             "fitness": round(g["pnl"] / (g["trades"] + 2.0), 5) if g["trades"] else 0.01}
            for g in genomes
        ]
        rows.sort(key=lambda r: r["fitness"], reverse=True)
        return SafeJSONResponse(rows)

    @app.get("/api/benchmark")
    def benchmark() -> JSONResponse:
        return SafeJSONResponse({
            "curve": store().benchmark_curve(500),
            "alpha": store().get_meta("alpha"),
            "entry": store().get_meta("benchmark_entry"),
        })

    @app.get("/api/news")
    def news() -> JSONResponse:
        return SafeJSONResponse(store().recent_news(50))

    @app.get("/api/analysis")
    def analysis() -> JSONResponse:
        """The latest compiled daily market analysis (per-symbol bias)."""
        return SafeJSONResponse(store().get_meta("market_analysis", {}) or {})

    @app.get("/api/signals")
    def signals(min_confidence: float = 0.6, limit: int = 200) -> JSONResponse:
        """Strongest recent decisions and their cause — the per-provider signal
        breakdown recorded in each 'decision' activity. Deduped to the most
        recent decision per symbol so the tab reads as 'current strong signals'."""
        seen: set[str] = set()
        out = []
        for a in store().recent_activity(limit):
            if a.get("kind") != "decision":
                continue
            d = a.get("detail") or {}
            symbol = a.get("symbol")
            if symbol in seen:
                continue
            seen.add(symbol)
            if d.get("direction") in (None, "hold"):
                continue
            if float(d.get("confidence", 0.0)) < min_confidence:
                continue
            out.append({
                "ts": a.get("ts"),
                "symbol": symbol,
                "direction": d.get("direction"),
                "confidence": d.get("confidence"),
                "rationale": d.get("rationale", ""),
                "signals": d.get("signals", []),
            })
        out.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return SafeJSONResponse(out)

    # ------------------------------------------------------------- control plane
    def _live_net_worth() -> float | None:
        econ = store_for("live").get_meta("economics", {}) or {}
        # Prefer the REAL TR account balance (what the guard actually gates
        # on), from whichever book the engine wrote it into; fall back to the
        # live book's computed net worth.
        real = (store_for("live").get_meta("tr_account_cash")
                or store_for("paper").get_meta("tr_account_cash"))
        if real is not None:
            return float(real)
        nw = econ.get("net_worth_eur")
        return float(nw) if nw is not None else None

    def _control_payload(c, nw) -> dict:
        return {
            "mode": c.mode,
            "armed": c.armed,
            "double_armed": c.double_armed,
            "live_armed": c.live_armed,
            "net_worth": nw,
            "low_balance": c.is_low_balance(nw),
            "low_balance_threshold": LOW_BALANCE_EUR,
            # Single source of truth for the IS/IS-NOT-executing banner.
            "executing": c.is_executing(nw),
        }

    @app.get("/api/control")
    def get_control() -> JSONResponse:
        return SafeJSONResponse(_control_payload(control(), _live_net_worth()))

    @app.post("/api/control/mode")
    def set_mode(payload: dict) -> JSONResponse:
        mode = (payload or {}).get("mode")
        c = control()
        try:
            c.set_mode(mode)
        except ValueError:
            return SafeJSONResponse({"error": f"invalid mode {mode!r}"}, status_code=400)
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/arm")
    def arm(payload: dict) -> JSONResponse:
        c = control()
        c.arm(confirm=bool((payload or {}).get("confirm")))
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/disarm")
    def disarm() -> JSONResponse:
        c = control()
        c.disarm()
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/double-arm")
    def double_arm(payload: dict) -> JSONResponse:
        c = control()
        c.set_double_armed(confirm=bool((payload or {}).get("confirm")))
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/double-disarm")
    def double_disarm() -> JSONResponse:
        c = control()
        c.double_disarm()
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    # ---------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings() -> JSONResponse:
        return SafeJSONResponse({
            "fields": settings_editor.schema(settings),
            "config_path": str(DEFAULT_TOML_PATH),
            "note": "Saved to config.toml. Restart lmtrade run to apply.",
        })

    @app.post("/api/settings")
    def post_settings(payload: dict) -> JSONResponse:
        changes = (payload or {}).get("changes", {})
        result = settings_editor.apply_changes(DEFAULT_TOML_PATH, changes)
        restarting = bool(result.get("applied"))
        if restarting:
            _schedule_restart()
        return SafeJSONResponse({**result, "restarting": restarting})

    @app.post("/api/dismiss-provider-warning")
    def dismiss_provider_warning() -> JSONResponse:
        """Clear the persistent provider warnings from the store so they don't
        reappear until the next limit event."""
        store().set_meta("provider_warnings", None)
        return SafeJSONResponse({"ok": True})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


# For `uvicorn lmtrade.web.app:app`
app = create_app()
