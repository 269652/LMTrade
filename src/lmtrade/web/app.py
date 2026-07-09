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

from ..config import Settings, load_settings
from ..core.state import Store

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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = Store(settings.db_path)
    app = FastAPI(title="LMTrade", version="0.1.0")

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

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        equity_positions = store.positions()
        open_opts = store.open_options()
        cash = float(store.get_meta("cash", settings.budget))
        econ = store.get_meta("economics", {})
        curve = store.equity_curve(limit=300)
        equity = curve[-1]["equity"] if curve else cash
        # Show the FULL book the engine counts toward max_positions: equity
        # positions AND open options/knockouts. Options were previously
        # omitted, so an options-only book (the common case) showed "0
        # positions" even when full. For an option, qty=contracts and
        # avg_price=entry premium; the real TR ISIN is surfaced when present.
        marks = store.get_meta("open_option_marks", {}) or {}
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
            "mode": store.get_meta("mode", settings.mode),
            "currency": settings.currency,
            "universe": store.get_meta("universe", settings.universe),
            "cash": round(cash, 4),
            "equity": round(equity, 4),
            "reserve": round(store.reserve_balance(), 4),
            "tr_account_cash": store.get_meta("tr_account_cash"),
            "last_realized_net_worth": store.get_meta("last_realized_net_worth"),
            "starting_cash": store.get_meta("starting_cash", settings.budget),
            "economics": econ,
            "positions": rows,
            "num_positions": len(rows),
        })

    @app.get("/api/trades")
    def trades(limit: int = 100) -> JSONResponse:
        return SafeJSONResponse(store.recent_trades(limit))

    @app.get("/api/activity")
    def activity(limit: int = 100) -> JSONResponse:
        return SafeJSONResponse(store.recent_activity(limit))

    @app.get("/api/logs")
    def logs(limit: int = 200) -> JSONResponse:
        return SafeJSONResponse(store.recent_logs(limit))

    @app.get("/api/equity")
    def equity() -> JSONResponse:
        return SafeJSONResponse(store.equity_curve(limit=500))

    @app.get("/api/costs")
    def costs() -> JSONResponse:
        return SafeJSONResponse(store.total_costs())

    @app.get("/api/realized")
    def realized(limit: int = 100) -> JSONResponse:
        """Closed options/knockouts with realized P&L, plus running totals —
        the counterpart to the open positions' unrealized P&L."""
        closed = store.closed_options(limit)
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
            "open": store.open_options(),
            "closed": store.closed_options(50),
        })

    @app.get("/api/leaderboard")
    def leaderboard() -> JSONResponse:
        genomes = store.get_meta("genomes", []) or []
        rows = [
            {"id": g["id"], "strategy": g["strategy"], "params": g["params"],
             "trades": g["trades"], "pnl": round(g["pnl"], 4),
             "fitness": round(g["pnl"] / g["trades"], 5) if g["trades"] else 0.01}
            for g in genomes
        ]
        rows.sort(key=lambda r: r["fitness"], reverse=True)
        return SafeJSONResponse(rows)

    @app.get("/api/benchmark")
    def benchmark() -> JSONResponse:
        return SafeJSONResponse({
            "curve": store.benchmark_curve(500),
            "alpha": store.get_meta("alpha"),
            "entry": store.get_meta("benchmark_entry"),
        })

    @app.get("/api/news")
    def news() -> JSONResponse:
        return SafeJSONResponse(store.recent_news(50))

    @app.get("/api/analysis")
    def analysis() -> JSONResponse:
        """The latest compiled daily market analysis (per-symbol bias)."""
        return SafeJSONResponse(store.get_meta("market_analysis", {}) or {})

    @app.get("/api/signals")
    def signals(min_confidence: float = 0.6, limit: int = 200) -> JSONResponse:
        """Strongest recent decisions and their cause — the per-provider signal
        breakdown recorded in each 'decision' activity. Deduped to the most
        recent decision per symbol so the tab reads as 'current strong signals'."""
        seen: set[str] = set()
        out = []
        for a in store.recent_activity(limit):
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

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


# For `uvicorn lmtrade.web.app:app`
app = create_app()
