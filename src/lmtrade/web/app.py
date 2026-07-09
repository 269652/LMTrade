"""FastAPI dashboard.

Read-only view over the Store: portfolio, economics/self-sustaining status,
trades, positions, activity feed, logs and the equity curve. The engine writes;
the web only reads, so it can run in the same or a separate process.

Run with:  lmtrade web
"""
from __future__ import annotations

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

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (TEMPLATES / "dashboard.html").read_text()

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        positions = store.positions()
        cash = float(store.get_meta("cash", settings.budget))
        econ = store.get_meta("economics", {})
        curve = store.equity_curve(limit=300)
        equity = curve[-1]["equity"] if curve else cash
        return SafeJSONResponse({
            "mode": store.get_meta("mode", settings.mode),
            "currency": settings.currency,
            "universe": store.get_meta("universe", settings.universe),
            "cash": round(cash, 4),
            "equity": round(equity, 4),
            "starting_cash": store.get_meta("starting_cash", settings.budget),
            "economics": econ,
            "positions": [
                {"symbol": p.symbol, "qty": round(p.qty, 6),
                 "avg_price": round(p.avg_price, 4)}
                for p in positions
            ],
            "num_positions": len(positions),
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

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


# For `uvicorn lmtrade.web.app:app`
app = create_app()
