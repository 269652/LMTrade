"""FastAPI dashboard.

Read-only view over the Store: portfolio, economics/self-sustaining status,
trades, positions, activity feed, logs and the equity curve. The engine writes;
the web only reads, so it can run in the same or a separate process.

Run with:  lmtrade web
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, load_settings
from ..core.state import Store

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"


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
        return JSONResponse({
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
        return JSONResponse(store.recent_trades(limit))

    @app.get("/api/activity")
    def activity(limit: int = 100) -> JSONResponse:
        return JSONResponse(store.recent_activity(limit))

    @app.get("/api/logs")
    def logs(limit: int = 200) -> JSONResponse:
        return JSONResponse(store.recent_logs(limit))

    @app.get("/api/equity")
    def equity() -> JSONResponse:
        return JSONResponse(store.equity_curve(limit=500))

    @app.get("/api/costs")
    def costs() -> JSONResponse:
        return JSONResponse(store.total_costs())

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


# For `uvicorn lmtrade.web.app:app`
app = create_app()
