"""Tests for the dashboard API: existing endpoints plus the new options,
strategy-leaderboard and benchmark/alpha endpoints. Written before the
endpoint implementation per strict TDD."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.web.app import create_app


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    return s


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    # Produce some real state first.
    store = Store(settings.db_path)
    engine = Engine(settings, store, PaperBroker(store, starting_cash=10.0, fee=0.1))
    engine.run_cycle()
    store.close()
    return TestClient(create_app(settings))


class TestExistingEndpoints:
    def test_summary_and_health(self, client):
        assert client.get("/healthz").json() == {"ok": True}
        s = client.get("/api/summary").json()
        assert s["mode"] == "paper"
        assert "economics" in s

    def test_trades_activity_logs_equity(self, client):
        for path in ("/api/trades", "/api/activity", "/api/logs", "/api/equity"):
            r = client.get(path)
            assert r.status_code == 200
            assert isinstance(r.json(), list)


class TestSummaryIncludesOptions:
    """The engine counts open OPTIONS toward max_positions, but the dashboard
    summary only counted equity positions — so a book full of options showed
    "0 positions", which read as broken. Summary must reflect the full book."""

    @pytest.fixture()
    def options_client(self, tmp_path: Path) -> TestClient:
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.open_option("AAPL", "call", strike=100.0, expiry_ts=4e12, iv=0.2,
                          contracts=3.0, entry_premium=2.5, genome_id=None,
                          tp_premium=3.75, sl_premium=1.5)
        store.open_option("NVDA", "put", strike=200.0, expiry_ts=4e12, iv=0.2,
                          contracts=1.0, entry_premium=4.0, genome_id=None,
                          tp_premium=6.0, sl_premium=2.4, instrument_type="knockout",
                          barrier=200.0, ratio=10.0, isin="DE000KO1")
        store.close()
        return TestClient(create_app(s))

    def test_num_positions_counts_open_options(self, options_client):
        s = options_client.get("/api/summary").json()
        assert s["num_positions"] == 2

    def test_positions_list_includes_options_with_isin(self, options_client):
        s = options_client.get("/api/summary").json()
        symbols = {p["symbol"] for p in s["positions"]}
        assert any("AAPL" in sym for sym in symbols)
        nvda = next(p for p in s["positions"] if "NVDA" in p["symbol"])
        assert nvda["isin"] == "DE000KO1"          # real TR knockout surfaced
        assert nvda["kind"] in ("put", "knockout") or "put" in nvda["symbol"].lower()


class TestInfiniteRunwayJsonSafety:
    """gpu_usd_per_hour=0 (no GPU rented) makes runway_hours float('inf') —
    Starlette's JSONResponse (allow_nan=False) crashes on that unless it's
    converted before serialization. Regression test for a live crash."""

    @pytest.fixture()
    def zero_gpu_client(self, tmp_path: Path) -> TestClient:
        s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                     data={"provider": "synthetic"},
                     economics={"gpu_usd_per_hour": 0.0})
        s.model.stack = ["heuristic"]
        s.data_dir = tmp_path
        store = Store(s.db_path)
        engine = Engine(s, store, PaperBroker(store, starting_cash=10.0, fee=0.1))
        engine.run_cycle()
        store.close()
        return TestClient(create_app(s))

    def test_summary_is_json_safe(self, zero_gpu_client):
        r = zero_gpu_client.get("/api/summary")
        assert r.status_code == 200
        assert r.json()["economics"]["runway_hours"] is None

    def test_activity_is_json_safe(self, zero_gpu_client):
        r = zero_gpu_client.get("/api/activity")
        assert r.status_code == 200


class TestLegacyInfiniteDataJsonSafety:
    """Activity rows / meta written before the runway_hours-as-None fix (or
    by any future producer that bypasses EconomicsSnapshot.as_dict()) can
    still carry a raw float('inf') in their stored JSON — Store round-trips
    it fine (plain json.dumps/loads allow it), only Starlette's strict
    JSONResponse rejects it. The API must sanitize at the boundary, not
    just at the one producer already fixed, so pre-existing data doesn't
    keep crashing every request."""

    def test_activity_with_raw_infinite_detail_is_json_safe(self, settings, tmp_path):
        store = Store(settings.db_path)
        store.add_activity("economics", "legacy poisoned record",
                            detail={"runway_hours": float("inf")})
        store.close()
        client = TestClient(create_app(settings))
        r = client.get("/api/activity")
        assert r.status_code == 200
        assert r.json()[0]["detail"]["runway_hours"] is None

    def test_summary_with_raw_infinite_economics_meta_is_json_safe(self, settings, tmp_path):
        store = Store(settings.db_path)
        store.set_meta("economics", {"runway_hours": float("inf")})
        store.close()
        client = TestClient(create_app(settings))
        r = client.get("/api/summary")
        assert r.status_code == 200
        assert r.json()["economics"]["runway_hours"] is None


class TestNewEndpoints:
    def test_options_endpoint(self, client):
        r = client.get("/api/options")
        assert r.status_code == 200
        body = r.json()
        assert "open" in body and "closed" in body
        assert isinstance(body["open"], list)

    def test_leaderboard_endpoint(self, client):
        r = client.get("/api/leaderboard")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list)
        if rows:  # learning enabled by default -> populated
            assert {"id", "strategy", "fitness", "trades"} <= set(rows[0])

    def test_benchmark_endpoint(self, client):
        r = client.get("/api/benchmark")
        assert r.status_code == 200
        body = r.json()
        assert "curve" in body and "alpha" in body
        assert isinstance(body["curve"], list)

    def test_news_endpoint(self, client):
        r = client.get("/api/news")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
