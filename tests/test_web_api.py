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
