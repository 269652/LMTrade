"""Tests for the option-position / news / benchmark tables in core/state.py."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.core.state import Store


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


class TestOptionPositions:
    def test_open_close_lifecycle(self, store: Store):
        oid = store.open_option("AAPL", "call", 100.0, time.time() + 7 * 86400,
                                iv=0.35, contracts=2.0, entry_premium=1.5,
                                genome_id="g1")
        open_rows = store.open_options()
        assert len(open_rows) == 1
        assert open_rows[0]["id"] == oid
        assert open_rows[0]["genome_id"] == "g1"
        assert open_rows[0]["status"] == "open"

        store.close_option(oid, exit_premium=2.5, pnl=2.0)
        assert store.open_options() == []
        closed = store.closed_options()
        assert len(closed) == 1
        assert closed[0]["pnl"] == pytest.approx(2.0)
        assert closed[0]["exit_premium"] == pytest.approx(2.5)

    def test_multiple_open_positions(self, store: Store):
        for k in ("call", "put"):
            store.open_option("SPY", k, 500.0, time.time() + 86400, 0.2, 1.0, 3.0, None)
        assert len(store.open_options()) == 2


class TestNewsCache:
    def test_latest_news_roundtrip(self, store: Store):
        assert store.latest_news("AAPL") is None
        store.add_news("AAPL", "old story", "neutral")
        store.add_news("AAPL", "new story", "bullish")
        latest = store.latest_news("AAPL")
        assert latest["text"] == "new story"
        assert latest["sentiment"] == "bullish"

    def test_recent_news_across_symbols(self, store: Store):
        store.add_news("AAPL", "a", None)
        store.add_news("MSFT", "b", "bearish")
        assert len(store.recent_news()) == 2


class TestBenchmark:
    def test_benchmark_curve_ordering(self, store: Store):
        store.record_benchmark(10.0)
        curve = store.benchmark_curve()
        assert len(curve) == 1
        assert curve[0]["equity"] == pytest.approx(10.0)
