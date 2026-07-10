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


class TestClear:
    """Store.clear() backs `lmtrade reset`. Default keeps news/analysis (they
    are expensive to regather); a full purge is opt-in."""

    def _seed(self, store: Store) -> None:
        from lmtrade.core.state import Position, Trade
        store.set_meta("starting_cash", 100.0)
        store.set_meta("market_analysis", {"ts": 1.0, "symbols": {"AAPL": {}}})
        store.add_news("AAPL", "some real news. SENTIMENT: bullish", "bullish", ts=1.0)
        store.record_trade(Trade("AAPL", "buy", 1.0, 100.0))
        store.upsert_position(Position("AAPL", 1.0, 100.0, 1.0))
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=1.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0)
        store.add_log("info", "hello")
        store.add_activity("trade", "did a thing")
        store.record_cost("fee", 1.0, "options")

    def test_default_clear_keeps_news_and_analysis(self, store: Store):
        self._seed(store)
        store.clear()
        assert store.recent_news(10), "news must survive a default clear"
        assert store.get_meta("market_analysis") is not None

    def test_default_clear_wipes_everything_else(self, store: Store):
        self._seed(store)
        store.clear()
        assert store.recent_trades(10) == []
        assert store.positions() == []
        assert store.open_options() == []
        assert store.recent_logs(10) == []
        assert store.recent_activity(10) == []
        assert store.total_costs() == {}
        assert store.get_meta("starting_cash") is None

    def test_purge_news_clears_everything_including_news_and_analysis(self, store: Store):
        self._seed(store)
        store.clear(purge_news=True)
        assert store.recent_news(10) == []
        assert store.get_meta("market_analysis") is None


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


class TestPendingOptionLifecycle:
    """A live order goes pending -> open (TR-confirmed) -> closed, so the
    dashboard can show 'an order is in flight' instead of nothing appearing
    until full confirmation. Paper fills are unaffected (default status
    stays 'open', immediate)."""

    def test_pending_status_not_returned_by_open_options(self, store: Store):
        oid = store.open_option("AAPL", "ko_call", 100.0, time.time() + 86400,
                                iv=0.0, contracts=1.0, entry_premium=2.0,
                                genome_id=None, status="pending")
        assert store.open_options() == []
        pending = store.pending_options()
        assert len(pending) == 1
        assert pending[0]["id"] == oid
        assert pending[0]["status"] == "pending"

    def test_mark_option_open_flips_status(self, store: Store):
        oid = store.open_option("AAPL", "ko_call", 100.0, time.time() + 86400,
                                iv=0.0, contracts=1.0, entry_premium=2.0,
                                genome_id=None, status="pending")
        store.mark_option_open(oid)
        assert store.pending_options() == []
        open_rows = store.open_options()
        assert len(open_rows) == 1
        assert open_rows[0]["id"] == oid
        assert open_rows[0]["status"] == "open"

    def test_default_status_is_open_paper_unaffected(self, store: Store):
        oid = store.open_option("AAPL", "call", 100.0, time.time() + 86400,
                                iv=0.35, contracts=1.0, entry_premium=1.0,
                                genome_id=None)
        assert store.open_options()[0]["id"] == oid
        assert store.pending_options() == []

    def test_delete_option_removes_pending_row(self, store: Store):
        # A rejected/unconfirmed order's pending row must not linger.
        oid = store.open_option("AAPL", "ko_call", 100.0, time.time() + 86400,
                                iv=0.0, contracts=1.0, entry_premium=2.0,
                                genome_id=None, status="pending")
        store.delete_option(oid)
        assert store.pending_options() == []
        assert store.open_options() == []


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
