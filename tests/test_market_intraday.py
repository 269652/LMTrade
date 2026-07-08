"""Tests for intraday (1m-bar) market data support in data/market.py and its
config plumbing. Written before implementation per strict TDD."""
from __future__ import annotations

from lmtrade.config import Settings
from lmtrade.data.market import MarketData


class TestIntradayConfig:
    def test_settings_parse_intraday_flag(self):
        s = Settings(data={"provider": "synthetic", "intraday": True})
        assert s.data.intraday is True
        s2 = Settings(data={"provider": "synthetic"})
        assert s2.data.intraday is True   # default on for high-cadence

    def test_market_accepts_intraday(self):
        md = MarketData("synthetic", lookback=30, intraday=True)
        q = md.quote("AAPL")
        assert q.price > 0
        assert len(q.history) == 30
        assert q.source == "synthetic"

    def test_intraday_series_differs_from_daily(self):
        # Different time bucketing must give a different (but valid) series.
        md_daily = MarketData("synthetic", lookback=30, intraday=False)
        md_intra = MarketData("synthetic", lookback=30, intraday=True)
        q_d = md_daily.quote("MSFT")
        q_i = md_intra.quote("MSFT")
        assert q_d.history != q_i.history
        assert all(p > 0 for p in q_i.history)
