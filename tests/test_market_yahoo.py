"""Tests for the httpx-based Yahoo Finance data path in data/market.py.

`yfinance`'s curl_cffi backend does browser TLS-fingerprint impersonation,
which this sandbox's MITM egress proxy cannot pass through (connection reset
during handshake — confirmed against the real proxy). Plain httpx against
Yahoo's public chart JSON endpoint works fine, so real market data is fetched
directly instead of through the yfinance package. All tests here are offline:
the fetcher is injected, never hits the network. Written before the
implementation per strict TDD."""
from __future__ import annotations

import pytest

from lmtrade.data.market import MarketData


def fake_fetcher_factory(calls: list, closes_by_key: dict | None = None,
                         raise_on: set | None = None):
    """Build a fake (symbol, range_, interval) -> list[float] fetcher that
    records calls and can be made to fail for specific (range_, interval)."""
    closes_by_key = closes_by_key or {}
    raise_on = raise_on or set()

    def fetcher(symbol: str, range_: str, interval: str) -> list[float]:
        calls.append((symbol, range_, interval))
        if (range_, interval) in raise_on:
            raise RuntimeError("simulated fetch failure")
        return closes_by_key.get((range_, interval), [100.0 + i for i in range(80)])

    return fetcher


class TestYahooQuote:
    def test_daily_quote_uses_injected_fetcher(self):
        calls: list = []
        fetcher = fake_fetcher_factory(calls)
        md = MarketData("yahoo", lookback=30, intraday=False, fetcher=fetcher)
        q = md.quote("AAPL")
        assert q.source == "yahoo"
        assert q.price == q.history[-1]
        assert len(q.history) == 30
        assert calls == [("AAPL", "3mo", "1d")]

    def test_intraday_quote_requests_1m_bars(self):
        calls: list = []
        fetcher = fake_fetcher_factory(calls)
        md = MarketData("yahoo", lookback=20, intraday=True, fetcher=fetcher)
        md.quote("MSFT")
        assert calls == [("MSFT", "1d", "1m")]

    def test_intraday_falls_back_to_daily_when_market_closed(self):
        calls: list = []
        # 1m request returns almost nothing (market closed) -> daily fallback
        fetcher = fake_fetcher_factory(
            calls, closes_by_key={("1d", "1m"): [101.0]})
        md = MarketData("yahoo", lookback=20, intraday=True, fetcher=fetcher)
        q = md.quote("MSFT")
        assert calls == [("MSFT", "1d", "1m"), ("MSFT", "3mo", "1d")]
        assert q.source == "yahoo"
        assert len(q.history) == 20

    def test_fetch_failure_falls_back_to_synthetic(self):
        calls: list = []
        fetcher = fake_fetcher_factory(calls, raise_on={("3mo", "1d")})
        md = MarketData("yahoo", lookback=20, intraday=False, fetcher=fetcher)
        q = md.quote("AAPL")
        assert q.source == "synthetic"
        assert q.price > 0

    def test_empty_response_falls_back_to_synthetic(self):
        calls: list = []
        fetcher = fake_fetcher_factory(calls, closes_by_key={("3mo", "1d"): []})
        md = MarketData("yahoo", lookback=20, intraday=False, fetcher=fetcher)
        q = md.quote("AAPL")
        assert q.source == "synthetic"

    def test_legacy_yfinance_provider_name_still_means_live_data(self):
        """Config files may still say provider: yfinance from before the
        curl_cffi/proxy issue was found; it must keep working, just via the
        httpx path instead of the yfinance package."""
        calls: list = []
        fetcher = fake_fetcher_factory(calls)
        md = MarketData("yfinance", lookback=10, fetcher=fetcher)
        assert md.provider == "yahoo"
        q = md.quote("SPY")
        assert q.source == "yahoo"

    def test_auto_provider_means_live_data(self):
        calls: list = []
        fetcher = fake_fetcher_factory(calls)
        md = MarketData("auto", lookback=10, fetcher=fetcher)
        assert md.provider == "yahoo"

    def test_synthetic_provider_never_calls_fetcher(self):
        calls: list = []
        fetcher = fake_fetcher_factory(calls)
        md = MarketData("synthetic", lookback=10, fetcher=fetcher)
        md.quote("AAPL")
        assert calls == []


class TestDefaultFetcherIsNotUsedInTests:
    def test_default_fetcher_only_constructed_lazily(self):
        """Constructing MarketData without a fetcher must not touch the
        network — the default httpx fetcher is only invoked on quote()."""
        md = MarketData("yahoo", fetcher=None)
        assert md.fetcher is not None  # a default is wired in, but not called yet
