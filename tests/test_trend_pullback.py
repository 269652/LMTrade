"""Trend-pullback strategy family: enter only on a short-term pullback
(z-score) AGAINST the medium-term trend direction — "buy dips in uptrends,
sell rips in downtrends" (the classic pullback-in-trend pattern, e.g.
Connors' RSI-2 variants). ANDs the bot's two existing edges (trend +
mean-reversion) instead of letting fusion average their disagreement.
Written before implementation per strict TDD."""
from __future__ import annotations

import pytest

from lmtrade.strategies.library import STRATEGIES, trend_pullback

P = {"trend_window": 40, "z_window": 10, "pullback_z": 1.0}


def uptrend_with_dip() -> list[float]:
    """55 rising bars, then a 5-bar dip: still above the trend SMA (uptrend
    intact) but locally stretched to the downside."""
    prices = [100.0 + i for i in range(55)]
    for _ in range(5):
        prices.append(prices[-1] - 2.0)
    return prices


def downtrend_with_rip() -> list[float]:
    prices = [200.0 - i for i in range(55)]
    for _ in range(5):
        prices.append(prices[-1] + 2.0)
    return prices


def steady_uptrend() -> list[float]:
    return [100.0 + i for i in range(60)]


class TestTrendPullback:
    def test_dip_in_uptrend_buys(self):
        d, s = trend_pullback(uptrend_with_dip(), P)
        assert d == "buy"
        assert 0.0 < s <= 1.0

    def test_rip_in_downtrend_sells(self):
        d, s = trend_pullback(downtrend_with_rip(), P)
        assert d == "sell"
        assert 0.0 < s <= 1.0

    def test_no_pullback_holds(self):
        # Trend up but price at local highs: nothing to buy into.
        assert trend_pullback(steady_uptrend(), P)[0] == "hold"

    def test_dip_deep_enough_to_break_trend_holds(self):
        # The dip takes price BELOW the trend SMA -> no longer an uptrend
        # pullback, just a downtrend. Must not catch the falling knife.
        prices = [100.0 + i for i in range(55)]
        for _ in range(20):
            prices.append(prices[-1] - 6.0)
        d, _ = trend_pullback(prices, P)
        assert d != "buy"

    def test_insufficient_history_holds(self):
        assert trend_pullback([100.0] * 20, P) == ("hold", 0.0)

    def test_zero_variance_holds(self):
        assert trend_pullback([100.0] * 60, P)[0] == "hold"

    def test_registered_with_bounds(self):
        assert "trend_pullback" in STRATEGIES
        spec = STRATEGIES["trend_pullback"]
        for key in ("trend_window", "z_window", "pullback_z"):
            assert key in spec.default_params
            assert key in spec.bounds
