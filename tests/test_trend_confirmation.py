"""Multi-timeframe trend confirmation (time-series momentum: Moskowitz-Ooi-
Pedersen 2012; Faber 2007).

A `trend` field (last price vs SMA-50) joins indicator_snapshot, and the
heuristic provider HALVES its conviction when its short-term signal fights
the medium-term trend. No boost when aligned (that's where overfit creeps
in), and a haircut rather than a veto so a strong mean-reversion signal can
still fire. Insufficient history -> trend None -> behavior unchanged.
Written before implementation per strict TDD."""
from __future__ import annotations

import pytest

from lmtrade.finance.indicators import indicator_snapshot, trend
from lmtrade.models.providers import HeuristicProvider


def uptrend_cross_down_medium() -> list[float]:
    """Medium-term DOWNTREND (last < SMA50) with a short-term bullish
    crossover (SMA10 > SMA30) and mid-range RSI: a counter-trend buy setup.
    Verified numerically: fast 137.6 > slow 134.2, last 139.4 < SMA50 161.5,
    RSI ~67.6 (no overbought/oversold term muddying the score)."""
    prices = [300.0 - 5 * i for i in range(36)]     # steep fall keeps SMA50 high
    level = prices[-1]
    for i in range(24):                              # oscillating shallow bounce
        level += 2.3 if i % 2 == 0 else -1.1
        prices.append(level)
    return prices[-60:]


def clean_uptrend() -> list[float]:
    """Oscillating uptrend (net +0.25/bar) — RSI ~60, not pinned at 100 the
    way a straight line is; a monotonic ramp trips the overbought term and
    tests the wrong thing."""
    out = [100.0]
    for i in range(59):
        out.append(out[-1] + (1.5 if i % 2 == 0 else -1.0))
    return out


def clean_downtrend() -> list[float]:
    out = [160.0]
    for i in range(59):
        out.append(out[-1] - (1.5 if i % 2 == 0 else -1.0))
    return out


class TestTrendIndicator:
    def test_up_and_down(self):
        assert trend(clean_uptrend()) == "up"
        assert trend(clean_downtrend()) == "down"

    def test_insufficient_history_is_none(self):
        assert trend([100.0] * 10) is None

    def test_snapshot_includes_trend(self):
        snap = indicator_snapshot(clean_uptrend())
        assert snap["trend"] == "up"

    def test_snapshot_trend_none_on_short_history(self):
        snap = indicator_snapshot([100.0] * 20)
        assert snap["trend"] is None


class TestHeuristicTrendHaircut:
    def _signal(self, prices):
        return HeuristicProvider().analyze("AAPL", {
            "indicators": indicator_snapshot(prices)})

    def test_aligned_buy_keeps_full_conviction(self):
        sig = self._signal(clean_uptrend())
        assert sig.direction == "buy"
        base = sig.confidence
        assert base >= 0.7          # crossover conviction, undamped

    def test_counter_trend_buy_is_dampened(self):
        prices = uptrend_cross_down_medium()
        snap = indicator_snapshot(prices)
        # Preconditions: bullish crossover fighting a medium downtrend.
        assert snap["sma_fast"] > snap["sma_slow"]
        assert snap["trend"] == "down"
        sig = self._signal(prices)
        aligned = self._signal(clean_uptrend())
        assert sig.direction in ("buy", "hold")
        assert sig.confidence < aligned.confidence   # haircut applied
        assert "counter-trend" in sig.rationale

    def test_no_trend_no_haircut(self):
        # 40 bars: crossover computable (needs 30), SMA50 not -> no dampening.
        prices = [100.0]
        for i in range(39):
            prices.append(prices[-1] + (1.5 if i % 2 == 0 else -1.0))
        sig = self._signal(prices)
        assert sig.direction == "buy"
        assert "counter-trend" not in sig.rationale

    def test_missing_indicators_still_degrade_to_hold(self):
        sig = HeuristicProvider().analyze("AAPL", {"indicators": {}})
        assert sig.direction == "hold"
