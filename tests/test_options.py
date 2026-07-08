"""Tests for the Black-Scholes options layer (finance/options.py)."""
from __future__ import annotations

import time

import pytest

from lmtrade.finance.options import (
    MAX_IV,
    MIN_IV,
    bs_delta,
    bs_price,
    mark_option,
    realized_iv,
    synth_option,
)


class TestBlackScholes:
    def test_atm_call_and_put_have_positive_value(self):
        c = bs_price(100, 100, 0.25, 0.3, "call")
        p = bs_price(100, 100, 0.25, 0.3, "put")
        assert c > 0 and p > 0

    def test_put_call_parity(self):
        # C - P = S - K*exp(-rT)
        import math
        s, k, t, iv, r = 100.0, 95.0, 0.5, 0.4, 0.03
        c = bs_price(s, k, t, iv, "call", r)
        p = bs_price(s, k, t, iv, "put", r)
        assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-6)

    def test_expiry_returns_intrinsic(self):
        assert bs_price(110, 100, 0.0, 0.3, "call") == pytest.approx(10.0)
        assert bs_price(90, 100, 0.0, 0.3, "put") == pytest.approx(10.0)
        assert bs_price(90, 100, 0.0, 0.3, "call") == 0.0

    def test_deeper_itm_call_worth_more(self):
        assert bs_price(120, 100, 0.1, 0.3, "call") > bs_price(105, 100, 0.1, 0.3, "call")

    def test_invalid_inputs_return_zero(self):
        assert bs_price(0, 100, 0.5, 0.3, "call") == 0.0
        assert bs_price(100, 0, 0.5, 0.3, "call") == 0.0
        assert bs_price(100, 100, 0.5, 0.0, "call") == 0.0

    def test_delta_bounds(self):
        assert 0.0 < bs_delta(100, 100, 0.25, 0.3, "call") < 1.0
        assert -1.0 < bs_delta(100, 100, 0.25, 0.3, "put") < 0.0
        assert bs_delta(100, 100, 0.0, 0.3, "call") == 0.0


class TestRealizedIV:
    def test_clamped_to_bounds(self):
        flat = [100.0] * 50                      # zero vol -> clamped up to MIN_IV
        assert realized_iv(flat) == MIN_IV
        wild = [100.0 * (1.5 if i % 2 else 0.5) for i in range(50)]
        assert realized_iv(wild) == MAX_IV

    def test_short_history_falls_back(self):
        assert realized_iv([100.0, 101.0]) == 0.35

    def test_reasonable_series_within_bounds(self):
        series = [100 + (i % 7) - 3 for i in range(60)]
        iv = realized_iv([float(x) for x in series])
        assert MIN_IV <= iv <= MAX_IV


class TestSynthChain:
    def test_synth_option_fields(self):
        hist = [100 + (i % 5) for i in range(60)]
        q = synth_option("AAPL", 102.0, [float(x) for x in hist], "call", expiry_days=7)
        assert q.underlying == "AAPL" and q.kind == "call"
        assert q.strike == pytest.approx(102.0)
        assert q.premium > 0
        assert q.expiry_ts > time.time()
        assert 0 < q.delta < 1

    def test_moneyness_shifts_strike(self):
        hist = [float(100)] * 60
        q = synth_option("SPY", 100.0, hist, "put", moneyness=0.95)
        assert q.strike == pytest.approx(95.0)

    def test_mark_decays_toward_intrinsic(self):
        # An OTM call marked at (nearly) expiry is worth (nearly) nothing.
        soon = time.time() + 60.0
        far = time.time() + 30 * 86400.0
        near_mark = mark_option(95.0, 100.0, soon, 0.4, "call")
        far_mark = mark_option(95.0, 100.0, far, 0.4, "call")
        assert near_mark < far_mark
        assert near_mark < 0.5
