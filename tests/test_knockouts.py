"""Tests for the Trade Republic-style knockout certificate / warrant
instrument model (finance/knockouts.py). Written before implementation per
strict TDD.

TR's real derivative catalog is warrants and knockout ("turbo") certificates
issued by partner banks — NOT standardized exchange options. A long KO is a
down-and-out barrier product: price ~ (spot - strike)/ratio + issuer premium,
leverage = spot/(spot - strike) (ratio cancels), and it becomes (near)
worthless the instant the underlying touches the barrier. The strike also
drifts up daily with financing costs. This model is what makes paper trading
on TR-realistic instruments honest.
"""
from __future__ import annotations

import pytest

from lmtrade.finance.knockouts import (
    Knockout,
    knockout_leverage,
    knockout_price,
    is_knocked_out,
    select_knockout,
    strike_after_financing,
)


class TestKnockoutPricing:
    def test_long_ko_price_is_intrinsic_over_ratio_plus_premium(self):
        # spot 100, strike 90, ratio 10, premium 0.02
        p = knockout_price(spot=100.0, strike=90.0, ratio=10.0, kind="ko_call",
                           premium=0.02)
        assert p == pytest.approx((100 - 90) / 10 + 0.02)

    def test_short_ko_price_mirrors(self):
        p = knockout_price(spot=100.0, strike=110.0, ratio=10.0, kind="ko_put",
                           premium=0.02)
        assert p == pytest.approx((110 - 100) / 10 + 0.02)

    def test_price_floors_at_zero_intrinsic(self):
        # long KO with spot below strike: knocked out territory; intrinsic 0
        p = knockout_price(spot=80.0, strike=90.0, ratio=10.0, kind="ko_call",
                           premium=0.0)
        assert p == 0.0

    def test_leverage_definition(self):
        # leverage = spot / (spot - strike) for long
        lev = knockout_leverage(spot=100.0, strike=90.0, kind="ko_call")
        assert lev == pytest.approx(10.0)
        lev = knockout_leverage(spot=100.0, strike=110.0, kind="ko_put")
        assert lev == pytest.approx(10.0)

    def test_higher_leverage_means_strike_closer_to_spot(self):
        lev5 = knockout_leverage(spot=100.0, strike=80.0, kind="ko_call")
        lev20 = knockout_leverage(spot=100.0, strike=95.0, kind="ko_call")
        assert lev20 > lev5


class TestBarrier:
    def test_long_ko_knocked_out_at_or_below_barrier(self):
        assert is_knocked_out(spot=89.9, barrier=90.0, kind="ko_call") is True
        assert is_knocked_out(spot=90.0, barrier=90.0, kind="ko_call") is True
        assert is_knocked_out(spot=90.1, barrier=90.0, kind="ko_call") is False

    def test_short_ko_knocked_out_at_or_above_barrier(self):
        assert is_knocked_out(spot=110.1, barrier=110.0, kind="ko_put") is True
        assert is_knocked_out(spot=110.0, barrier=110.0, kind="ko_put") is True
        assert is_knocked_out(spot=109.9, barrier=110.0, kind="ko_put") is False


class TestFinancing:
    def test_long_strike_drifts_up_daily(self):
        k = strike_after_financing(strike=90.0, kind="ko_call", days=365,
                                   annual_rate=0.03)
        assert k == pytest.approx(90.0 * 1.03, rel=1e-3)

    def test_short_strike_drifts_down(self):
        k = strike_after_financing(strike=110.0, kind="ko_put", days=365,
                                   annual_rate=0.03)
        assert k == pytest.approx(110.0 / 1.03, rel=1e-2)

    def test_zero_days_is_identity(self):
        assert strike_after_financing(90.0, "ko_call", 0, 0.03) == 90.0


class TestSelectKnockout:
    def test_builds_instrument_at_target_leverage(self):
        ko = select_knockout(underlying="AAPL", spot=100.0, direction="buy",
                             target_leverage=5.0)
        assert isinstance(ko, Knockout)
        assert ko.kind == "ko_call"
        # strike = spot*(1 - 1/L) for long
        assert ko.strike == pytest.approx(80.0, rel=1e-6)
        assert ko.barrier >= ko.strike       # barrier at or above strike (long)
        assert knockout_leverage(100.0, ko.strike, "ko_call") == pytest.approx(5.0)
        assert ko.price > 0

    def test_sell_direction_builds_short_ko(self):
        ko = select_knockout(underlying="AAPL", spot=100.0, direction="sell",
                             target_leverage=4.0)
        assert ko.kind == "ko_put"
        assert ko.strike == pytest.approx(125.0, rel=1e-6)
        assert ko.barrier <= ko.strike       # barrier at or below strike (short)

    def test_leverage_clamped_to_sane_range(self):
        ko = select_knockout(underlying="AAPL", spot=100.0, direction="buy",
                             target_leverage=500.0)   # absurd -> clamped
        lev = knockout_leverage(100.0, ko.strike, "ko_call")
        assert lev <= 20.0
