"""Shared TR-derivatives test double.

Policy: the engine has no synthetic-instrument fallback — paper AND live both
only ever trade real TR knockout instruments (paper simulates the fill; live
places a real order). Any test that expects the engine to open a position
must inject a TR client. Most tests only care that SOME plausible knockout is
available for whatever symbol/spot they use, not the specific instrument, so
this fake mirrors `finance.knockouts.select_knockout` (an ATM-ish strike at a
target leverage) rather than a fixed catalog — the resulting mark tracks the
entry premium sanely across cycles, so tests that exercise TP/SL/hold-time
logic over multiple cycles behave as if a real instrument were quoted.
"""
from __future__ import annotations

from lmtrade.brokers.tr_derivatives import TRDerivativeQuote, TRDerivativesBase
from lmtrade.finance.knockouts import DEFAULT_PREMIUM, DEFAULT_RATIO, knockout_price


class AnyKnockoutTR(TRDerivativesBase):
    """Returns a plausible ATM-ish knockout for ANY requested underlying at
    ANY spot, so tests don't need a per-symbol catalog."""

    def __init__(self, ratio: float = DEFAULT_RATIO, target_leverage: float = 5.0,
                 premium: float = DEFAULT_PREMIUM):
        self.ratio = ratio
        self.target_leverage = target_leverage
        self.premium = premium

    def available(self) -> bool:
        return True

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        return []   # unused directly — find_knockout is overridden below

    def find_knockout(self, underlying: str, direction: str, spot: float,
                      target_leverage: float) -> TRDerivativeQuote | None:
        if spot <= 0:
            return None
        kind = "ko_call" if direction == "buy" else "ko_put"
        lev = target_leverage or self.target_leverage
        if kind == "ko_call":
            strike = spot * (1.0 - 1.0 / lev)
        else:
            strike = spot * (1.0 + 1.0 / lev)
        barrier = strike   # classic turbo: barrier == strike
        price = knockout_price(spot, strike, self.ratio, kind, self.premium)
        if price <= 0:
            return None
        isin = f"DE000{underlying[:6].upper().ljust(6, 'X')}T"
        return TRDerivativeQuote(
            isin=isin, underlying=underlying, kind=kind,
            strike=round(strike, 4), barrier=round(barrier, 4), ratio=self.ratio,
            price=round(price, 6), leverage=lev, issuer="TestBank")
