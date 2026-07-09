"""Trade Republic-style knockout certificate ("turbo") and warrant modeling.

TR's derivative catalog is issuer products (HVB, SocGen, Morgan Stanley, ...),
not exchange options. A long knockout is economically a down-and-out barrier
call bought at intrinsic value over a ratio:

    price  = max(0, spot - strike) / ratio + issuer_premium
    lever  = spot / (spot - strike)          (ratio cancels out)
    dead   = the instant spot touches the barrier (classic turbos:
             barrier == strike, worthless; stop-loss turbos: barrier above
             strike, small residual — we model the conservative worthless case)

The strike drifts with financing: the issuer finances (spot - price*ratio)
for you and charges interest by raising the strike daily (long) or lowering
it (short). This is why holding high-leverage KOs for weeks quietly bleeds.

This module gives paper trading an instrument model that behaves like what
Trade Republic actually sells, instead of pretending TR has vanilla options.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_LEVERAGE = 20.0      # sane retail range; TR lists higher but spreads eat you
MIN_LEVERAGE = 2.0
DEFAULT_RATIO = 10.0
DEFAULT_PREMIUM = 0.02   # issuer spread/premium per certificate
FINANCING_RATE = 0.03    # annual financing on the strike level


@dataclass
class Knockout:
    underlying: str
    kind: str              # ko_call | ko_put
    strike: float
    barrier: float
    ratio: float
    price: float           # per certificate
    leverage: float
    isin: str | None = None
    issuer: str = "synthetic"


def knockout_price(spot: float, strike: float, ratio: float, kind: str,
                   premium: float = DEFAULT_PREMIUM) -> float:
    if spot <= 0 or strike <= 0 or ratio <= 0:
        return 0.0
    intrinsic = (spot - strike) if kind == "ko_call" else (strike - spot)
    if intrinsic <= 0:
        return 0.0
    return intrinsic / ratio + premium


def knockout_leverage(spot: float, strike: float, kind: str) -> float:
    intrinsic = (spot - strike) if kind == "ko_call" else (strike - spot)
    if intrinsic <= 0:
        return float("inf")
    return spot / intrinsic


def is_knocked_out(spot: float, barrier: float, kind: str) -> bool:
    if kind == "ko_call":
        return spot <= barrier
    return spot >= barrier


def strike_after_financing(strike: float, kind: str, days: float,
                           annual_rate: float = FINANCING_RATE) -> float:
    """Financing drift: the issuer charges interest on the financed level by
    moving the strike against the holder — up for longs, down for shorts."""
    if days <= 0:
        return strike
    factor = (1.0 + annual_rate) ** (days / 365.0)
    return strike * factor if kind == "ko_call" else strike / factor


def select_knockout(underlying: str, spot: float, direction: str,
                    target_leverage: float, ratio: float = DEFAULT_RATIO,
                    premium: float = DEFAULT_PREMIUM) -> Knockout:
    """Construct a synthetic KO at (clamped) target leverage — the fallback
    instrument when no live TR catalog is available, and the template for
    ranking real TR instruments by leverage fit."""
    lev = max(MIN_LEVERAGE, min(MAX_LEVERAGE, target_leverage))
    kind = "ko_call" if direction == "buy" else "ko_put"
    if kind == "ko_call":
        strike = spot * (1.0 - 1.0 / lev)
        barrier = strike            # classic turbo: barrier == strike
    else:
        strike = spot * (1.0 + 1.0 / lev)
        barrier = strike
    price = knockout_price(spot, strike, ratio, kind, premium)
    return Knockout(underlying=underlying, kind=kind, strike=round(strike, 4),
                    barrier=round(barrier, 4), ratio=ratio,
                    price=round(price, 6), leverage=lev)
