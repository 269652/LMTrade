"""Black-Scholes options pricing and a synthetic option chain.

Trade Republic has no public options-chain API (it trades warrants/knock-outs
via a private API), so for paper trading we synthesize a chain: strikes around
spot, implied vol estimated from realized vol, priced with Black-Scholes. This
gives the bot leveraged, defined-risk instruments (long calls/puts only) whose
P&L behaves like real short-dated options.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

RISK_FREE = 0.03
CONTRACT_SIZE = 1.0          # 1 unit of underlying per contract (warrant-like)
MIN_IV = 0.15
MAX_IV = 1.50


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    spot: float, strike: float, t_years: float, iv: float,
    kind: str, r: float = RISK_FREE,
) -> float:
    """Black-Scholes European option price. kind: 'call' | 'put'."""
    if t_years <= 0:
        intrinsic = spot - strike if kind == "call" else strike - spot
        return max(0.0, intrinsic)
    if spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, t_years: float, iv: float, kind: str) -> float:
    if t_years <= 0 or spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (RISK_FREE + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1.0


def realized_iv(history: list[float], periods_per_year: float = 252.0) -> float:
    """Annualized realized vol from a close series, clamped to sane IV bounds."""
    if len(history) < 10:
        return 0.35
    rets = [(history[i] - history[i - 1]) / history[i - 1]
            for i in range(1, len(history)) if history[i - 1] > 0]
    if not rets:
        return 0.35
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    iv = math.sqrt(var) * math.sqrt(periods_per_year)
    return max(MIN_IV, min(MAX_IV, iv))


@dataclass
class OptionQuote:
    underlying: str
    kind: str              # call | put
    strike: float
    expiry_ts: float
    iv: float
    premium: float         # per contract
    delta: float

    @property
    def t_years(self) -> float:
        return max(0.0, (self.expiry_ts - time.time()) / (365.0 * 86400.0))


def synth_option(
    underlying: str, spot: float, history: list[float], kind: str,
    expiry_days: float = 7.0, moneyness: float = 1.0,
) -> OptionQuote:
    """Synthesize a near-ATM option quote. moneyness: strike = spot * moneyness."""
    iv = realized_iv(history)
    strike = round(spot * moneyness, 2)
    expiry_ts = time.time() + expiry_days * 86400.0
    t = expiry_days / 365.0
    premium = bs_price(spot, strike, t, iv, kind)
    delta = bs_delta(spot, strike, t, iv, kind)
    return OptionQuote(underlying, kind, strike, expiry_ts, iv, max(0.01, premium), delta)


def mark_option(
    spot: float, strike: float, expiry_ts: float, iv: float, kind: str
) -> float:
    """Current mark of an existing option position (per contract)."""
    t = max(0.0, (expiry_ts - time.time()) / (365.0 * 86400.0))
    return bs_price(spot, strike, t, iv, kind)
