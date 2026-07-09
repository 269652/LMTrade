"""Quantitative position sizing and regime detection.

Real, public quant methods — no magic claimed:

- kelly_fraction: the Kelly criterion f* = p - q/b sizes positions from the
  strategy's EMPIRICAL edge (win rate p, payoff ratio b = avg_win/avg_loss).
  We always use a *fraction* of Kelly downstream (full Kelly is famously too
  aggressive for estimated, non-stationary edges).
- vol_scale: volatility targeting — scale exposure down when realized vol
  exceeds the target so risk per position stays roughly constant across
  calm and wild markets. Never scales UP above 1.
- regime: a simple realized-vol regime filter (recent vol vs the series' own
  history). In a "storm" regime the engine demands more conviction and takes
  smaller size; trend/mean-reversion edges measured in calm markets are the
  first thing to break when volatility spikes.
"""
from __future__ import annotations

import math

TRADING_DAYS = 252.0


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly optimal fraction, floored at 0 (never size a negative edge)."""
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    p = max(0.0, min(1.0, win_rate))
    b = avg_win / avg_loss
    f = p - (1.0 - p) / b
    return max(0.0, f)


def _realized_annual_vol(history: list[float], window: int) -> float | None:
    if len(history) < window + 1:
        return None
    tail = history[-(window + 1):]
    rets = [(tail[i] - tail[i - 1]) / tail[i - 1]
            for i in range(1, len(tail)) if tail[i - 1] > 0]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def vol_scale(history: list[float], target_annual_vol: float = 0.20,
              window: int = 20) -> float:
    """Volatility-targeting multiplier in (0, 1]. 1 when realized vol is at
    or below target (we never leverage UP on quiet markets — the options/KO
    layer already provides leverage), shrinking proportionally above it."""
    vol = _realized_annual_vol(history, window)
    if vol is None or vol <= 0:
        return 1.0
    return min(1.0, target_annual_vol / vol)


def regime(history: list[float], short_window: int = 10,
           long_window: int = 60, storm_ratio: float = 1.5) -> str:
    """'storm' when recent realized vol exceeds the longer-run vol by
    storm_ratio, else 'calm'. Defaults calm on insufficient history."""
    recent = _realized_annual_vol(history, short_window)
    longer = _realized_annual_vol(history, long_window)
    if recent is None or longer is None or longer <= 0:
        return "calm"
    return "storm" if recent / longer >= storm_ratio else "calm"
