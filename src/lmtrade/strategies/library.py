"""Parameterized trading strategies.

Each strategy is a pure function over a close-price series returning
(direction, strength). Parameters are plain dicts so the optimizer can mutate
them; each entry declares its default params and per-param mutation bounds.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

SignalFn = Callable[[list[float], dict], tuple[str, float]]


@dataclass
class StrategySpec:
    name: str
    fn: SignalFn
    default_params: dict = field(default_factory=dict)
    # per-param (min, max) bounds the optimizer must respect when mutating
    bounds: dict = field(default_factory=dict)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def momentum(history: list[float], p: dict) -> tuple[str, float]:
    fast, slow = int(p.get("fast", 10)), int(p.get("slow", 30))
    threshold = float(p.get("threshold", 0.002))
    if len(history) < slow + 1:
        return "hold", 0.0
    f = sum(history[-fast:]) / fast
    s = sum(history[-slow:]) / slow
    if s <= 0:
        return "hold", 0.0
    spread = (f - s) / s
    if spread > threshold:
        return "buy", _clip01(spread / (threshold * 10))
    if spread < -threshold:
        return "sell", _clip01(-spread / (threshold * 10))
    return "hold", 0.0


def mean_reversion(history: list[float], p: dict) -> tuple[str, float]:
    window = int(p.get("window", 20))
    z_entry = float(p.get("z_entry", 1.5))
    if len(history) < window + 1:
        return "hold", 0.0
    tail = history[-window:]
    mean = sum(tail) / window
    stdev = statistics.pstdev(tail)
    if stdev <= 1e-9:
        return "hold", 0.0
    z = (history[-1] - mean) / stdev
    if z > z_entry:            # stretched above the mean -> fade it
        return "sell", _clip01((z - z_entry) / z_entry)
    if z < -z_entry:
        return "buy", _clip01((-z - z_entry) / z_entry)
    return "hold", 0.0


def breakout(history: list[float], p: dict) -> tuple[str, float]:
    lookback = int(p.get("lookback", 30))
    if len(history) < lookback + 2:
        return "hold", 0.0
    window = history[-(lookback + 1):-1]
    hi, lo = max(window), min(window)
    last = history[-1]
    span = max(1e-9, hi - lo) if hi > lo else max(1e-9, hi * 0.01)
    if last > hi:
        return "buy", _clip01((last - hi) / span)
    if last < lo:
        return "sell", _clip01((lo - last) / span)
    return "hold", 0.0


def tsmom(history: list[float], p: dict) -> tuple[str, float]:
    """Vol-scaled time-series momentum (Moskowitz-Ooi-Pedersen 2012; vol
    management per Barroso & Santa-Clara 2015). Direction is the sign of the
    k-bar return; conviction is that return's t-statistic — drift divided by
    noise*sqrt(k) — so a grinding low-vol trend scores higher than the same
    move in a churning market. Entry requires |t| > t_entry."""
    k = int(p.get("lookback", 20))
    vol_window = int(p.get("vol_window", 20))
    t_entry = float(p.get("t_entry", 1.0))
    need = max(k + 1, vol_window + 1)
    if len(history) < need or history[-k - 1] <= 0:
        return "hold", 0.0
    ret = (history[-1] - history[-k - 1]) / history[-k - 1]
    tail = history[-(vol_window + 1):]
    rets = [(tail[i] - tail[i - 1]) / tail[i - 1]
            for i in range(1, len(tail)) if tail[i - 1] > 0]
    if len(rets) < 2:
        return "hold", 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    vol = var ** 0.5
    if vol <= 1e-12:
        return "hold", 0.0
    t_stat = ret / (vol * (k ** 0.5))
    if t_stat > t_entry:
        return "buy", _clip01((t_stat - t_entry) / (2 * t_entry))
    if t_stat < -t_entry:
        return "sell", _clip01((-t_stat - t_entry) / (2 * t_entry))
    return "hold", 0.0


STRATEGIES: dict[str, StrategySpec] = {
    "momentum": StrategySpec(
        "momentum", momentum,
        default_params={"fast": 10, "slow": 30, "threshold": 0.002},
        bounds={"fast": (3, 20), "slow": (21, 60), "threshold": (0.0005, 0.02)},
    ),
    "mean_reversion": StrategySpec(
        "mean_reversion", mean_reversion,
        default_params={"window": 20, "z_entry": 1.5},
        bounds={"window": (10, 50), "z_entry": (1.0, 3.0)},
    ),
    "breakout": StrategySpec(
        "breakout", breakout,
        default_params={"lookback": 30},
        bounds={"lookback": (10, 60)},
    ),
    "tsmom": StrategySpec(
        "tsmom", tsmom,
        default_params={"lookback": 20, "vol_window": 20, "t_entry": 1.0},
        bounds={"lookback": (10, 60), "vol_window": (10, 40),
                "t_entry": (0.5, 2.5)},
    ),
}


def signal_for(name: str, history: list[float], params: dict) -> tuple[str, float]:
    spec = STRATEGIES[name]
    return spec.fn(history, params)
