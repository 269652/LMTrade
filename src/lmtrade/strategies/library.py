"""Parameterized trading strategies.

Each strategy is a pure function over a close-price series returning
(direction, strength). Parameters are plain dicts so the optimizer can mutate
them; each entry declares its default params and per-param mutation bounds.

Strategies also receive an optional market context —
{"symbol": str, "histories": {symbol: closes}, "benchmark": str} — supplied
by the engine from the current cycle's quotes. Single-series families ignore
it; cross-sectional families (xsmom, rel_value) need it and hold without it.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

SignalFn = Callable[..., tuple[str, float]]


@dataclass
class StrategySpec:
    name: str
    fn: SignalFn
    default_params: dict = field(default_factory=dict)
    # per-param (min, max) bounds the optimizer must respect when mutating
    bounds: dict = field(default_factory=dict)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def momentum(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
    fast, slow = int(p.get("fast", 10)), int(p.get("slow", 30))
    # Floor keeps a (mis)configured threshold of 0 from dividing by zero in
    # the strength scaling below.
    threshold = max(1e-9, float(p.get("threshold", 0.002)))
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


def mean_reversion(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
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


def breakout(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
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


def tsmom(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
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


def trend_pullback(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
    """Pullback-in-trend: trade only when a short-term z-score stretch runs
    AGAINST an intact medium-term trend — buy dips in uptrends, sell rips in
    downtrends. ANDing the trend and mean-reversion edges (instead of letting
    the fusion average their disagreement) is the classic pullback pattern
    (e.g. Connors-style entries)."""
    trend_window = int(p.get("trend_window", 40))
    z_window = int(p.get("z_window", 10))
    pullback_z = float(p.get("pullback_z", 1.0))
    if len(history) < max(trend_window, z_window) + 1:
        return "hold", 0.0
    last = history[-1]
    trend_ma = sum(history[-trend_window:]) / trend_window
    tail = history[-z_window:]
    mean = sum(tail) / z_window
    stdev = statistics.pstdev(tail)
    if stdev <= 1e-9:
        return "hold", 0.0
    z = (last - mean) / stdev
    strength = _clip01((abs(z) - pullback_z) / (2 * pullback_z))
    if last > trend_ma and z < -pullback_z:      # dip inside an uptrend
        return "buy", strength
    if last < trend_ma and z > pullback_z:       # rip inside a downtrend
        return "sell", strength
    return "hold", 0.0


def _k_return(h: list[float], k: int) -> float | None:
    if len(h) < k + 1 or h[-k - 1] <= 0:
        return None
    return (h[-1] - h[-k - 1]) / h[-k - 1]


def xsmom(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
    """Cross-sectional (relative-strength) momentum: rank the symbol's k-bar
    return against the rest of the universe; buy the leaders, fade the
    laggards (Jegadeesh & Titman 1993 — the most replicated equity anomaly).
    Needs the market context; holds without it or with too few peers."""
    k = int(p.get("lookback", 10))
    top_q = float(p.get("top_q", 0.25))
    min_peers = 5
    if not ctx:
        return "hold", 0.0
    own = _k_return(history, k)
    if own is None:
        return "hold", 0.0
    sym = ctx.get("symbol")
    peers = []
    for s, h in (ctx.get("histories") or {}).items():
        if s == sym:
            continue
        r = _k_return(h, k)
        if r is not None:
            peers.append(r)
    if len(peers) < min_peers:
        return "hold", 0.0
    pct = sum(1 for r in peers if r < own) / len(peers)   # percentile rank
    if pct >= 1 - top_q:
        return "buy", _clip01((pct - (1 - top_q)) / top_q)
    if pct <= top_q:
        return "sell", _clip01((top_q - pct) / top_q)
    return "hold", 0.0


def rel_value(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
    """Pairs-lite relative-value reversion: z-score of the symbol's cumulative
    return spread vs the benchmark over `window` bars; fade a rich spread,
    buy a cheap one (spread reversion in the spirit of Gatev et al. 2006,
    with the index standing in for the pair partner)."""
    w = int(p.get("window", 20))
    z_entry = float(p.get("z_entry", 1.5))
    if not ctx:
        return "hold", 0.0
    bench = (ctx.get("histories") or {}).get(ctx.get("benchmark"))
    if not bench or len(history) < w + 1 or len(bench) < w + 1:
        return "hold", 0.0
    hs, bs = history[-(w + 1):], bench[-(w + 1):]
    if hs[0] <= 0 or bs[0] <= 0:
        return "hold", 0.0
    rel = [hs[i] / hs[0] - bs[i] / bs[0] for i in range(w + 1)]
    mean = sum(rel) / len(rel)
    stdev = statistics.pstdev(rel)
    if stdev <= 1e-9:
        return "hold", 0.0
    z = (rel[-1] - mean) / stdev
    strength = _clip01((abs(z) - z_entry) / (2 * z_entry))
    if z > z_entry:
        return "sell", strength      # rich vs benchmark -> fade
    if z < -z_entry:
        return "buy", strength       # cheap vs benchmark
    return "hold", 0.0


def hotswap(history: list[float], p: dict, ctx: dict | None = None) -> tuple[str, float]:
    """High-confidence momentum strategy designed for hotswap eviction entries.

    Only signals when momentum is both present AND exceeds confidence_min,
    so a 'buy' from this strategy is a genuine strong conviction trade —
    the kind that justifies evicting an existing red position to make room.
    """
    fast = int(p.get("fast", 5))
    slow = int(p.get("slow", 20))
    threshold = float(p.get("threshold", 0.003))
    confidence_min = float(p.get("confidence_min", 0.65))
    if len(history) < slow + 1:
        return "hold", 0.0
    f = sum(history[-fast:]) / fast
    s = sum(history[-slow:]) / slow
    if s <= 0:
        return "hold", 0.0
    spread = (f - s) / s
    if spread > threshold:
        strength = _clip01(spread / (threshold * 10))
        if strength >= confidence_min:
            return "buy", strength
    if spread < -threshold:
        strength = _clip01(-spread / (threshold * 10))
        if strength >= confidence_min:
            return "sell", strength
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
    "trend_pullback": StrategySpec(
        "trend_pullback", trend_pullback,
        default_params={"trend_window": 40, "z_window": 10, "pullback_z": 1.0},
        bounds={"trend_window": (20, 50), "z_window": (5, 20),
                "pullback_z": (0.5, 2.5)},
    ),
    "xsmom": StrategySpec(
        "xsmom", xsmom,
        default_params={"lookback": 10, "top_q": 0.25},
        bounds={"lookback": (5, 40), "top_q": (0.1, 0.4)},
    ),
    "rel_value": StrategySpec(
        "rel_value", rel_value,
        default_params={"window": 20, "z_entry": 1.5},
        bounds={"window": (10, 40), "z_entry": (1.0, 3.0)},
    ),
    "hotswap": StrategySpec(
        "hotswap", hotswap,
        default_params={"fast": 5, "slow": 20, "threshold": 0.003,
                        "confidence_min": 0.65},
        bounds={"fast": (3, 10), "slow": (10, 40),
                "threshold": (0.001, 0.01), "confidence_min": (0.5, 0.9)},
    ),
}


def signal_for(name: str, history: list[float], params: dict,
               ctx: dict | None = None) -> tuple[str, float]:
    spec = STRATEGIES[name]
    return spec.fn(history, params, ctx)
