"""Cross-sectional strategy families + market-context plumbing.

- xsmom: cross-sectional (relative-strength) momentum — rank the symbol's
  k-bar return against the rest of the universe; favor leaders, fade
  laggards (Jegadeesh & Titman 1993, the most replicated equity anomaly).
- rel_value: pairs-lite relative-value reversion — z-score of the symbol's
  cumulative return spread vs the benchmark (SPY); fade rich, buy cheap
  (spread reversion in the spirit of Gatev et al. 2006).

Both need to see MORE than their own price series, so strategies now accept
an optional market context {symbol, histories, benchmark} which the engine
provides from the cycle's quotes. No context -> hold (degrade gracefully).
Written before implementation per strict TDD."""
from __future__ import annotations

import pytest

from lmtrade.strategies.library import STRATEGIES, rel_value, signal_for, xsmom

XP = {"lookback": 10, "top_q": 0.25}
RP = {"window": 20, "z_entry": 1.5}


def flat(n=60, level=100.0) -> list[float]:
    return [level] * n


def riser(n=60, step=1.0) -> list[float]:
    return [100.0 + step * i for i in range(n)]


def faller(n=60, step=1.0) -> list[float]:
    return [200.0 - step * i for i in range(n)]


def ctx_for(symbol: str, histories: dict, benchmark: str = "SPY") -> dict:
    return {"symbol": symbol, "histories": histories, "benchmark": benchmark}


class TestXsmom:
    def _universe(self) -> dict:
        # One clear leader, one clear laggard, five flat-ish peers.
        u = {f"P{i}": [100.0 + 0.01 * i * j for j in range(60)] for i in range(5)}
        u["LEAD"] = riser(step=2.0)
        u["LAG"] = faller(step=2.0)
        return u

    def test_leader_buys(self):
        u = self._universe()
        d, s = xsmom(u["LEAD"], XP, ctx_for("LEAD", u))
        assert d == "buy"
        assert 0.0 < s <= 1.0

    def test_laggard_sells(self):
        u = self._universe()
        d, s = xsmom(u["LAG"], XP, ctx_for("LAG", u))
        assert d == "sell"
        assert 0.0 < s <= 1.0

    def test_middle_of_pack_holds(self):
        u = self._universe()
        d, _ = xsmom(u["P2"], XP, ctx_for("P2", u))
        assert d == "hold"

    def test_no_context_holds(self):
        assert xsmom(riser(), XP, None) == ("hold", 0.0)

    def test_too_few_peers_holds(self):
        u = {"A": riser(), "B": faller()}
        assert xsmom(u["A"], XP, ctx_for("A", u))[0] == "hold"

    def test_registered(self):
        assert "xsmom" in STRATEGIES
        assert "top_q" in STRATEGIES["xsmom"].bounds


class TestRelValue:
    def test_symbol_rich_vs_flat_benchmark_sells(self):
        sym = flat(50) + [100.0 + 2.0 * i for i in range(1, 11)]   # late run-up
        u = {"X": sym, "SPY": flat(60)}
        d, s = rel_value(sym, RP, ctx_for("X", u))
        assert d == "sell"
        assert 0.0 < s <= 1.0

    def test_symbol_cheap_vs_flat_benchmark_buys(self):
        sym = flat(50) + [100.0 - 2.0 * i for i in range(1, 11)]
        u = {"X": sym, "SPY": flat(60)}
        d, _ = rel_value(sym, RP, ctx_for("X", u))
        assert d == "buy"

    def test_moving_with_benchmark_holds(self):
        # Symbol and benchmark rally together: no spread, nothing to fade.
        u = {"X": riser(), "SPY": riser()}
        assert rel_value(u["X"], RP, ctx_for("X", u))[0] == "hold"

    def test_missing_benchmark_holds(self):
        assert rel_value(riser(), RP, ctx_for("X", {"X": riser()})) == ("hold", 0.0)

    def test_no_context_holds(self):
        assert rel_value(riser(), RP, None) == ("hold", 0.0)

    def test_registered(self):
        assert "rel_value" in STRATEGIES
        assert "z_entry" in STRATEGIES["rel_value"].bounds


class TestContextPlumbing:
    def test_signal_for_forwards_context(self):
        u = {"LEAD": riser(step=2.0),
             **{f"P{i}": flat() for i in range(6)}}
        d, _ = signal_for("xsmom", u["LEAD"], XP, ctx_for("LEAD", u))
        assert d == "buy"

    def test_legacy_strategies_ignore_context(self):
        # Old single-series families must accept (and ignore) a context.
        d, _ = signal_for("momentum", riser(), {"fast": 10, "slow": 30,
                                                "threshold": 0.0},
                          ctx_for("X", {}))
        assert d in ("buy", "sell", "hold")
