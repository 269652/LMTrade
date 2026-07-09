"""Tests for the parameterized strategy library and the evolutionary optimizer
(strategies/library.py, strategies/optimizer.py). Written before implementation
per strict TDD."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from lmtrade.core.state import Store
from lmtrade.strategies.library import STRATEGIES, signal_for
from lmtrade.strategies.optimizer import Genome, StrategyOptimizer


def rising(n=80):  # steady uptrend
    return [100.0 * (1.01 ** i) for i in range(n)]


def falling(n=80):
    return [100.0 * (0.99 ** i) for i in range(n)]


def flat(n=80):
    return [100.0] * n


class TestStrategyLibrary:
    def test_registry_has_strategies(self):
        assert {"momentum", "mean_reversion", "breakout"} <= set(STRATEGIES)

    def test_momentum_direction(self):
        d, s = signal_for("momentum", rising(), {"fast": 5, "slow": 20, "threshold": 0.001})
        assert d == "buy" and 0 < s <= 1
        d, _ = signal_for("momentum", falling(), {"fast": 5, "slow": 20, "threshold": 0.001})
        assert d == "sell"
        d, _ = signal_for("momentum", flat(), {"fast": 5, "slow": 20, "threshold": 0.001})
        assert d == "hold"

    def test_mean_reversion_fades_extremes(self):
        # A sudden spike above a flat series should be sold (fade).
        spiked = flat(60) + [115.0]
        d, s = signal_for("mean_reversion", spiked, {"window": 20, "z_entry": 1.5})
        assert d == "sell" and s > 0
        dipped = flat(60) + [85.0]
        d, _ = signal_for("mean_reversion", dipped, {"window": 20, "z_entry": 1.5})
        assert d == "buy"

    def test_breakout_buys_new_highs(self):
        series = flat(60) + [101.0, 103.0]
        d, s = signal_for("breakout", series, {"lookback": 30})
        assert d == "buy" and s > 0
        series = flat(60) + [99.0, 97.0]
        d, _ = signal_for("breakout", series, {"lookback": 30})
        assert d == "sell"

    def test_insufficient_history_holds(self):
        for name in STRATEGIES:
            d, s = signal_for(name, [100.0, 101.0], STRATEGIES[name].default_params)
            assert d == "hold" and s == 0.0

    def test_strength_bounded(self):
        for name in STRATEGIES:
            _, s = signal_for(name, rising(), STRATEGIES[name].default_params)
            assert 0.0 <= s <= 1.0


@pytest.fixture()
def store(tmp_path: Path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


class TestOptimizer:
    def _opt(self, store, epsilon=0.0, population=6):
        return StrategyOptimizer(store, population=population, epsilon=epsilon,
                                 rng=random.Random(42))

    def test_initial_population(self, store):
        opt = self._opt(store)
        genomes = opt.genomes()
        assert len(genomes) == 6
        # Population spans multiple strategy types with valid params.
        assert len({g.strategy for g in genomes}) >= 2
        for g in genomes:
            assert g.strategy in STRATEGIES
            assert g.trades == 0 and g.pnl == 0.0

    def test_select_exploits_best_once_all_are_tried(self, store):
        # Contract updated for UCB1 selection: untried genomes are sampled
        # first (that's the point — epsilon-greedy starved them), so every
        # genome gets one trade before the proven best is expected to win.
        opt = self._opt(store, epsilon=0.0)
        genomes = opt.genomes()
        # Equal sample counts -> equal exploration bonus -> fitness decides.
        for _ in range(5):
            for g in genomes:
                opt.record_result(g.id, pnl=1.0 if g.id == genomes[2].id else 0.0)
        best = opt.select()
        assert best.id == genomes[2].id

    def test_record_result_accumulates(self, store):
        opt = self._opt(store)
        g = opt.genomes()[0]
        opt.record_result(g.id, 0.5)
        opt.record_result(g.id, -0.2)
        g2 = [x for x in opt.genomes() if x.id == g.id][0]
        assert g2.trades == 2
        assert g2.pnl == pytest.approx(0.3)

    def test_evolve_replaces_worst_with_mutant_of_best(self, store):
        opt = self._opt(store)
        genomes = opt.genomes()
        for _ in range(3):
            opt.record_result(genomes[0].id, 2.0)     # best
            opt.record_result(genomes[1].id, -2.0)    # worst
        worst_id = genomes[1].id
        opt.evolve()
        after = opt.genomes()
        assert len(after) == 6
        assert worst_id not in {g.id for g in after}
        # The mutant starts fresh.
        fresh = [g for g in after if g.trades == 0]
        assert fresh

    def test_persistence_roundtrip(self, store):
        opt = self._opt(store)
        g = opt.genomes()[0]
        opt.record_result(g.id, 1.23)
        # A new optimizer over the same store sees the same population.
        opt2 = StrategyOptimizer(store, population=6, epsilon=0.0,
                                 rng=random.Random(7))
        g2 = [x for x in opt2.genomes() if x.id == g.id][0]
        assert g2.pnl == pytest.approx(1.23)
        assert g2.trades == 1

    def test_genome_signal_end_to_end(self, store):
        opt = self._opt(store)
        g = opt.select()
        d, s = g.signal(rising())
        assert d in ("buy", "sell", "hold")
        assert 0.0 <= s <= 1.0
