"""Bandit-quality fixes for the strategy learner, written before the fix (TDD).

Observed live: one momentum genome banked 3 trades (+17.14) while every other
genome sat at 0 trades — epsilon-greedy's winner-takes-all pathology. Once any
genome's fitness beats the 0.01 optimistic prior it wins every exploit pick,
and 20% exploration split across 7 genomes (~3% each, further filtered by
gates/slots) never completes a trade for the rest.

Fixes under test:
1. UCB1 selection (Auer, Cesa-Bianchi & Fischer 2002): score = fitness +
   c*sqrt(ln N / n_i). Untried genomes are sampled first; under-sampled ones
   keep being revisited at a logarithmic rate. No genome starves.
2. Species protection in evolve(): the mutant always inherits the best
   genome's strategy, so without protection the population collapses to
   clones of one family. A family's last member is never replaced.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from lmtrade.core.state import Store
from lmtrade.strategies.optimizer import Genome, StrategyOptimizer


@pytest.fixture()
def store(tmp_path: Path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def opt_with(store, genomes: list[Genome], ucb_c: float = 2.0,
             epsilon: float = 0.0) -> StrategyOptimizer:
    o = StrategyOptimizer(store, population=len(genomes), epsilon=epsilon,
                          rng=random.Random(42), ucb_c=ucb_c)
    o._save(genomes)
    return o


def genome(id_, strategy="momentum", trades=0, pnl=0.0, wins=0, losses=0,
           win_sum=0.0, loss_sum=0.0) -> Genome:
    return Genome(id=id_, strategy=strategy, params={}, trades=trades, pnl=pnl,
                  wins=wins, losses=losses, win_sum=win_sum, loss_sum=loss_sum)


class TestUCBSelection:
    def test_untried_genomes_sampled_before_a_proven_winner(self, store):
        """The starvation regression: a hot genome must NOT monopolize
        selection while untried genomes exist."""
        o = opt_with(store, [
            genome("hot", trades=3, pnl=17.14, wins=3, win_sum=17.14),
            genome("cold1", strategy="mean_reversion"),
            genome("cold2", strategy="breakout"),
        ])
        assert o.select().id in ("cold1", "cold2")   # never "hot" while untried exist

    def test_all_untried_covered_before_any_repeat(self, store):
        o = opt_with(store, [
            genome("a"), genome("b", strategy="mean_reversion"),
            genome("c", strategy="breakout"),
        ])
        picked = set()
        for _ in range(3):
            g = o.select()
            picked.add(g.id)
            o.record_result(g.id, 0.1)               # now it's tried
        assert picked == {"a", "b", "c"}

    def test_equal_counts_exploits_best_fitness(self, store):
        o = opt_with(store, [
            genome("best", trades=5, pnl=5.0, wins=5, win_sum=5.0),
            genome("meh", strategy="mean_reversion", trades=5, pnl=1.0,
                   wins=5, win_sum=1.0),
        ])
        assert o.select().id == "best"

    def test_under_sampled_genome_gets_revisited(self, store):
        """Heavily-sampled winner vs barely-sampled decent genome: the
        exploration bonus must eventually favor the under-sampled one.
        fitness: 1.0+2*sqrt(ln52/50)=1.56 < 0.5+2*sqrt(ln52/2)=3.31."""
        o = opt_with(store, [
            genome("hot", trades=50, pnl=50.0, wins=50, win_sum=50.0),
            genome("under", strategy="breakout", trades=2, pnl=1.0,
                   wins=2, win_sum=1.0),
        ])
        assert o.select().id == "under"

    def test_epsilon_still_explores_randomly(self, store):
        o = opt_with(store, [
            genome("a", trades=5, pnl=5.0),
            genome("b", strategy="breakout", trades=5, pnl=0.0),
        ], epsilon=1.0)
        seen = {o.select().id for _ in range(20)}
        assert seen == {"a", "b"}                    # pure random hits both


class TestSpeciesProtection:
    def test_sole_family_member_survives_evolution(self, store):
        """The worst genome is the ONLY breakout — replacing it with a
        momentum mutant would extinct the family. It must survive; the
        next-worst genome from a multi-member family goes instead."""
        o = opt_with(store, [
            genome("m1", trades=5, pnl=10.0, wins=5, win_sum=10.0),
            genome("m2", trades=5, pnl=-1.0, losses=5, loss_sum=1.0),
            genome("b1", strategy="breakout", trades=5, pnl=-5.0,
                   losses=5, loss_sum=5.0),          # worst, but sole breakout
        ])
        o.evolve(min_trades=3)
        after = {g.strategy for g in o.genomes()}
        assert "breakout" in after                   # family not extinct
        ids = {g.id for g in o.genomes()}
        assert "b1" in ids                           # survivor is the original
        assert "m2" not in ids                       # next-worst got replaced

    def test_normal_evolution_when_worst_family_has_siblings(self, store):
        o = opt_with(store, [
            genome("m1", trades=5, pnl=10.0, wins=5, win_sum=10.0),
            genome("m2", trades=5, pnl=-5.0, losses=5, loss_sum=5.0),
            genome("m3", trades=5, pnl=1.0, wins=5, win_sum=1.0),
        ])
        o.evolve(min_trades=3)
        assert "m2" not in {g.id for g in o.genomes()}

    def test_no_replaceable_genome_is_a_noop(self, store):
        # Two families, one member each, both proven: nothing may be replaced.
        o = opt_with(store, [
            genome("m1", trades=5, pnl=10.0),
            genome("b1", strategy="breakout", trades=5, pnl=-5.0),
        ])
        assert o.evolve(min_trades=3) is None
        assert {g.id for g in o.genomes()} == {"m1", "b1"}
