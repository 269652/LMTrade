"""Vol-scaled time-series momentum strategy family ("tsmom").

Direction = sign of the k-bar return; strength = the return's t-statistic
(ret / (per-bar vol * sqrt(k))), so conviction is high only when drift is
large RELATIVE to noise. Proven: time-series momentum (Moskowitz, Ooi &
Pedersen 2012); vol-managed momentum roughly doubles Sharpe and cuts the
crash tail (Barroso & Santa-Clara 2015). Learnable params: lookback,
vol_window, t_entry.

Also: ensure_families() — an existing persisted population (created before a
new family was registered) must gain the new family without extincting an
old one, or a running bot would never explore it. Written before
implementation per strict TDD."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from lmtrade.core.state import Store
from lmtrade.strategies.library import STRATEGIES, tsmom
from lmtrade.strategies.optimizer import Genome, StrategyOptimizer


def drift_series(n=80, drift=0.6, noise=0.4, seed=7, start=100.0) -> list[float]:
    rng = random.Random(seed)
    out = [start]
    for _ in range(n - 1):
        out.append(max(1.0, out[-1] + drift + rng.uniform(-noise, noise)))
    return out


P = {"lookback": 20, "vol_window": 20, "t_entry": 1.0}


class TestTsmomSignal:
    def test_strong_uptrend_buys(self):
        d, s = tsmom(drift_series(drift=0.6, noise=0.3), P)
        assert d == "buy"
        assert 0.0 < s <= 1.0

    def test_strong_downtrend_sells(self):
        d, s = tsmom(drift_series(drift=-0.6, noise=0.3, start=200.0), P)
        assert d == "sell"
        assert 0.0 < s <= 1.0

    def test_noisy_flat_holds(self):
        # Deterministic zigzag: real per-bar vol, ~zero k-bar drift -> t ~ 0.
        # (A random flat walk would make this a coin flip on the seed: over
        # k bars a driftless series' t-stat is ~N(0,1), which exceeds
        # t_entry=1 about a third of the time.)
        zigzag = [100.0 + 3.0 * (i % 2) for i in range(80)]
        d, _ = tsmom(zigzag, P)
        assert d == "hold"

    def test_same_drift_more_noise_means_less_conviction(self):
        # Moderate drift so the clean case clears t_entry without both cases
        # saturating the [0,1] clip (which would make them incomparable).
        _, s_clean = tsmom(drift_series(drift=0.3, noise=1.0), P)
        _, s_noisy = tsmom(drift_series(drift=0.3, noise=3.0), P)
        assert s_clean > s_noisy      # the whole point of vol-scaling

    def test_insufficient_history_holds(self):
        assert tsmom([100.0] * 10, P) == ("hold", 0.0)

    def test_zero_vol_holds(self):
        assert tsmom([100.0] * 80, P)[0] == "hold"

    def test_registered_with_bounds(self):
        assert "tsmom" in STRATEGIES
        spec = STRATEGIES["tsmom"]
        for key in ("lookback", "vol_window", "t_entry"):
            assert key in spec.default_params
            assert key in spec.bounds


class TestEnsureFamilies:
    def test_legacy_population_gains_new_family(self, tmp_path: Path):
        store = Store(tmp_path / "t.db")
        # Persist a pre-tsmom population (as an old deployment would have).
        old = [
            Genome(id="m1", strategy="momentum", params={}, trades=5, pnl=5.0),
            Genome(id="m2", strategy="momentum", params={}, trades=5, pnl=-1.0),
            Genome(id="r1", strategy="mean_reversion", params={}),
            Genome(id="b1", strategy="breakout", params={}),
        ]
        store.set_meta("genomes", [g.__dict__ for g in old])
        opt = StrategyOptimizer(store, population=4, rng=random.Random(1))
        families = {g.strategy for g in opt.genomes()}
        assert "tsmom" in families                 # new family injected
        assert {"mean_reversion", "breakout"} <= families   # sole members survive
        assert len(opt.genomes()) == 4             # population size respected
        store.close()

    def test_complete_population_untouched(self, tmp_path: Path):
        store = Store(tmp_path / "t.db")
        full = [Genome(id=f"g{i}", strategy=name, params={})
                for i, name in enumerate(STRATEGIES)]
        store.set_meta("genomes", [g.__dict__ for g in full])
        opt = StrategyOptimizer(store, population=len(full), rng=random.Random(1))
        assert {g.id for g in opt.genomes()} == {g.id for g in full}
        store.close()
