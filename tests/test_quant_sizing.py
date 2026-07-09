"""Tests for the quantitative sizing/regime layer (finance/sizing.py):
fractional Kelly from empirical per-genome win/loss stats, volatility
targeting, and a realized-vol regime filter. These are real, public quant
methods — not magic — and each degrades to the previous behavior when it
lacks data. Written before implementation per strict TDD."""
from __future__ import annotations

import pytest

from lmtrade.finance.sizing import kelly_fraction, regime, vol_scale
from lmtrade.strategies.optimizer import Genome


class TestKellyFraction:
    def test_positive_edge_gives_positive_fraction(self):
        # 60% win rate, wins twice the size of losses: f* = p - q/b = .6 - .4/2 = .4
        f = kelly_fraction(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert f == pytest.approx(0.4)

    def test_negative_edge_gives_zero(self):
        f = kelly_fraction(win_rate=0.3, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_degenerate_inputs_give_zero(self):
        assert kelly_fraction(0.6, 0.0, 1.0) == 0.0
        assert kelly_fraction(0.6, 1.0, 0.0) == 0.0

    def test_genome_tracks_win_loss_stats(self, tmp_path):
        from lmtrade.core.state import Store
        from lmtrade.strategies.optimizer import StrategyOptimizer
        import random

        store = Store(tmp_path / "t.db")
        opt = StrategyOptimizer(store, population=4, epsilon=0.0,
                                rng=random.Random(1))
        g = opt.genomes()[0]
        opt.record_result(g.id, 2.0)
        opt.record_result(g.id, -1.0)
        opt.record_result(g.id, 4.0)
        g2 = [x for x in opt.genomes() if x.id == g.id][0]
        assert g2.wins == 2 and g2.losses == 1
        assert g2.win_sum == pytest.approx(6.0)
        assert g2.loss_sum == pytest.approx(1.0)
        # kelly inputs derivable
        assert g2.win_rate == pytest.approx(2 / 3)
        assert g2.avg_win == pytest.approx(3.0)
        assert g2.avg_loss == pytest.approx(1.0)
        store.close()

    def test_old_persisted_genomes_without_stats_still_load(self, tmp_path):
        from lmtrade.core.state import Store
        from lmtrade.strategies.optimizer import StrategyOptimizer
        import random

        store = Store(tmp_path / "t.db")
        # Simulate a genome dict persisted before win/loss stats existed.
        store.set_meta("genomes", [{
            "id": "old1", "strategy": "momentum",
            "params": {"fast": 5, "slow": 20, "threshold": 0.002},
            "trades": 3, "pnl": 1.5,
        }])
        opt = StrategyOptimizer(store, population=4, epsilon=0.0,
                                rng=random.Random(1))
        g = [x for x in opt.genomes() if x.id == "old1"][0]
        assert g.wins == 0 and g.losses == 0   # defaults, no crash
        store.close()


class TestVolScale:
    def test_calm_series_no_downscale(self):
        # tiny realized vol -> scale capped at 1.0 (never size UP)
        history = [100 + 0.01 * i for i in range(60)]
        assert vol_scale(history, target_annual_vol=0.20) == 1.0

    def test_wild_series_scales_down(self):
        history = [100 * (1.10 if i % 2 else 0.90) ** 1 for i in range(60)]
        s = vol_scale(history, target_annual_vol=0.20)
        assert 0.0 < s < 1.0

    def test_short_history_neutral(self):
        assert vol_scale([100.0, 101.0], target_annual_vol=0.20) == 1.0


class TestRegime:
    def test_calm_when_recent_vol_below_history(self):
        # long calm history, calm tail
        history = [100 + (i % 3) * 0.1 for i in range(120)]
        assert regime(history) == "calm"

    def test_storm_when_recent_vol_spikes(self):
        calm = [100 + (i % 3) * 0.1 for i in range(100)]
        wild = [calm[-1] * (1.05 if i % 2 else 0.95) ** 1 for i in range(20)]
        assert regime(calm + wild) == "storm"

    def test_short_history_defaults_calm(self):
        assert regime([100.0] * 10) == "calm"
