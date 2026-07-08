"""Tests for the walk-forward backtester (backtest/walk_forward.py) and its
CLI command. Written before implementation per strict TDD. Fully offline —
synthetic history only."""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lmtrade.cli as cli
from lmtrade.backtest.walk_forward import (
    WalkForward,
    fetch_history,
    simulate_genome,
)
from lmtrade.config import Settings
from lmtrade.core.state import Store
from lmtrade.strategies.optimizer import Genome

runner = CliRunner()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.data_dir = tmp_path
    s.learning.population = 6
    return s


def uptrend(n=300):
    return [100.0 * (1.005 ** i) for i in range(n)]


class TestFetchHistory:
    def test_synthetic_history_has_requested_length(self):
        h = fetch_history("AAPL", bars=250, provider="synthetic")
        assert len(h) == 250
        assert all(p > 0 for p in h)

    def test_deterministic_within_call(self):
        a = fetch_history("MSFT", bars=100, provider="synthetic")
        b = fetch_history("MSFT", bars=100, provider="synthetic")
        assert a == b

    def test_live_provider_uses_injected_fetcher(self):
        """Real data comes from the same httpx-based Yahoo fetcher as
        MarketData (not the yfinance package — see data/market.py for why).
        Offline: the fetcher is injected, never touches the network."""
        calls: list = []

        def fake_fetcher(symbol, range_, interval):
            calls.append((symbol, range_, interval))
            return [100.0 + i for i in range(300)]

        h = fetch_history("AAPL", bars=200, provider="auto", fetcher=fake_fetcher)
        assert len(h) == 200
        assert h == [100.0 + i for i in range(100, 300)]
        assert calls and calls[0][0] == "AAPL"

    def test_live_provider_falls_back_to_synthetic_on_fetch_failure(self):
        def failing_fetcher(symbol, range_, interval):
            raise RuntimeError("network unavailable")

        h = fetch_history("AAPL", bars=150, provider="auto", fetcher=failing_fetcher)
        assert len(h) == 150
        assert all(p > 0 for p in h)

    def test_live_provider_falls_back_when_too_few_bars_returned(self):
        def short_fetcher(symbol, range_, interval):
            return [100.0, 101.0, 102.0]   # far fewer than requested

        h = fetch_history("AAPL", bars=200, provider="auto", fetcher=short_fetcher)
        assert len(h) == 200   # synthetic fallback made up the length


class TestSimulateGenome:
    def _genome(self):
        return Genome(id="g1", strategy="momentum",
                      params={"fast": 5, "slow": 20, "threshold": 0.001})

    def test_returns_closed_trades_with_pnl(self, settings):
        trades = simulate_genome(self._genome(), uptrend(), settings)
        assert isinstance(trades, list)
        assert trades, "a momentum genome must trade a strong uptrend"
        for t in trades:
            assert "pnl" in t and "kind" in t
            assert t["kind"] in ("call", "put")

    def test_momentum_profits_on_strong_uptrend(self, settings):
        trades = simulate_genome(self._genome(), uptrend(), settings)
        assert sum(t["pnl"] for t in trades) > 0

    def test_short_history_yields_no_trades(self, settings):
        assert simulate_genome(self._genome(), [100.0] * 10, settings) == []


class TestWalkForward:
    def test_run_produces_folds_and_trains_population(self, settings):
        store = Store(settings.db_path)
        wf = WalkForward(settings, store, rng=random.Random(1))
        result = wf.run(symbols=["AAPL"], bars=300, train_bars=120, test_bars=60)
        store_genomes = store.get_meta("genomes")
        store.close()

        assert result["folds"], "must produce at least one walk-forward fold"
        for fold in result["folds"]:
            assert {"fold", "train_pnl", "test_pnl"} <= set(fold)
        # Learning happened: genomes carry recorded trades and persist.
        assert store_genomes
        assert any(g["trades"] > 0 for g in store_genomes)
        assert "total_test_pnl" in result
        assert "leaderboard" in result

    def test_fold_count_matches_data(self, settings):
        store = Store(settings.db_path)
        wf = WalkForward(settings, store, rng=random.Random(1))
        # 300 bars, 120 train + 60 test, stepping by test size -> 3 folds
        result = wf.run(symbols=["AAPL"], bars=300, train_bars=120, test_bars=60)
        store.close()
        assert len(result["folds"]) == 3

    def test_too_little_data_raises(self, settings):
        store = Store(settings.db_path)
        wf = WalkForward(settings, store)
        with pytest.raises(ValueError):
            wf.run(symbols=["AAPL"], bars=50, train_bars=120, test_bars=60)
        store.close()


class TestBacktestCLI:
    def test_backtest_command_runs(self, settings, monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        result = runner.invoke(cli.app, ["backtest", "--bars", "300",
                                         "--train", "120", "--test", "60"])
        assert result.exit_code == 0, result.output
        out = result.output.lower()
        assert "fold" in out
        assert "leaderboard" in out or "strategy" in out
