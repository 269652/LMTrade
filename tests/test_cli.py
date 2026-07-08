"""CLI tests (typer runner): status shows alpha/leaderboard, analyze runs the
daily review on demand. Written before implementation per strict TDD."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lmtrade.cli as cli
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store

runner = CliRunner()


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    monkeypatch.setattr(cli, "load_settings", lambda: s)
    return s


def seed_state(settings: Settings) -> None:
    store = Store(settings.db_path)
    engine = Engine(settings, store, PaperBroker(store, starting_cash=10.0, fee=0.1))
    engine.run_cycle()
    store.close()


class TestStatus:
    def test_status_shows_alpha_and_leaderboard(self, settings):
        seed_state(settings)
        result = runner.invoke(cli.app, ["status"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "alpha" in out
        assert "strategy" in out          # leaderboard table present
        assert "open options" in out or "options" in out


class TestStatusValuation:
    def test_status_net_worth_includes_open_options(self, settings):
        """Regression: status recomputed net worth from equity positions only,
        silently dropping the value tied up in open option premiums."""
        seed_state(settings)
        store = Store(settings.db_path)
        cash = float(store.get_meta("cash"))
        has_options = bool(store.open_options())
        store.close()
        if not has_options:
            pytest.skip("engine cycle opened no options this run")
        result = runner.invoke(cli.app, ["status"])
        # Net worth line must exceed bare cash when premiums are at risk.
        import re
        nw = float(re.search(r"Net worth\s*│\s*([\d.]+)", result.output).group(1))
        assert nw > cash + 1e-6


class TestAnalyze:
    def test_analyze_applies_fake_review(self, settings, monkeypatch):
        seed_state(settings)
        response = json.dumps({"risk": {"min_confidence": 0.65}, "notes": "n"})
        monkeypatch.setattr(
            "lmtrade.research.daily._anthropic_caller",
            lambda s: (lambda prompt: (response, 0.0)),
        )
        result = runner.invoke(cli.app, ["analyze"])
        assert result.exit_code == 0
        assert "min_confidence" in result.output

    def test_analyze_without_key_reports_unavailable(self, settings, monkeypatch):
        seed_state(settings)
        monkeypatch.setattr("lmtrade.research.daily._anthropic_caller", lambda s: None)
        result = runner.invoke(cli.app, ["analyze"])
        assert result.exit_code == 0
        assert "unavailable" in result.output.lower() or "no " in result.output.lower()
