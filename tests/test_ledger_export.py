"""Tests for incremental trade-ledger export (a diffable CSV committed to the
bot-state branch alongside the DB each hourly run — unlike the opaque SQLite
file, trade history is reviewable directly via git). Written before
implementation per strict TDD."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lmtrade.cli as cli
from lmtrade.config import Settings
from lmtrade.core.state import Store, Trade

runner = CliRunner()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"])
    s.data_dir = tmp_path
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


class TestTradesSince:
    def test_returns_only_rows_after_marker(self, store):
        store.record_trade(Trade("AAPL", "buy", 1.0, 100.0))
        store.record_trade(Trade("MSFT", "buy", 2.0, 200.0))
        first_id = store.recent_trades(10)[-1]["id"]
        store.record_trade(Trade("SPY", "sell", 1.0, 300.0))
        rows = store.trades_since(first_id)
        assert [r["symbol"] for r in rows] == ["MSFT", "SPY"]

    def test_since_zero_returns_all(self, store):
        store.record_trade(Trade("AAPL", "buy", 1.0, 100.0))
        store.record_trade(Trade("MSFT", "buy", 2.0, 200.0))
        assert len(store.trades_since(0)) == 2

    def test_since_latest_returns_empty(self, store):
        store.record_trade(Trade("AAPL", "buy", 1.0, 100.0))
        latest = store.recent_trades(1)[0]["id"]
        assert store.trades_since(latest) == []


class TestExportLedgerCLI:
    def test_creates_csv_with_header_and_rows(self, settings, store, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        store.record_trade(Trade("AAPL", "buy", 1.5, 100.0, fee=0.1,
                                 reason="test open", confidence=0.9))
        store.close()

        out = tmp_path / "ledger.csv"
        result = runner.invoke(cli.app, ["export-ledger", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["side"] == "buy"

    def test_second_call_appends_only_new_rows(self, settings, store, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        store.record_trade(Trade("AAPL", "buy", 1.0, 100.0))
        out = tmp_path / "ledger.csv"
        runner.invoke(cli.app, ["export-ledger", str(out)])

        store.record_trade(Trade("MSFT", "buy", 2.0, 200.0))
        store.close()
        result = runner.invoke(cli.app, ["export-ledger", str(out)])
        assert result.exit_code == 0

        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[-1]["symbol"] == "MSFT"

    def test_no_new_trades_is_a_noop(self, settings, store, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        store.record_trade(Trade("AAPL", "buy", 1.0, 100.0))
        out = tmp_path / "ledger.csv"
        runner.invoke(cli.app, ["export-ledger", str(out)])
        store.close()
        result = runner.invoke(cli.app, ["export-ledger", str(out)])
        assert result.exit_code == 0
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_marker_persists_across_cli_invocations(self, settings, store, tmp_path, monkeypatch):
        """Regression: the 'last exported id' marker must survive process
        restarts (each hourly firing is a fresh process) — stored in the DB,
        not in memory."""
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        store.record_trade(Trade("AAPL", "buy", 1.0, 100.0))
        store.close()
        out = tmp_path / "ledger.csv"
        runner.invoke(cli.app, ["export-ledger", str(out)])

        st2 = Store(settings.db_path)
        st2.record_trade(Trade("MSFT", "buy", 2.0, 200.0))
        st2.close()
        runner.invoke(cli.app, ["export-ledger", str(out)])

        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert [r["symbol"] for r in rows] == ["AAPL", "MSFT"]
