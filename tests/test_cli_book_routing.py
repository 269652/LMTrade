"""Book routing for the paper/live toggle: the LIVE database must contain
ONLY real fills. Unarmed live is a real-account VIEW — the simulation keeps
trading the paper book; only live+armed trades (real orders) go to the live
book. Written before the fix per strict TDD."""
from __future__ import annotations

from pathlib import Path

import pytest

import lmtrade.cli as cli
from lmtrade.config import Settings
from lmtrade.core.control import ControlState


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    return s


def control_for(settings, mode, armed=False) -> ControlState:
    c = ControlState.load(settings.control_path)
    c.set_mode(mode)
    if armed:
        c.arm(confirm=True)
        c.set_double_armed(confirm=True)
    return c


class TestBookSelection:
    def test_paper_mode_uses_paper_book(self, settings):
        engine, store = cli._build_engine_for_control(
            settings, control_for(settings, "paper"))
        assert store.db_path == settings.db_path
        assert engine.broker.mode == "paper"
        store.close()

    def test_unarmed_live_keeps_trading_the_paper_book(self, settings):
        # Live view, simulated execution: the live ledger must NOT receive
        # paper fills, and the paper simulation must not be interrupted.
        engine, store = cli._build_engine_for_control(
            settings, control_for(settings, "live", armed=False))
        assert store.db_path == settings.db_path       # paper book
        assert engine.broker.mode == "paper"
        store.close()

    def test_armed_live_uses_live_book_and_live_broker(self, settings, monkeypatch):
        monkeypatch.setenv("TR_PHONE", "+49123")
        monkeypatch.setenv("TR_PIN", "1234")
        engine, store = cli._build_engine_for_control(
            settings, control_for(settings, "live", armed=True))
        assert store.db_path == settings.live_db_path  # live book
        assert engine.broker.mode == "live"
        assert engine.broker.armed is True
        store.close()


class TestTrSyncTrigger:
    """Switching TO live mode (whether or not armed) must reconcile the live
    book against the real TR account — both live (dashboard toggle flip,
    which reruns _build_engine_for_control via the rebuild loop) and on a
    fresh startup already in live mode (e.g. after `lmtrade reset` + restart,
    since control.json is a separate file the reset never touches)."""

    def _tr_stub(self, monkeypatch, portfolio=None, cash=100.0):
        calls = []

        class FakeTR:
            def available(self):
                return True

            def portfolio(self):
                calls.append(1)
                return portfolio if portfolio is not None else []

            def account_cash(self):
                return cash

        # _build_engine_for_control does `from .brokers.tr_derivatives import
        # build_tr_derivatives` INSIDE the function body, re-resolved on every
        # call — patch the source attribute, not a (nonexistent) cli.* one.
        monkeypatch.setattr(
            "lmtrade.brokers.tr_derivatives.build_tr_derivatives",
            lambda settings: FakeTR())
        return calls

    def test_switching_to_unarmed_live_triggers_sync(self, settings, monkeypatch):
        calls = self._tr_stub(monkeypatch, portfolio=[
            {"isin": "DE000NEW1", "size": 2.0, "avg_price": 5.0}])
        engine, store = cli._build_engine_for_control(
            settings, control_for(settings, "live", armed=False))
        assert calls, "sync_tr_portfolio must run when switching to live (even unarmed)"
        # Imported into the LIVE book (fallback_store here, since unarmed
        # trades paper) — not the paper book the engine is actively trading.
        assert engine.fallback_store.position("DE000NEW1") is not None
        store.close()

    def test_switching_to_armed_live_triggers_sync(self, settings, monkeypatch):
        monkeypatch.setenv("TR_PHONE", "+49123")
        monkeypatch.setenv("TR_PIN", "1234")
        calls = self._tr_stub(monkeypatch, portfolio=[
            {"isin": "DE000NEW2", "size": 1.0, "avg_price": 9.0}])
        engine, store = cli._build_engine_for_control(
            settings, control_for(settings, "live", armed=True))
        assert calls
        assert store.position("DE000NEW2") is not None   # imported into the active live book
        store.close()

    def test_paper_mode_does_not_sync(self, settings, monkeypatch):
        calls = self._tr_stub(monkeypatch, portfolio=[
            {"isin": "DE000SHOULDNT", "size": 1.0, "avg_price": 1.0}])
        engine, store = cli._build_engine_for_control(
            settings, control_for(settings, "paper"))
        assert not calls, "paper mode must never touch the real TR account"
        store.close()

    def test_sync_after_reset_and_restart_restores_positions(self, settings, monkeypatch):
        # Simulate: bot was in live mode, user ran `lmtrade reset` (wipes the
        # DBs but NOT control.json), then restarted. The very first engine
        # build must re-import real TR positions with no manual step.
        control_for(settings, "live", armed=False)   # persists mode to control.json
        calls = self._tr_stub(monkeypatch, portfolio=[
            {"isin": "DE000RESTORED", "size": 3.0, "avg_price": 4.0}])
        from lmtrade.core.control import ControlState
        fresh_control = ControlState.load(settings.control_path)   # simulates a new process
        engine, store = cli._build_engine_for_control(settings, fresh_control)
        assert calls
        assert engine.fallback_store.position("DE000RESTORED") is not None
        store.close()
