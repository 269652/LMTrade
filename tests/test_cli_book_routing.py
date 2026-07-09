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
