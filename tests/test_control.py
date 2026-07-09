"""Control plane for the paper/live toggle and the live-execution arming
guard. Persisted so the engine (which executes) and the web dashboard (which
toggles) agree across processes. Written before implementation per strict
TDD. Nothing here places a real order — this is only the gate state."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.core.control import LOW_BALANCE_EUR, ControlState


@pytest.fixture()
def path(tmp_path: Path) -> Path:
    return tmp_path / "control.json"


class TestDefaultsAndPersistence:
    def test_defaults_to_paper_disarmed(self, path):
        c = ControlState.load(path)
        assert c.mode == "paper"
        assert c.armed is False

    def test_roundtrips_via_disk(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True, net_worth=500.0)
        again = ControlState.load(path)
        assert again.mode == "live"
        assert again.armed is True

    def test_missing_file_is_safe_default(self, path):
        assert not path.exists()
        assert ControlState.load(path).mode == "paper"

    def test_corrupt_file_falls_back_to_paper(self, path):
        path.write_text("{ not json")
        c = ControlState.load(path)
        assert c.mode == "paper" and c.armed is False


class TestModeSwitch:
    def test_switch_to_live_does_not_auto_arm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.mode == "live"
        assert c.armed is False        # live view, still simulated until armed

    def test_switch_back_to_paper_disarms(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True, net_worth=500.0)
        assert c.armed is True
        c.set_mode("paper")
        assert c.armed is False        # leaving live must never leave it armed

    def test_invalid_mode_rejected(self, path):
        c = ControlState.load(path)
        with pytest.raises(ValueError):
            c.set_mode("bogus")


class TestArmingGuard:
    def test_cannot_arm_in_paper_mode(self, path):
        c = ControlState.load(path)
        assert c.arm(confirm=True, net_worth=500.0) is False
        assert c.armed is False

    def test_arm_requires_confirm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.arm(confirm=False, net_worth=500.0) is False
        assert c.armed is False

    def test_arm_with_confirm_above_threshold(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.arm(confirm=True, net_worth=500.0) is True
        assert c.armed is True

    def test_low_net_worth_needs_double_confirm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        # Under the fee-drag threshold: single confirm is not enough.
        assert c.arm(confirm=True, net_worth=LOW_BALANCE_EUR - 1) is False
        assert c.armed is False
        assert c.arm(confirm=True, double_confirm=True,
                     net_worth=LOW_BALANCE_EUR - 1) is True
        assert c.armed is True

    def test_at_threshold_single_confirm_ok(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.arm(confirm=True, net_worth=LOW_BALANCE_EUR) is True

    def test_disarm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True, net_worth=500.0)
        c.disarm()
        assert c.armed is False

    def test_low_balance_flag(self, path):
        c = ControlState.load(path)
        assert c.is_low_balance(50.0) is True
        assert c.is_low_balance(150.0) is False
