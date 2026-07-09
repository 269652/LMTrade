"""Runtime control plane for the paper/live toggle and the live-execution
arming guard.

Persisted to a small JSON file so the engine (which executes trades) and the
web dashboard (which flips the toggle) agree even when they run in separate
processes. This module only holds the GATE STATE — it never places an order.
Real execution reads `mode == "live" and armed` before doing anything with
real money; see brokers/trade_republic.py and the engine's execution routing.

Two deliberate frictions, both here so they can't be bypassed by a stray UI
bug:
  1. Arming is only possible in live mode and requires an explicit confirm.
  2. When net worth is below LOW_BALANCE_EUR — where Trade Republic's flat
     ~1 EUR fee is more than 1% of the account, a severe drag — arming needs
     a SECOND ("double") confirmation.
Switching back to paper always disarms.
"""
from __future__ import annotations

import json
from pathlib import Path

# Below this net worth the flat ~1 EUR TR fee exceeds 1% of the account, so
# live execution demands the extra double-confirm.
LOW_BALANCE_EUR = 100.0

_VALID_MODES = ("paper", "live")


class ControlState:
    def __init__(self, path: Path, mode: str = "paper", armed: bool = False):
        self.path = path
        self.mode = mode if mode in _VALID_MODES else "paper"
        self.armed = bool(armed) and self.mode == "live"

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "ControlState":
        try:
            data = json.loads(Path(path).read_text())
            return cls(path, mode=data.get("mode", "paper"),
                       armed=bool(data.get("armed", False)))
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            return cls(path)   # safe default: paper, disarmed

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"mode": self.mode, "armed": self.armed}))

    def as_dict(self) -> dict:
        return {"mode": self.mode, "armed": self.armed}

    # -- transitions ---------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"invalid mode {mode!r}; expected one of {_VALID_MODES}")
        self.mode = mode
        if mode != "live":
            self.armed = False        # leaving live must never stay armed
        self._save()

    def is_low_balance(self, net_worth: float | None) -> bool:
        return net_worth is not None and net_worth < LOW_BALANCE_EUR

    def arm(self, confirm: bool, double_confirm: bool = False,
            net_worth: float | None = None) -> bool:
        """Attempt to arm live execution. Returns True only if it actually
        armed. Requires live mode + confirm, and — when net worth is below
        LOW_BALANCE_EUR — a second double_confirm as well."""
        if self.mode != "live" or not confirm:
            return False
        if self.is_low_balance(net_worth) and not double_confirm:
            return False
        self.armed = True
        self._save()
        return True

    def disarm(self) -> None:
        self.armed = False
        self._save()

    @property
    def live_armed(self) -> bool:
        """True only when real orders should actually be placed."""
        return self.mode == "live" and self.armed
