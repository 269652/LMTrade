"""Regression test: with gpu_usd_per_hour=0 (no GPU rented locally),
runway_hours is literally float('inf'). Starlette's JSONResponse uses
allow_nan=False (RFC 8259 JSON has no Infinity), so any endpoint returning
EconomicsSnapshot.as_dict() verbatim crashes with "Out of range float
values are not JSON compliant: inf" on a live run. Written before the fix
per strict TDD."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lmtrade.config import Settings
from lmtrade.core.state import Store
from lmtrade.economics.cost_accounting import CostAccountant


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                 economics={"gpu_usd_per_hour": 0.0})
    s.data_dir = tmp_path
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    st.set_meta("starting_cash", settings.budget)
    yield st
    st.close()


class TestInfiniteRunwaySerialization:
    def test_snapshot_runway_is_still_infinite(self, settings, store):
        # The raw dataclass keeps the real float — CLI status formatting
        # (f"{runway:.1f} h") handles inf fine and this must stay accurate.
        acc = CostAccountant(settings, store)
        snap = acc.snapshot(cash_eur=10.0, positions_value_eur=0.0)
        assert math.isinf(snap.runway_hours)

    def test_as_dict_runway_is_json_safe(self, settings, store):
        acc = CostAccountant(settings, store)
        snap = acc.snapshot(cash_eur=10.0, positions_value_eur=0.0)
        data = snap.as_dict()
        assert data["runway_hours"] is None
        # Must not raise — this is exactly what Starlette's JSONResponse does.
        json.dumps(data, allow_nan=False)

    def test_finite_runway_unaffected(self, tmp_path):
        s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                     economics={"gpu_usd_per_hour": 0.2})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.set_meta("starting_cash", s.budget)
        acc = CostAccountant(s, store)
        snap = acc.snapshot(cash_eur=10.0, positions_value_eur=0.0)
        data = snap.as_dict()
        assert isinstance(data["runway_hours"], float)
        assert data["runway_hours"] > 0
        store.close()
