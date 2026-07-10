"""Live-book <-> Trade Republic reconciliation and the shared TR session.

- One pytr session per phone number: the live broker and the derivatives
  client previously opened SEPARATE websocket connections with the same
  cookie and TR rejected the second one with HTTP 401 (observed live:
  "Connected." followed immediately by a 401 on the next connection).
- sync_tr_portfolio(): on live startup the live book is reconciled against
  the REAL TR portfolio — phantom local rows (not present at TR) are
  deleted, real TR positions missing locally are imported, and the cash
  meta is refreshed — so net worth is exactly real cash + real positions.

Written before implementation per strict TDD; fully offline via fakes."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.core.state import Position, Store
from lmtrade.core.tr_sync import sync_tr_portfolio


@pytest.fixture()
def store(tmp_path: Path):
    s = Store(tmp_path / "live.db")
    yield s
    s.close()


class FakeTR:
    def __init__(self, cash=250.0, portfolio=None):
        self._cash = cash
        self._portfolio = portfolio

    def account_cash(self):
        return self._cash

    def portfolio(self):
        return self._portfolio


class TestSyncTrPortfolio:
    def test_cash_meta_refreshed(self, store):
        sync_tr_portfolio(store, FakeTR(cash=250.0, portfolio=[]))
        assert store.get_meta("tr_account_cash") == pytest.approx(250.0)

    def test_phantom_option_rows_deleted(self, store):
        # A local "live" option whose ISIN is NOT in the real TR portfolio —
        # a phantom from an earlier bug. Must be removed, not valued.
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=5.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0,
                          instrument_type="knockout", barrier=80.0, ratio=10.0,
                          isin="DE000PHANTOM")
        sync_tr_portfolio(store, FakeTR(portfolio=[]))
        assert store.open_options() == []

    def test_real_option_rows_kept(self, store):
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=5.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0,
                          instrument_type="knockout", barrier=80.0, ratio=10.0,
                          isin="DE000REAL01")
        sync_tr_portfolio(store, FakeTR(portfolio=[
            {"isin": "DE000REAL01", "size": 5.0, "avg_price": 2.0}]))
        assert len(store.open_options()) == 1

    def test_untracked_tr_position_imported(self, store):
        sync_tr_portfolio(store, FakeTR(portfolio=[
            {"isin": "DE000NEW001", "size": 3.0, "avg_price": 12.5}]))
        pos = store.position("DE000NEW001")
        assert pos is not None
        assert pos.qty == pytest.approx(3.0)
        assert pos.avg_price == pytest.approx(12.5)

    def test_phantom_equity_rows_deleted(self, store):
        store.upsert_position(Position("US0000PHANT0", 2.0, 50.0, 0.0))
        sync_tr_portfolio(store, FakeTR(portfolio=[]))
        assert store.position("US0000PHANT0") is None

    def test_portfolio_unavailable_is_a_noop(self, store):
        # Session down (portfolio None): never delete anything on no data.
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=5.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0,
                          instrument_type="knockout", barrier=80.0, ratio=10.0,
                          isin="DE000REAL01")
        sync_tr_portfolio(store, FakeTR(cash=None, portfolio=None))
        assert len(store.open_options()) == 1

    def test_pending_row_confirmed_in_real_portfolio_flips_to_open(self, store):
        # Crash-recovery case: the engine created a pending row, placed the
        # order, then the process died before mark_option_open() ran. TR's
        # own portfolio is authoritative — if it's really there, promote it.
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=5.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0,
                          instrument_type="knockout", barrier=80.0, ratio=10.0,
                          isin="DE000CRASHED", status="pending")
        sync_tr_portfolio(store, FakeTR(portfolio=[
            {"isin": "DE000CRASHED", "size": 5.0, "avg_price": 2.0}]))
        assert store.pending_options() == []
        opts = store.open_options()
        assert len(opts) == 1
        assert opts[0]["isin"] == "DE000CRASHED"
        assert opts[0]["status"] == "open"

    def test_pending_row_not_in_real_portfolio_is_deleted(self, store):
        # Crash-recovery case: the order attempt never actually went through
        # (or was rejected) before the process died — TR doesn't have it, so
        # the pending row is a phantom, same as a stale open row would be.
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=5.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0,
                          instrument_type="knockout", barrier=80.0, ratio=10.0,
                          isin="DE000NEVERWENT", status="pending")
        sync_tr_portfolio(store, FakeTR(portfolio=[]))
        assert store.pending_options() == []
        assert store.open_options() == []

    def test_pending_row_unchanged_when_portfolio_unavailable(self, store):
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=5.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0,
                          instrument_type="knockout", barrier=80.0, ratio=10.0,
                          isin="DE000REAL01", status="pending")
        sync_tr_portfolio(store, FakeTR(cash=None, portfolio=None))
        assert len(store.pending_options()) == 1


class TestSharedSession:
    def test_broker_and_derivatives_share_one_api(self, monkeypatch, tmp_path):
        from lmtrade.brokers import tr_derivatives as td

        created = []

        class FakeApi:
            def __init__(self):
                created.append(self)

        monkeypatch.setattr(td, "_new_pytr_api", lambda phone, pin: FakeApi())
        td._SHARED_APIS.clear()
        a = td.get_shared_api("+49123", "1111")
        b = td.get_shared_api("+49123", "1111")
        assert a is b                      # ONE session, not two
        assert len(created) == 1
        td.drop_shared_api("+49123")
        c = td.get_shared_api("+49123", "1111")
        assert c is not a                  # dropped -> fresh session
