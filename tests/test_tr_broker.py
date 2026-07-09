"""Tests for the live Trade Republic execution broker
(brokers/trade_republic.py). Fully offline: the pytr session is always an
injected fake, so no real order is ever placed by the suite.

Real order placement is defense-in-depth gated: the broker refuses unless it
was constructed `armed=True` (the engine passes control.live_armed), on top of
the control-plane gate. Written before implementation per strict TDD."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.config import Settings
from lmtrade.core.state import Store


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="live", budget=100.0, universe=["AAPL"])
    s.data_dir = tmp_path
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.live_db_path)
    yield st
    st.close()


class FakeOrderApi:
    """Fake pytr session: order responses come from a queue (one per
    market_order call) so warning->acknowledge->confirm flows are testable.
    recv() can also be fed cross-talk frames for OTHER subscriptions first,
    which the broker must skip past (shared-websocket reality)."""

    def __init__(self, resume_ok=True, order_payloads=None, cash_payload=None,
                 crosstalk=None):
        self.resume_ok = resume_ok
        self._order_payloads = (list(order_payloads) if order_payloads is not None
                                else [{"orderId": "x1"}])
        self._cash_payload = (cash_payload if cash_payload is not None
                              else [{"currencyId": "EUR", "amount": 88.0}])
        self._crosstalk = list(crosstalk or [])   # (sub_id, payload) frames
        self.orders: list[dict] = []
        self.calls: list[str] = []
        self._order_n = 0

    def resume_websession(self) -> bool:
        return self.resume_ok

    async def market_order(self, isin, exchange, order_type, size, expiry,
                           sell_fractions, expiry_date=None, warnings_shown=None):
        self.orders.append({"isin": isin, "exchange": exchange, "side": order_type,
                            "size": size, "expiry": expiry,
                            "warnings_shown": warnings_shown})
        self.calls.append("market_order")
        self._order_n += 1
        return f"sub-order-{self._order_n}"

    async def cash(self):
        self.calls.append("cash")
        return "sub-cash"

    async def recv(self):
        if self._crosstalk:
            return (*self._crosstalk.pop(0), )
        if self.calls[-1] == "cash":
            return ("sub-cash", {}, self._cash_payload)
        payload = (self._order_payloads.pop(0) if self._order_payloads
                   else {"orderId": "fallback"})
        return (f"sub-order-{self._order_n}", {}, payload)

    async def unsubscribe(self, sub_id):
        pass


def make_broker(store, settings, *, armed, api, monkeypatch):
    monkeypatch.setenv("TR_PHONE", "+491234")
    monkeypatch.setenv("TR_PIN", "1234")
    from lmtrade.brokers.trade_republic import TradeRepublicBroker
    return TradeRepublicBroker(store, settings, armed=armed, api_factory=lambda: api)


class TestConstruction:
    def test_requires_credentials(self, store, settings, monkeypatch):
        monkeypatch.delenv("TR_PHONE", raising=False)
        monkeypatch.delenv("TR_PIN", raising=False)
        from lmtrade.brokers.trade_republic import TradeRepublicBroker
        with pytest.raises(RuntimeError):
            TradeRepublicBroker(store, settings, armed=False, api_factory=lambda: None)

    def test_mode_is_live(self, store, settings, monkeypatch):
        b = make_broker(store, settings, armed=False, api=FakeOrderApi(), monkeypatch=monkeypatch)
        assert b.mode == "live"


class TestOrderGating:
    def test_unarmed_broker_refuses_to_place(self, store, settings, monkeypatch):
        api = FakeOrderApi()
        b = make_broker(store, settings, armed=False, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 3.0)
        assert res.ok is False
        assert "arm" in res.message.lower()
        assert api.orders == []          # nothing was sent to TR

    def test_armed_broker_places_real_order(self, store, settings, monkeypatch):
        api = FakeOrderApi()
        b = make_broker(store, settings, armed=True, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 3.0, exchange="LSX")
        assert res.ok is True
        assert len(api.orders) == 1
        assert api.orders[0]["isin"] == "DE000KO1"
        assert api.orders[0]["side"] == "buy"
        assert api.orders[0]["size"] == 3.0

    def test_tr_error_payload_is_a_failed_order(self, store, settings, monkeypatch):
        api = FakeOrderApi(order_payloads=[{"errors": [{"errorCode": "TOO_SMALL"}]}])
        b = make_broker(store, settings, armed=True, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 0.001)
        assert res.ok is False
        assert "TOO_SMALL" in res.message

    def test_unresumable_session_fails_safe(self, store, settings, monkeypatch):
        api = FakeOrderApi(resume_ok=False)
        b = make_broker(store, settings, armed=True, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 3.0)
        assert res.ok is False
        assert api.orders == []


class TestOrderConfirmation:
    """A real order is only 'placed' when TR POSITIVELY confirms it (an order
    id). Live incident: TR responded without an error (a warnings-only
    acknowledgment), the broker declared success, the dashboard recorded a
    position — and the TR app showed nothing. 'No error' is NOT 'executed'."""

    def test_ambiguous_payload_is_not_a_fill(self, store, settings, monkeypatch):
        api = FakeOrderApi(order_payloads=[{"something": "else"}])
        b = make_broker(store, settings, armed=True, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 3.0)
        assert res.ok is False
        assert "unconfirmed" in res.message.lower()

    def test_warning_is_acknowledged_once_then_confirmed(self, store, settings,
                                                         monkeypatch):
        api = FakeOrderApi(order_payloads=[
            {"warnings": [{"type": "costWarning"}]},   # TR wants an ack
            {"orderId": "real-1"},                      # confirmed on resubmit
        ])
        b = make_broker(store, settings, armed=True, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 3.0)
        assert res.ok is True
        assert len(api.orders) == 2
        assert api.orders[0]["warnings_shown"] in (None, [])
        assert api.orders[1]["warnings_shown"] == ["costWarning"]

    def test_warning_then_still_unconfirmed_fails(self, store, settings, monkeypatch):
        api = FakeOrderApi(order_payloads=[
            {"warnings": [{"type": "costWarning"}]},
            {"warnings": [{"type": "costWarning"}]},   # TR still not confirming
        ])
        b = make_broker(store, settings, armed=True, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 3.0)
        assert res.ok is False
        assert len(api.orders) == 2                    # acknowledged once, no loop

    def test_crosstalk_frames_are_skipped(self, store, settings, monkeypatch):
        # Shared websocket: a frame for ANOTHER subscription arrives first;
        # the broker must keep receiving until OUR subscription answers.
        api = FakeOrderApi(order_payloads=[{"orderId": "x9"}],
                           crosstalk=[("sub-ticker", {}, {"bid": 1.0})])
        b = make_broker(store, settings, armed=True, api=api, monkeypatch=monkeypatch)
        res = b.place_order("DE000KO1", "buy", 3.0)
        assert res.ok is True


class TestLiveCash:
    def test_account_cash_from_tr(self, store, settings, monkeypatch):
        api = FakeOrderApi(cash_payload=[{"currencyId": "EUR", "amount": 88.0}])
        b = make_broker(store, settings, armed=False, api=api, monkeypatch=monkeypatch)
        assert b.cash() == pytest.approx(88.0)
