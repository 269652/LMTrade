"""Tests for the optional Trade Republic derivatives adapter
(brokers/tr_derivatives.py) and its engine integration: when TR auth is
configured AND the client is reachable, paper trading uses TR-realistic
knockout instruments (real ISINs when the live client provides them); when
not, the engine falls back to the existing synthetic Black-Scholes options —
unchanged behavior. All offline: the live pytr client is faked. Written
before implementation per strict TDD."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.brokers.tr_derivatives import (
    FakeTRDerivatives,
    PytrDerivatives,
    TRDerivativeQuote,
    build_tr_derivatives,
)
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


def buy_signal_history(base: float, n: int = 60) -> list[float]:
    return [base + i * 0.6 + 3 * math.sin(i * 0.7) for i in range(n)]


class ScriptedMarket:
    def __init__(self, quotes: dict[str, Quote]):
        self._quotes = quotes

    def quote(self, symbol: str) -> Quote:
        return self._quotes.get(symbol, Quote(symbol, 100.0, [100.0] * 60, "yahoo"))


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.risk.min_confidence = 0.5
    s.options.enabled = True
    s.learning.enabled = False
    s.tr.use_derivatives = True
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


class TestFactory:
    def test_no_credentials_returns_none(self, settings, monkeypatch):
        monkeypatch.delenv("TR_PHONE", raising=False)
        monkeypatch.delenv("TR_PIN", raising=False)
        assert build_tr_derivatives(settings) is None

    def test_disabled_in_config_returns_none(self, settings, monkeypatch):
        monkeypatch.setenv("TR_PHONE", "+491234")
        monkeypatch.setenv("TR_PIN", "1234")
        settings.tr.use_derivatives = False
        assert build_tr_derivatives(settings) is None


class FakeAsyncTRApi:
    """Minimal async fake mirroring the actual pytr>=0.4 `TradeRepublicApi`
    surface PytrDerivatives relies on (resume_websession/search/
    search_derivative/recv/unsubscribe) — verified against the real
    installed pytr 0.4.9 source, not guessed. Lets PytrDerivatives' own glue
    logic be tested without the optional pytr dependency installed and
    without ever touching TR's real (websocket) API."""

    def __init__(self, resume_ok=True, search_results=None, derivative_results=None):
        self.resume_ok = resume_ok
        self._search_results = (
            search_results if search_results is not None else [{"isin": "US0378331005"}])
        self._derivative_results = derivative_results if derivative_results is not None else []
        self.calls: list[tuple] = []

    def resume_websession(self) -> bool:
        return self.resume_ok

    async def search(self, query, asset_type="stock"):
        self.calls.append(("search", query, asset_type))
        return "sub-search"

    async def search_derivative(self, isin, product_type):
        self.calls.append(("search_derivative", isin, product_type))
        return "sub-deriv"

    async def recv(self):
        kind = self.calls[-1][0]
        if kind == "search":
            return ("sub-search", {}, {"results": self._search_results})
        return ("sub-deriv", {}, {"results": self._derivative_results})

    async def unsubscribe(self, sub_id):
        self.calls.append(("unsubscribe", sub_id))


class TestPytrLoginFlow:
    """The installed pytr>=0.4 TradeRepublicApi has no `.login()` method —
    confirmed by inspecting the actual installed package. Non-interactive use
    (from inside the engine loop) must resume a cached session via
    resume_websession() and degrade — never block on interactive 2FA input —
    when there isn't one to resume."""

    def test_resumes_cached_session_without_interactive_login(self):
        api = FakeAsyncTRApi(resume_ok=True)
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.available() is True

    def test_unresumable_session_degrades_without_blocking(self):
        api = FakeAsyncTRApi(resume_ok=False)
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.available() is False
        assert client.search("AAPL", "buy") == []

    def test_factory_exception_degrades_gracefully(self):
        def boom():
            raise RuntimeError("no pytr installed")

        client = PytrDerivatives("+491234", "1234", api_factory=boom)
        assert client.available() is False

    def test_login_result_is_cached_not_repeated(self):
        calls = []

        def factory():
            calls.append(1)
            return FakeAsyncTRApi(resume_ok=True)

        client = PytrDerivatives("+491234", "1234", api_factory=factory)
        client.available()
        client.available()
        assert len(calls) == 1


class TestPytrSearchGlue:
    """search() must resolve the underlying ticker to an ISIN first (TR's
    search_derivative takes an ISIN, not a ticker) then query derivatives —
    the previous code called a nonexistent `derivative_search` method with
    the wrong signature entirely."""

    def test_resolves_isin_then_queries_derivatives(self):
        api = FakeAsyncTRApi(
            search_results=[{"isin": "US0378331005"}],
            derivative_results=[{
                "isin": "DE000ABC123", "strike": 180.0, "barrier": 180.0,
                "ratio": 10.0, "ask": 2.5, "leverage": 5.0,
                "issuerDisplayName": "TestBank",
            }])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        quotes = client.search("AAPL", "buy")
        assert len(quotes) == 1
        assert quotes[0].isin == "DE000ABC123"
        assert quotes[0].kind == "ko_call"
        assert quotes[0].leverage == pytest.approx(5.0)
        assert ("search", "AAPL", "stock") in api.calls
        assert ("search_derivative", "US0378331005", "knockout") in api.calls

    def test_sell_direction_maps_to_ko_put(self):
        api = FakeAsyncTRApi(
            derivative_results=[{"isin": "DE1", "strike": 100.0, "ask": 1.0, "leverage": 4.0}])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        quotes = client.search("AAPL", "sell")
        assert quotes[0].kind == "ko_put"

    def test_no_isin_match_returns_empty(self):
        api = FakeAsyncTRApi(search_results=[])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.search("UNKNOWN", "buy") == []
        assert not any(c[0] == "search_derivative" for c in api.calls)

    def test_malformed_derivative_items_are_skipped_not_crashed(self):
        api = FakeAsyncTRApi(
            search_results=[{"isin": "US0378331005"}],
            derivative_results=[
                {"isin": "DE1"},  # missing strike -> skipped
                {"isin": "DE2", "strike": 50.0, "ask": 1.0, "leverage": 3.0},
            ])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        quotes = client.search("AAPL", "buy")
        assert len(quotes) == 1
        assert quotes[0].isin == "DE2"

    def test_unavailable_client_short_circuits_search(self):
        api = FakeAsyncTRApi(resume_ok=False)
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.search("AAPL", "buy") == []
        assert api.calls == []


class TestFakeClient:
    def test_search_returns_ko_quotes_sorted_by_leverage_fit(self):
        fake = FakeTRDerivatives(catalog={
            "AAPL": [
                TRDerivativeQuote(isin="DE000A1", underlying="AAPL",
                                  kind="ko_call", strike=80.0, barrier=80.0,
                                  ratio=10.0, price=2.05, leverage=5.0,
                                  issuer="FakeBank"),
                TRDerivativeQuote(isin="DE000A2", underlying="AAPL",
                                  kind="ko_call", strike=95.0, barrier=95.0,
                                  ratio=10.0, price=0.55, leverage=20.0,
                                  issuer="FakeBank"),
            ]})
        best = fake.find_knockout("AAPL", direction="buy", spot=100.0,
                                  target_leverage=6.0)
        assert best.isin == "DE000A1"     # 5x is closer to 6x than 20x
        assert fake.available() is True

    def test_no_instrument_for_symbol_returns_none(self):
        fake = FakeTRDerivatives(catalog={})
        assert fake.find_knockout("MSFT", "buy", 100.0, 5.0) is None


class TestEngineIntegration:
    def _engine(self, settings, store, client):
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 100.0, buy_signal_history(80.0), "yahoo"),
        })
        eng = Engine(settings, store, broker, market=market,
                     tr_derivatives=client)
        eng._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        return eng

    def test_with_tr_client_opens_knockout_position(self, settings, store):
        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000KO1", underlying="AAPL",
                                       kind="ko_call", strike=80.0, barrier=80.0,
                                       ratio=10.0, price=2.05, leverage=5.0,
                                       issuer="FakeBank")]})
        engine = self._engine(settings, store, client)
        engine.run_cycle()
        opts = store.open_options()
        assert opts, "expected a knockout position"
        o = opts[0]
        assert o["instrument_type"] == "knockout"
        assert o["isin"] == "DE000KO1"
        assert o["barrier"] == pytest.approx(80.0)
        assert o["tp_premium"] and o["sl_premium"]   # stops still placed

    def test_without_tr_client_falls_back_to_bs_options(self, settings, store):
        engine = self._engine(settings, store, client=None)
        engine.run_cycle()
        opts = store.open_options()
        assert opts
        assert opts[0]["instrument_type"] == "option"   # legacy path unchanged
        assert opts[0]["isin"] is None

    def test_knockout_position_knocked_out_when_barrier_touched(self, settings, store):
        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000KO1", underlying="AAPL",
                                       kind="ko_call", strike=95.0, barrier=95.0,
                                       ratio=10.0, price=0.55, leverage=20.0,
                                       issuer="FakeBank")]})
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        market = ScriptedMarket({
            # Spot crashes through the 95 barrier
            "AAPL": Quote("AAPL", 94.0, [100.0] * 60, "yahoo"),
        })
        # Plant an open KO position directly, then run a cycle to manage it.
        store.open_option("AAPL", "ko_call", strike=95.0, expiry_ts=4e12,
                          iv=0.0, contracts=10.0, entry_premium=0.55,
                          genome_id=None, tp_premium=0.85, sl_premium=0.30,
                          instrument_type="knockout", barrier=95.0,
                          ratio=10.0, isin="DE000KO1")
        engine = Engine(settings, store, broker, market=market,
                        tr_derivatives=client)
        engine.run_cycle()
        closed = store.closed_options()
        assert closed, "barrier touch must close the position"
        assert closed[0]["exit_premium"] == pytest.approx(0.0, abs=1e-9)
        # Total loss of premium (that's what a knockout IS)
        assert closed[0]["pnl"] < 0

    def test_min_hold_does_not_block_knockout(self, settings, store):
        """A knockout is an involuntary event — min_hold_hours must never
        keep a dead instrument on the books."""
        settings.options.min_hold_hours = 1000.0
        client = FakeTRDerivatives(catalog={})
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        market = ScriptedMarket({"AAPL": Quote("AAPL", 94.0, [100.0] * 60, "yahoo")})
        store.open_option("AAPL", "ko_call", strike=95.0, expiry_ts=4e12,
                          iv=0.0, contracts=10.0, entry_premium=0.55,
                          genome_id=None, tp_premium=0.85, sl_premium=0.30,
                          instrument_type="knockout", barrier=95.0,
                          ratio=10.0, isin="DE000KO1")
        engine = Engine(settings, store, broker, market=market,
                        tr_derivatives=client)
        engine.run_cycle()
        assert store.closed_options(), "knockout fires regardless of min_hold"
