"""Tests for the dashboard API: existing endpoints plus the new options,
strategy-leaderboard and benchmark/alpha endpoints. Written before the
endpoint implementation per strict TDD."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.web.app import create_app


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    return s


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    # Produce some real state first.
    store = Store(settings.db_path)
    engine = Engine(settings, store, PaperBroker(store, starting_cash=10.0, fee=0.1))
    engine.run_cycle()
    store.close()
    return TestClient(create_app(settings))


class TestExistingEndpoints:
    def test_summary_and_health(self, client):
        assert client.get("/healthz").json() == {"ok": True}
        s = client.get("/api/summary").json()
        assert s["mode"] == "paper"
        assert "economics" in s

    def test_trades_activity_logs_equity(self, client):
        for path in ("/api/trades", "/api/activity", "/api/logs", "/api/equity"):
            r = client.get(path)
            assert r.status_code == 200
            assert isinstance(r.json(), list)


class TestSummaryIncludesOptions:
    """The engine counts open OPTIONS toward max_positions, but the dashboard
    summary only counted equity positions — so a book full of options showed
    "0 positions", which read as broken. Summary must reflect the full book."""

    @pytest.fixture()
    def options_client(self, tmp_path: Path) -> TestClient:
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.open_option("AAPL", "call", strike=100.0, expiry_ts=4e12, iv=0.2,
                          contracts=3.0, entry_premium=2.5, genome_id=None,
                          tp_premium=3.75, sl_premium=1.5)
        store.open_option("NVDA", "put", strike=200.0, expiry_ts=4e12, iv=0.2,
                          contracts=1.0, entry_premium=4.0, genome_id=None,
                          tp_premium=6.0, sl_premium=2.4, instrument_type="knockout",
                          barrier=200.0, ratio=10.0, isin="DE000KO1")
        store.close()
        return TestClient(create_app(s))

    def test_num_positions_counts_open_options(self, options_client):
        s = options_client.get("/api/summary").json()
        assert s["num_positions"] == 2

    def test_positions_list_includes_options_with_isin(self, options_client):
        s = options_client.get("/api/summary").json()
        symbols = {p["symbol"] for p in s["positions"]}
        assert any("AAPL" in sym for sym in symbols)
        nvda = next(p for p in s["positions"] if "NVDA" in p["symbol"])
        assert nvda["isin"] == "DE000KO1"          # real TR knockout surfaced
        assert nvda["kind"] in ("put", "knockout") or "put" in nvda["symbol"].lower()

    def test_positions_include_live_pnl_from_persisted_marks(self, tmp_path):
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.open_option("AAPL", "call", strike=100.0, expiry_ts=4e12, iv=0.2,
                          contracts=2.0, entry_premium=2.5, genome_id=None,
                          tp_premium=3.75, sl_premium=1.5)
        opt_id = store.open_options()[0]["id"]
        store.set_meta("open_option_marks", {str(opt_id): {
            "mark_premium": 3.0, "value": 6.0, "unrealized_pnl": 1.0, "spot": 110.0}})
        store.close()
        client = TestClient(create_app(s))
        row = next(p for p in client.get("/api/summary").json()["positions"]
                   if "AAPL" in p["symbol"])
        assert row["unrealized_pnl"] == 1.0
        assert row["value"] == 6.0

    def test_positions_pnl_none_when_no_mark_yet(self, options_client):
        # Before the first cycle marks them, P&L is unknown (None), not a crash.
        row = next(p for p in options_client.get("/api/summary").json()["positions"]
                   if "AAPL" in p["symbol"])
        assert row["unrealized_pnl"] is None


class TestDashboardShell:
    def test_index_serves_cache_busted_appjs(self, client):
        html = client.get("/").text
        # A version query defeats the browser cache so a pulled app.js is
        # actually re-fetched instead of rendered against a new HTML shell.
        assert "app.js?v=" in html

    def test_leaderboard_available_for_strategies_tab(self, client):
        r = client.get("/api/leaderboard")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestAnalysisEndpoint:
    def test_analysis_returns_stored_market_analysis(self, tmp_path):
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.set_meta("market_analysis", {"valid_hours": 26, "ts": 1000.0,
                       "symbols": {"AAPL": {"bias": "bullish", "confidence": 0.6}}})
        store.close()
        data = TestClient(create_app(s)).get("/api/analysis").json()
        assert data["symbols"]["AAPL"]["bias"] == "bullish"

    def test_analysis_empty_when_none(self, client):
        assert client.get("/api/analysis").json() == {}


class TestSignalsEndpoint:
    """The Signals tab surfaces the strongest recent decisions and their
    cause (the per-provider signal breakdown recorded in decision activity)."""

    @pytest.fixture()
    def signals_client(self, tmp_path: Path) -> TestClient:
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.add_activity("decision", "AAPL: BUY conf 0.90", "AAPL", {
            "direction": "buy", "confidence": 0.9,
            "signals": [{"provider": "heuristic", "direction": "buy",
                         "confidence": 0.9, "rationale": "fast SMA above slow"}]})
        store.add_activity("decision", "MSFT: HOLD conf 0.00", "MSFT", {
            "direction": "hold", "confidence": 0.0, "signals": []})
        store.add_activity("decision", "NVDA: SELL conf 0.30", "NVDA", {
            "direction": "sell", "confidence": 0.3, "signals": []})
        store.close()
        return TestClient(create_app(s))

    def test_only_strong_directional_signals(self, signals_client):
        rows = signals_client.get("/api/signals?min_confidence=0.5").json()
        syms = {r["symbol"] for r in rows}
        assert "AAPL" in syms          # strong buy
        assert "MSFT" not in syms      # hold excluded
        assert "NVDA" not in syms      # below threshold

    def test_signal_includes_cause(self, signals_client):
        rows = signals_client.get("/api/signals?min_confidence=0.5").json()
        aapl = next(r for r in rows if r["symbol"] == "AAPL")
        assert aapl["direction"] == "buy"
        assert any("SMA" in s["rationale"] for s in aapl["signals"])


class TestRealizedPnl:
    """Closed options/knockouts carry realized P&L that had no home in the
    UI — only unrealized (open) P&L and equity trades were shown. The
    /api/realized endpoint surfaces closed positions plus running totals."""

    @pytest.fixture()
    def realized_client(self, tmp_path: Path) -> TestClient:
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        # A winner and a loser, closed.
        for sym, entry, exit_, pnl in [("AAPL", 2.0, 3.0, 2.0), ("NVDA", 4.0, 1.0, -6.0)]:
            oid = store.open_option(sym, "call", strike=100.0, expiry_ts=4e12, iv=0.2,
                                    contracts=2.0, entry_premium=entry, genome_id=None,
                                    tp_premium=entry * 1.5, sl_premium=entry * 0.6)
            store.close_option(oid, exit_premium=exit_, pnl=pnl)
        store.close()
        return TestClient(create_app(s))

    def test_realized_endpoint_lists_closed_with_totals(self, realized_client):
        data = realized_client.get("/api/realized").json()
        assert data["total_pnl"] == pytest.approx(-4.0)   # +2 and -6
        assert data["wins"] == 1
        assert data["losses"] == 1
        assert len(data["rows"]) == 2
        row = data["rows"][0]
        assert "symbol" in row and "pnl" in row and "exit_premium" in row

    def test_realized_empty_is_zeroed(self, client):
        data = client.get("/api/realized").json()
        assert data["total_pnl"] == 0
        assert data["rows"] == []


class TestControlPlane:
    """The paper/live toggle and armed live-execution guard, driven from the
    dashboard. Book-aware: reads/writes switch between the paper and live
    databases with the mode."""

    @pytest.fixture()
    def app_settings(self, tmp_path: Path) -> Settings:
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        # Distinct marker rows in each book so we can prove the view switches.
        paper = Store(s.db_path)
        paper.set_meta("starting_cash", 100.0)
        paper.close()
        live = Store(s.live_db_path)
        live.set_meta("starting_cash", 999.0)
        live.set_meta("economics", {"net_worth_eur": 500.0})
        live.close()
        return s

    @pytest.fixture()
    def cclient(self, app_settings) -> TestClient:
        return TestClient(create_app(app_settings))

    def test_defaults_to_paper(self, cclient):
        assert cclient.get("/api/control").json()["mode"] == "paper"

    def test_switch_mode_changes_the_book_view(self, cclient):
        assert cclient.get("/api/summary").json()["starting_cash"] == 100.0
        cclient.post("/api/control/mode", json={"mode": "live"})
        # Now the view serves the LIVE book.
        assert cclient.get("/api/summary").json()["starting_cash"] == 999.0

    def test_invalid_mode_rejected(self, cclient):
        assert cclient.post("/api/control/mode", json={"mode": "x"}).status_code == 400

    def test_cannot_arm_in_paper(self, cclient):
        r = cclient.post("/api/control/arm", json={"confirm": True}).json()
        assert r["armed"] is False

    def test_arm_in_live_above_threshold(self, cclient):
        cclient.post("/api/control/mode", json={"mode": "live"})   # live net worth 500
        r = cclient.post("/api/control/arm", json={"confirm": True}).json()
        assert r["armed"] is True
        assert cclient.get("/api/control").json()["live_armed"] is True

    def test_low_balance_second_guard_gates_execution(self, tmp_path):
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        live = Store(s.live_db_path)
        live.set_meta("tr_account_cash", 40.0)   # real balance under threshold
        live.close()
        c = TestClient(create_app(s))
        c.post("/api/control/mode", json={"mode": "live"})
        # First guard arms, but with a low balance it is NOT executing yet.
        armed = c.post("/api/control/arm", json={"confirm": True}).json()
        assert armed["armed"] is True
        assert armed["low_balance"] is True
        assert armed["executing"] is False
        # Second guard flips execution on.
        dbl = c.post("/api/control/double-arm", json={"confirm": True}).json()
        assert dbl["double_armed"] is True
        assert dbl["executing"] is True

    def test_double_disarm_stops_execution(self, tmp_path):
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        live = Store(s.live_db_path)
        live.set_meta("tr_account_cash", 40.0)
        live.close()
        c = TestClient(create_app(s))
        c.post("/api/control/mode", json={"mode": "live"})
        c.post("/api/control/arm", json={"confirm": True})
        c.post("/api/control/double-arm", json={"confirm": True})
        assert c.get("/api/control").json()["executing"] is True
        c.post("/api/control/double-disarm")
        assert c.get("/api/control").json()["executing"] is False

    def test_healthy_balance_executes_without_second_guard(self, cclient):
        cclient.post("/api/control/mode", json={"mode": "live"})   # net worth 500
        cclient.post("/api/control/arm", json={"confirm": True})
        assert cclient.get("/api/control").json()["executing"] is True

    def test_disarm(self, cclient):
        cclient.post("/api/control/mode", json={"mode": "live"})
        cclient.post("/api/control/arm", json={"confirm": True})
        cclient.post("/api/control/disarm")
        assert cclient.get("/api/control").json()["armed"] is False

    def test_switch_back_to_paper_disarms(self, cclient):
        cclient.post("/api/control/mode", json={"mode": "live"})
        cclient.post("/api/control/arm", json={"confirm": True})
        cclient.post("/api/control/mode", json={"mode": "paper"})
        assert cclient.get("/api/control").json()["armed"] is False


class TestLiveViewRealBalances:
    """In live mode the summary must reflect the REAL TR account — never
    fabricate the paper budget for an empty live book, and read the TR cash
    meta from whichever book the engine is writing it into."""

    @pytest.fixture()
    def s(self, tmp_path: Path) -> Settings:
        st = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                      data={"provider": "synthetic"})
        st.data_dir = tmp_path
        return st

    def _live(self, s) -> TestClient:
        c = TestClient(create_app(s))
        c.post("/api/control/mode", json={"mode": "live"})
        return c

    def test_empty_live_book_shows_no_fabricated_cash(self, s):
        data = self._live(s).get("/api/summary").json()
        assert data["cash"] is None          # not the paper budget 100
        assert data["tr_account_cash"] is None

    def test_tr_cash_read_from_live_book(self, s):
        live = Store(s.live_db_path)
        live.set_meta("tr_account_cash", 55.5)
        live.close()
        data = self._live(s).get("/api/summary").json()
        assert data["cash"] == pytest.approx(55.5)
        assert data["tr_account_cash"] == pytest.approx(55.5)

    def test_tr_cash_falls_back_to_paper_book_copy(self, s):
        # The engine writes tr_account_cash into whichever book it trades —
        # with unarmed live trading the paper book, the meta lands there.
        paper = Store(s.db_path)
        paper.set_meta("tr_account_cash", 77.7)
        paper.close()
        data = self._live(s).get("/api/summary").json()
        assert data["tr_account_cash"] == pytest.approx(77.7)

    def test_paper_mode_unaffected(self, s):
        paper = Store(s.db_path)
        paper.set_meta("cash", 42.0)
        paper.close()
        data = TestClient(create_app(s)).get("/api/summary").json()
        assert data["cash"] == pytest.approx(42.0)


class TestReserveInSummary:
    def test_summary_exposes_reserve(self, tmp_path):
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.add_reserve(12.5)
        store.close()
        data = TestClient(create_app(s)).get("/api/summary").json()
        assert data["reserve"] == pytest.approx(12.5)


class TestTrAccountCashInSummary:
    def test_summary_exposes_tr_account_cash(self, tmp_path):
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                     data={"provider": "synthetic"})
        s.data_dir = tmp_path
        store = Store(s.db_path)
        store.set_meta("tr_account_cash", 987.65)
        store.close()
        data = TestClient(create_app(s)).get("/api/summary").json()
        assert data["tr_account_cash"] == pytest.approx(987.65)

    def test_tr_account_cash_none_by_default(self, client):
        assert client.get("/api/summary").json()["tr_account_cash"] is None


class TestInfiniteRunwayJsonSafety:
    """gpu_usd_per_hour=0 (no GPU rented) makes runway_hours float('inf') —
    Starlette's JSONResponse (allow_nan=False) crashes on that unless it's
    converted before serialization. Regression test for a live crash."""

    @pytest.fixture()
    def zero_gpu_client(self, tmp_path: Path) -> TestClient:
        s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                     data={"provider": "synthetic"},
                     economics={"gpu_usd_per_hour": 0.0})
        s.model.stack = ["heuristic"]
        s.data_dir = tmp_path
        store = Store(s.db_path)
        engine = Engine(s, store, PaperBroker(store, starting_cash=10.0, fee=0.1))
        engine.run_cycle()
        store.close()
        return TestClient(create_app(s))

    def test_summary_is_json_safe(self, zero_gpu_client):
        r = zero_gpu_client.get("/api/summary")
        assert r.status_code == 200
        assert r.json()["economics"]["runway_hours"] is None

    def test_activity_is_json_safe(self, zero_gpu_client):
        r = zero_gpu_client.get("/api/activity")
        assert r.status_code == 200


class TestLegacyInfiniteDataJsonSafety:
    """Activity rows / meta written before the runway_hours-as-None fix (or
    by any future producer that bypasses EconomicsSnapshot.as_dict()) can
    still carry a raw float('inf') in their stored JSON — Store round-trips
    it fine (plain json.dumps/loads allow it), only Starlette's strict
    JSONResponse rejects it. The API must sanitize at the boundary, not
    just at the one producer already fixed, so pre-existing data doesn't
    keep crashing every request."""

    def test_activity_with_raw_infinite_detail_is_json_safe(self, settings, tmp_path):
        store = Store(settings.db_path)
        store.add_activity("economics", "legacy poisoned record",
                            detail={"runway_hours": float("inf")})
        store.close()
        client = TestClient(create_app(settings))
        r = client.get("/api/activity")
        assert r.status_code == 200
        assert r.json()[0]["detail"]["runway_hours"] is None

    def test_summary_with_raw_infinite_economics_meta_is_json_safe(self, settings, tmp_path):
        store = Store(settings.db_path)
        store.set_meta("economics", {"runway_hours": float("inf")})
        store.close()
        client = TestClient(create_app(settings))
        r = client.get("/api/summary")
        assert r.status_code == 200
        assert r.json()["economics"]["runway_hours"] is None


class TestNewEndpoints:
    def test_options_endpoint(self, client):
        r = client.get("/api/options")
        assert r.status_code == 200
        body = r.json()
        assert "open" in body and "closed" in body
        assert isinstance(body["open"], list)

    def test_leaderboard_endpoint(self, client):
        r = client.get("/api/leaderboard")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list)
        if rows:  # learning enabled by default -> populated
            assert {"id", "strategy", "fitness", "trades"} <= set(rows[0])

    def test_benchmark_endpoint(self, client):
        r = client.get("/api/benchmark")
        assert r.status_code == 200
        body = r.json()
        assert "curve" in body and "alpha" in body
        assert isinstance(body["curve"], list)

    def test_news_endpoint(self, client):
        r = client.get("/api/news")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
