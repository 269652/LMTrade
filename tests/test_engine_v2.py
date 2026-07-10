"""Tests for the v2 engine: options trading, strategy learning, benchmark
tracking, and scheduled research jobs. Written before implementation (TDD)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.strategies.optimizer import StrategyOptimizer


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.risk.min_confidence = 0.5
    s.options.enabled = True
    s.learning.enabled = True
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make_engine(settings, store, **kwargs) -> Engine:
    broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
    return Engine(settings, store, broker, **kwargs)


class TestPaperBrokerCash:
    def test_adjust_cash_credit_and_debit(self, settings, store):
        broker = PaperBroker(store, starting_cash=10.0)
        assert broker.adjust_cash(-4.0) is True
        assert broker.cash() == pytest.approx(6.0)
        assert broker.adjust_cash(3.0) is True
        assert broker.cash() == pytest.approx(9.0)

    def test_debit_rejects_overdraw(self, settings, store):
        broker = PaperBroker(store, starting_cash=5.0)
        assert broker.adjust_cash(-10.0) is False
        assert broker.cash() == pytest.approx(5.0)


class TestOptionsTrading:
    def test_engine_opens_option_with_genome_attribution(self, settings, store):
        # Policy: only real TR knockout instruments are ever traded (paper
        # simulates the fill; no synthetic Black-Scholes fallback), so opening
        # a position in a test requires a TR client with a catalog.
        from lmtrade.brokers.tr_derivatives import FakeTRDerivatives, TRDerivativeQuote

        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000TEST1", underlying="AAPL",
                                       kind="ko_call", strike=80.0, barrier=80.0,
                                       ratio=10.0, price=2.05, leverage=5.0,
                                       issuer="TestBank"),
                     TRDerivativeQuote(isin="DE000TEST2", underlying="AAPL",
                                       kind="ko_put", strike=120.0, barrier=120.0,
                                       ratio=10.0, price=2.05, leverage=5.0,
                                       issuer="TestBank")]})
        engine = make_engine(settings, store, tr_derivatives=client)
        cash_before = engine.broker.cash()
        for _ in range(5):
            engine.run_cycle()
            if store.open_options():
                break
        opts = store.open_options()
        assert opts, "engine should open at least one option position"
        assert opts[0]["underlying"] == "AAPL"
        assert opts[0]["kind"] in ("ko_call", "ko_put")
        assert opts[0]["isin"] in ("DE000TEST1", "DE000TEST2")
        assert opts[0]["contracts"] > 0
        if settings.learning.enabled:
            assert opts[0]["genome_id"]
        assert engine.broker.cash() < cash_before

    def test_option_closed_near_expiry_and_learning_recorded(self, settings, store):
        engine = make_engine(settings, store)
        # Plant an option expiring inside the force-close window.
        oid = store.open_option(
            "AAPL", "call", strike=1.0, expiry_ts=time.time() + 3600,  # 1h left
            iv=0.4, contracts=1.0, entry_premium=0.5, genome_id="gtest",
        )
        # Make the genome known to the optimizer so attribution lands.
        genomes = store.get_meta("genomes") or []
        genomes.append({"id": "gtest", "strategy": "momentum",
                        "params": {"fast": 5, "slow": 20, "threshold": 0.002},
                        "trades": 0, "pnl": 0.0})
        store.set_meta("genomes", genomes)

        engine.run_cycle()

        assert oid not in [o["id"] for o in store.open_options()]
        closed = store.closed_options()
        assert closed and closed[0]["id"] == oid
        assert closed[0]["pnl"] is not None
        g = [x for x in (store.get_meta("genomes") or []) if x["id"] == "gtest"][0]
        assert g["trades"] == 1

    def test_option_take_profit_exit(self, settings, store):
        settings.risk.min_confidence = 1.1   # block new entries: isolate the exit
        engine = make_engine(settings, store)
        # Deep ITM call opened at a tiny premium -> mark >> entry -> take profit.
        store.open_option("AAPL", "call", strike=0.01,
                          expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=0.01, entry_premium=0.01,
                          genome_id=None)
        cash_before = engine.broker.cash()
        engine.run_cycle()
        closed = store.closed_options()
        assert closed, "deep ITM option should hit take-profit"
        assert closed[0]["pnl"] > 0
        assert engine.broker.cash() > cash_before  # proceeds credited


class TestBenchmark:
    def test_benchmark_curve_and_alpha(self, settings, store):
        engine = make_engine(settings, store)
        engine.run_cycle()
        engine.run_cycle()
        assert store.get_meta("benchmark_entry") is not None
        curve = store.benchmark_curve()
        assert len(curve) >= 1
        assert curve[0]["equity"] == pytest.approx(settings.budget, rel=0.2)
        alpha = store.get_meta("alpha")
        assert alpha is not None


class TestScheduledResearch:
    def test_hourly_news_fetched_once_per_interval(self, settings, store):
        calls: list[str] = []

        def fake_news(symbol):
            calls.append(symbol)
            return f"{symbol} steady. SENTIMENT: neutral", 0.0

        clock = [1_000_000.0]
        engine = make_engine(settings, store, news_fetcher=fake_news,
                             now=lambda: clock[0])
        engine.run_cycle()
        assert calls == ["AAPL"]
        engine.run_cycle()                       # still within the hour
        assert calls == ["AAPL"]
        clock[0] += 3601
        engine.run_cycle()
        assert calls == ["AAPL", "AAPL"]

    def test_news_fetch_concurrency_capped_at_5(self, settings, store):
        import threading

        settings.universe = [f"SYM{i}" for i in range(12)]
        concurrent = [0]
        max_seen = [0]
        lock = threading.Lock()

        def slow_fetch(symbol):
            with lock:
                concurrent[0] += 1
                max_seen[0] = max(max_seen[0], concurrent[0])
            import time as _t
            _t.sleep(0.05)
            with lock:
                concurrent[0] -= 1
            return f"{symbol} steady. SENTIMENT: neutral", 0.0

        engine = make_engine(settings, store, news_fetcher=slow_fetch)
        engine.run_cycle()
        assert max_seen[0] <= 5, "news fetch must be capped at 5 concurrent requests"

    def test_daily_analysis_runs_and_applies(self, settings, store):
        response = json.dumps({"risk": {"min_confidence": 0.7}, "notes": "ok"})
        engine = make_engine(settings, store,
                             analysis_caller=lambda p: (response, 0.0))
        engine.run_cycle()
        assert settings.risk.min_confidence == pytest.approx(0.7)
        assert any(a["kind"] == "insight" for a in store.recent_activity(20))

    def test_fresh_analysis_in_db_is_not_rebuilt(self, settings, store):
        # A valid, non-expired analysis already in the DB must be reused, not
        # recompiled on (re)start — the expensive web-search call is skipped.
        store.set_meta("market_analysis", {
            "valid_hours": settings.research.daily_analysis_interval_hours,
            "ts": 1_000_000.0,
            "symbols": {"AAPL": {"bias": "bullish", "confidence": 0.6, "notes": "x"}},
        })
        prompts: list[str] = []

        def caller(p):
            prompts.append(p)
            return json.dumps({"symbols": {"AAPL": {"bias": "bearish",
                                                    "confidence": 0.9}}}), 0.0

        # Clock just after the analysis timestamp -> still well within validity.
        engine = make_engine(settings, store, analysis_caller=caller,
                             now=lambda: 1_000_100.0)
        engine.run_cycle()
        assert not any("directional bias" in p for p in prompts)  # no recompile
        a = store.get_meta("market_analysis")
        assert a["ts"] == 1_000_000.0                             # untouched
        assert a["symbols"]["AAPL"]["bias"] == "bullish"

    def test_stale_analysis_in_db_is_rebuilt(self, settings, store):
        store.set_meta("market_analysis", {
            "valid_hours": settings.research.daily_analysis_interval_hours,
            "ts": 1_000_000.0,
            "symbols": {"AAPL": {"bias": "bullish", "confidence": 0.6}},
        })
        prompts: list[str] = []

        def caller(p):
            prompts.append(p)
            return json.dumps({"symbols": {"AAPL": {"bias": "bearish",
                                                    "confidence": 0.9}}}), 0.0

        # Clock past the validity window -> stale, must recompile.
        stale = 1_000_000.0 + settings.research.daily_analysis_interval_hours * 3600 + 10
        engine = make_engine(settings, store, analysis_caller=caller,
                             now=lambda: stale)
        engine.run_cycle()
        assert any("directional bias" in p for p in prompts)      # recompiled
        assert store.get_meta("market_analysis")["symbols"]["AAPL"]["bias"] == "bearish"


class TestEquityValuation:
    def test_equity_includes_option_marks(self, settings, store):
        engine = make_engine(settings, store)
        engine.run_cycle()
        curve = store.equity_curve(10)
        assert curve
        # cash spent on premium must be (approximately) recovered in equity mark
        last = curve[-1]
        assert last["equity"] >= last["cash"]


class TestOptionMarkPersistence:
    """Each cycle the engine already marks every open option to compute
    equity; it must persist those per-option marks + unrealized P&L so the
    dashboard can show live profit/loss instead of only entry premium."""

    def test_run_cycle_persists_unrealized_pnl(self, settings, store):
        # Block TP/SL exits so the planted option is guaranteed still open
        # when marks are persisted at the end of the cycle.
        settings.options.min_hold_hours = 10_000.0
        store.open_option("AAPL", "call", strike=1_000_000.0, expiry_ts=4e12,
                          iv=0.2, contracts=2.0, entry_premium=1.0,
                          genome_id=None, tp_premium=1.5, sl_premium=0.6)
        opt_id = store.open_options()[0]["id"]
        engine = make_engine(settings, store)
        engine.run_cycle()
        marks = store.get_meta("open_option_marks", {})
        assert str(opt_id) in marks
        row = marks[str(opt_id)]
        assert "unrealized_pnl" in row
        assert "value" in row
        assert "mark_premium" in row


class TestNewsStaleGate:
    """When Claude is the configured news/analysis provider and it goes dark
    (unavailable, or news simply hasn't updated in a while), the engine must
    stop opening NEW positions — existing ones are still fully managed
    (TP/SL/expiry/knockout) — and resume automatically the moment fresh
    (<2h old) news is available again. No separate 'unpause' action needed."""

    def _client(self):
        from _tr_test_helpers import AnyKnockoutTR
        return AnyKnockoutTR()

    def test_blocks_new_entries_when_claude_configured_and_news_stale(self, settings, store):
        settings.research.news_provider = "claude_cli"
        from lmtrade.agents.fusion import Decision
        engine = make_engine(settings, store, tr_derivatives=self._client())
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert store.open_options() == []

    def test_allows_new_entries_when_fresh_news_exists(self, settings, store):
        settings.research.news_provider = "claude_cli"
        store.add_news("AAPL", "real news. SENTIMENT: bullish", "bullish", ts=time.time())
        from lmtrade.agents.fusion import Decision
        engine = make_engine(settings, store, tr_derivatives=self._client())
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert store.open_options(), "fresh news must unblock new entries"

    def test_stale_news_blocked_even_if_more_than_2h_old(self, settings, store):
        settings.research.news_provider = "claude_cli"
        store.add_news("AAPL", "old news. SENTIMENT: bullish", "bullish",
                       ts=time.time() - 3 * 3600)   # 3h old > 2h gate
        from lmtrade.agents.fusion import Decision
        engine = make_engine(settings, store, tr_derivatives=self._client())
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert store.open_options() == []

    def test_existing_positions_still_managed_when_gated(self, settings, store):
        settings.research.news_provider = "claude_cli"
        settings.options.min_hold_hours = 0.0
        # Deep-OTM short-expiry call: force-closes near expiry regardless of gate.
        store.open_option("AAPL", "call", strike=1_000_000.0,
                          expiry_ts=time.time() + 60.0, iv=0.2, contracts=1.0,
                          entry_premium=1.0, genome_id=None,
                          tp_premium=1e9, sl_premium=0.0)
        engine = make_engine(settings, store, tr_derivatives=self._client())
        engine.run_cycle()
        assert store.closed_options(), "existing positions must still be managed when gated"

    def test_no_gate_when_claude_not_configured(self, settings, store):
        # Default provider isn't claude_cli — the gate must not apply, so an
        # empty news table (no research configured at all) never blocks entries.
        assert settings.research.news_provider != "claude_cli"
        from lmtrade.agents.fusion import Decision
        engine = make_engine(settings, store, tr_derivatives=self._client())
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert store.open_options(), "gate must not apply when Claude isn't the provider"

    def test_gate_also_applies_when_only_analysis_provider_is_claude(self, settings, store):
        settings.research.news_provider = "yahoo"   # arbitrary non-claude value
        settings.research.analysis_provider = "claude_cli"
        from lmtrade.agents.fusion import Decision
        engine = make_engine(settings, store, tr_derivatives=self._client())
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert store.open_options() == []


class TestSharedResearchAcrossBooks:
    """News and the compiled market analysis are process-wide research, not
    book-specific state — fetching them once must be visible to BOTH the
    paper and live engines without a second (wasted) fetch, whichever book
    happens to be 'active' (self.store) when each engine cycle runs."""

    def test_news_fetched_while_paper_active_seen_by_live_engine_without_refetch(
        self, settings, store
    ):
        from lmtrade.core.state import Store

        live = Store(store.db_path.parent / "live.db")
        calls: list[str] = []

        def fetcher(symbol):
            calls.append(symbol)
            return f"{symbol} steady. SENTIMENT: neutral", 0.0

        # Paper engine (self.store=paper, fallback_store=live) fetches news.
        paper_engine = make_engine(settings, store, news_fetcher=fetcher,
                                   fallback_store=live)
        paper_engine.run_cycle()
        assert calls == ["AAPL"]

        # Live engine (self.store=live, fallback_store=paper) must see the
        # SAME news without refetching, even though ITS OWN book (live) never
        # directly received the fetch.
        live_engine = make_engine(settings, live, news_fetcher=fetcher,
                                  fallback_store=store)
        live_engine.run_cycle()
        assert calls == ["AAPL"], "must not refetch news already fetched by the other book"
        live.close()

    def test_analysis_compiled_while_paper_active_seen_by_live_engine_without_recompile(
        self, settings, store
    ):
        from lmtrade.core.state import Store

        live = Store(store.db_path.parent / "live.db")
        prompts: list[str] = []

        def caller(p):
            prompts.append(p)
            return json.dumps({"symbols": {"AAPL": {"bias": "bullish",
                                                    "confidence": 0.8}}}), 0.0

        def analysis_prompts() -> int:
            return sum(1 for p in prompts if "directional bias" in p)

        paper_engine = make_engine(settings, store, analysis_caller=caller,
                                   fallback_store=live)
        paper_engine.run_cycle()
        assert analysis_prompts() == 1

        live_engine = make_engine(settings, live, analysis_caller=caller,
                                  fallback_store=store)
        live_engine.run_cycle()
        assert analysis_prompts() == 1, "must not recompile analysis already fresh in the other book"
        assert live.get_meta("market_analysis")["symbols"]["AAPL"]["bias"] == "bullish"
        live.close()

    def test_inference_cost_charged_once_not_mirrored_to_both_books(self, settings, store):
        # Sharing DATA must not double-bill: one real API call costs money
        # once, charged to whichever book's engine actually made the call.
        from lmtrade.core.state import Store

        live = Store(store.db_path.parent / "live.db")

        def caller(p):
            return json.dumps({"symbols": {"AAPL": {"bias": "bullish", "confidence": 0.8}}}), 0.05

        paper_engine = make_engine(settings, store, analysis_caller=caller,
                                   fallback_store=live)
        paper_engine.run_cycle()
        assert store.total_costs().get("inference", 0.0) > 0.0    # billed to the active book
        assert live.total_costs().get("inference", 0.0) == pytest.approx(0.0)  # never mirrored
        live.close()


class TestCrossBookValuation:
    """The live dashboard view must ALWAYS show real TR cash and an accurate
    net worth (real cash + real position value) — even while UNARMED, when
    the trading engine is actively running against the PAPER book (armed
    live execution is a separate guard from which book the dashboard shows).
    Real live positions are synced in via tr_sync; this engine must keep
    marking them fresh every cycle using the SAME already-fetched prices,
    regardless of which book it is currently trading."""

    def test_fallback_book_marks_refreshed_using_this_cycles_prices(self, settings, store):
        from lmtrade.core.state import Store

        fallback = Store(store.db_path.parent / "fallback.db")
        fallback.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                             contracts=5.0, entry_premium=2.0, genome_id=None,
                             tp_premium=3.0, sl_premium=1.0,
                             instrument_type="knockout", barrier=80.0, ratio=10.0,
                             isin="DE000REAL01")
        opt_id = fallback.open_options()[0]["id"]

        from _tr_test_helpers import AnyKnockoutTR
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        engine = Engine(settings, store, broker, fallback_store=fallback,
                        tr_derivatives=AnyKnockoutTR())
        engine.run_cycle()

        marks = fallback.get_meta("open_option_marks", {})
        assert str(opt_id) in marks, "fallback (live) book's position must be marked too"
        assert "unrealized_pnl" in marks[str(opt_id)]
        fallback.close()

    def test_tr_account_cash_mirrored_into_fallback_book(self, settings, store):
        from lmtrade.core.state import Store

        fallback = Store(store.db_path.parent / "fallback.db")

        class CashClient:
            def available(self):
                return True

            def account_cash(self):
                return 555.0

            def portfolio(self):
                return None

            def search(self, *a, **k):
                return []

            def find_knockout(self, *a, **k):
                return None

        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        engine = Engine(settings, store, broker, fallback_store=fallback,
                        tr_derivatives=CashClient())
        engine.run_cycle()

        assert fallback.get_meta("tr_account_cash") == pytest.approx(555.0)
        fallback.close()

    def test_no_fallback_store_does_not_crash(self, settings, store):
        # fallback_store is optional (e.g. some construction paths omit it);
        # cross-book marking must be a no-op, not a crash.
        engine = make_engine(settings, store)
        engine.fallback_store = None
        engine.run_cycle()   # must not raise


class TestRealizedNetWorthMark:
    """last_realized_net_worth is a realized high-water mark: net worth locked
    in at the most recent closed trade, reset on each close."""

    def test_initializes_to_starting_cash(self, settings, store):
        engine = make_engine(settings, store)
        engine.run_cycle()
        assert store.get_meta("last_realized_net_worth") == pytest.approx(settings.budget)

    def test_resets_mark_on_realized_close(self, settings, store):
        # An option whose stop-loss is set absurdly high closes immediately
        # this cycle (mark <= sl), realizing a loss.
        store.open_option("AAPL", "call", strike=100.0, expiry_ts=4e12, iv=0.2,
                          contracts=1.0, entry_premium=1.0, genome_id=None,
                          tp_premium=1e9, sl_premium=1e9)
        engine = make_engine(settings, store)
        engine.run_cycle()
        assert store.closed_options_count() == 1
        econ = store.get_meta("economics")
        assert store.get_meta("last_realized_net_worth") == pytest.approx(
            econ["net_worth_eur"])

    def test_mark_unchanged_without_a_close(self, settings, store):
        engine = make_engine(settings, store)
        engine.run_cycle()
        first = store.get_meta("last_realized_net_worth")
        engine.run_cycle()   # no close happened
        assert store.get_meta("last_realized_net_worth") == first


class TestBookFullVisibility:
    """When every position slot is taken, the engine skips the whole
    decision/entry step and previously logged nothing but economics — which
    reads as "stuck / doing nothing". It must say why."""

    def test_full_book_emits_explanatory_log(self, settings, store):
        settings.loop.max_positions = 2
        # Fill the book with two open options so no slot is free.
        for i in range(2):
            store.open_option(f"SYM{i}", "call", strike=100.0, expiry_ts=4e12,
                              iv=0.2, contracts=1.0, entry_premium=1.0,
                              genome_id=None, tp_premium=1.5, sl_premium=0.6)
        engine = make_engine(settings, store)
        engine.run_cycle()
        logs = " ".join(l["message"] for l in store.recent_logs(50))
        activity = " ".join(a["summary"] for a in store.recent_activity(50))
        haystack = (logs + " " + activity).lower()
        assert "no free" in haystack or "full" in haystack

    def test_free_slots_do_not_emit_full_log(self, settings, store):
        settings.loop.max_positions = 8
        engine = make_engine(settings, store)
        engine.run_cycle()
        logs = " ".join(l["message"] for l in store.recent_logs(50)).lower()
        assert "no free slot" not in logs

    def test_pending_orders_count_toward_max_positions(self, settings, store):
        # A live order still awaiting TR confirmation occupies a slot just
        # like a confirmed one — otherwise the book could accept more
        # concurrent orders than max_positions while several are in flight.
        settings.loop.max_positions = 1
        store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=4e12, iv=0.0,
                          contracts=1.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0, instrument_type="knockout",
                          barrier=80.0, ratio=10.0, isin="DE000PENDING", status="pending")
        engine = make_engine(settings, store)
        engine.run_cycle()
        logs = " ".join(l["message"] for l in store.recent_logs(50)).lower()
        assert "no free slot" in logs or "full" in logs


class TestProfitStash:
    """Regression: profit_stash_pct saves to config.toml as an integer percent (e.g. 30)
    but the engine treats it as a fraction (0-1). A value of 30 causes
    stash = pnl * 30, which always exceeds available cash, so adjust_cash
    silently returns False and the reserve never grows.
    Fix: normalise any value > 1 as a percentage at config load time."""

    def test_stash_pct_integer_percent_normalised_to_fraction(self):
        """EconomicsConfig(profit_stash_pct=30) must behave as 0.30, not 30.0."""
        from lmtrade.config import EconomicsConfig
        cfg = EconomicsConfig(profit_stash_pct=30)
        assert cfg.profit_stash_pct == pytest.approx(0.30), (
            "profit_stash_pct=30 should be normalised to 0.30 (30%)"
        )

    def test_stash_pct_fraction_unchanged(self):
        """A fractional value (0 <= v <= 1) must not be modified."""
        from lmtrade.config import EconomicsConfig
        assert EconomicsConfig(profit_stash_pct=0.3).profit_stash_pct == pytest.approx(0.3)
        assert EconomicsConfig(profit_stash_pct=0.0).profit_stash_pct == pytest.approx(0.0)
        assert EconomicsConfig(profit_stash_pct=1.0).profit_stash_pct == pytest.approx(1.0)

    def test_profitable_close_increases_reserve(self, settings, store):
        """Closing a winning position with profit_stash_pct=0.3 must add to reserve."""
        settings.economics.profit_stash_pct = 0.3
        settings.options.min_hold_hours = 0.0
        # Deep-ITM call: mark >> entry -> TP fires immediately, pnl >> 0.
        store.open_option(
            "AAPL", "call", strike=0.001, expiry_ts=time.time() + 7 * 86400,
            iv=0.2, contracts=1.0, entry_premium=0.01,
            genome_id=None, tp_premium=0.02, sl_premium=0.0,
        )
        engine = make_engine(settings, store)
        engine.run_cycle()
        assert store.closed_options_count() >= 1, "deep-ITM option should have been closed"
        assert store.reserve_balance() > 0, (
            "reserve must increase after a profitable close with profit_stash_pct=0.3"
        )

    def test_no_reserve_increase_on_losing_close(self, settings, store):
        """Only wins contribute to the reserve; losses must not."""
        settings.economics.profit_stash_pct = 0.3
        settings.options.min_hold_hours = 0.0
        # Deep-OTM call: mark ~= 0 << entry -> SL fires, pnl < 0.
        store.open_option(
            "AAPL", "call", strike=1_000_000.0, expiry_ts=time.time() + 7 * 86400,
            iv=0.2, contracts=1.0, entry_premium=1.0,
            genome_id=None, tp_premium=1.5, sl_premium=0.99,
        )
        engine = make_engine(settings, store)
        engine.run_cycle()
        assert store.reserve_balance() == pytest.approx(0.0), (
            "a losing close must never add to the reserve"
        )


class TestStaleEviction:
    """Positions that go sideways for too long must be closed to free a slot
    for stronger incoming signals. Only old positions that are NOT making
    meaningful gains qualify — big winners are held, young positions are
    untouched."""

    _EXPIRY = time.time() + 7 * 86_400   # 7 days out, well above min_hours_to_expiry

    def _stale_opt(self, store):
        """Deep-OTM option: mark ≈ 0, entry=1.0 → pnl_pct ≈ -100% (below any threshold).
        sl=0.0 + min_hold_hours=1000 keep the normal SL gate from firing."""
        return store.open_option(
            "AAPL", "call", strike=1_000_000.0,
            expiry_ts=self._EXPIRY,
            iv=0.2, contracts=1.0, entry_premium=1.0,
            genome_id=None, tp_premium=1e9, sl_premium=0.0,
        )

    def _winning_opt(self, store):
        """Deep-ITM option: mark >> entry → pnl_pct >> stale_max_profit_pct."""
        return store.open_option(
            "AAPL", "call", strike=0.001,
            expiry_ts=self._EXPIRY,
            iv=0.2, contracts=1.0, entry_premium=0.01,
            genome_id=None, tp_premium=1e9, sl_premium=0.0,
        )

    def _setup(self, settings):
        settings.loop.stale_evict_enabled = True
        settings.loop.stale_hours = 48.0
        settings.loop.stale_max_profit_pct = 0.10
        settings.options.min_hold_hours = 1000.0   # block TP/SL; only stale evicts

    def test_old_sideways_position_evicted(self, settings, store):
        """An option older than stale_hours with pnl < stale_max_profit_pct must be closed."""
        self._setup(settings)
        oid = self._stale_opt(store)
        # Pretend 72 hours have passed since the position was opened.
        future = time.time() + 72 * 3600
        make_engine(settings, store, now=lambda: future).run_cycle()

        assert oid not in {o["id"] for o in store.open_options()}, (
            "stale option should be evicted after 72 h with near-zero PnL"
        )
        trades = store.recent_activity(50)
        assert any("stale" in (t.get("summary") or "").lower() for t in trades), (
            "stale close must appear in activity log"
        )

    def test_young_position_not_evicted(self, settings, store):
        """An option that is only minutes old must not be evicted, even if pnl ≈ 0."""
        self._setup(settings)
        oid = self._stale_opt(store)
        # No time offset — the option was just opened.
        make_engine(settings, store).run_cycle()

        assert oid in {o["id"] for o in store.open_options()}, (
            "young option must not be evicted by stale logic"
        )

    def test_winning_position_not_evicted(self, settings, store):
        """A position with pnl >= stale_max_profit_pct must never be evicted."""
        self._setup(settings)
        oid = self._winning_opt(store)
        future = time.time() + 72 * 3600
        make_engine(settings, store, now=lambda: future).run_cycle()

        assert oid in {o["id"] for o in store.open_options()}, (
            "winning option must NOT be stale-evicted even when old"
        )

    def test_stale_eviction_disabled(self, settings, store):
        """stale_evict_enabled=False suppresses eviction even when conditions are met."""
        self._setup(settings)
        settings.loop.stale_evict_enabled = False
        oid = self._stale_opt(store)
        future = time.time() + 72 * 3600
        make_engine(settings, store, now=lambda: future).run_cycle()

        assert oid in {o["id"] for o in store.open_options()}, (
            "stale eviction must respect stale_evict_enabled=False"
        )

    def test_stale_eviction_frees_slot_for_new_entry(self, settings, store):
        """Stale eviction reduces open_count so the candidates loop can enter a new position."""
        self._setup(settings)
        settings.loop.max_positions = 1  # book is full with one option
        # A single stale option fills the only slot.
        self._stale_opt(store)
        open_before = len(store.open_options())
        assert open_before == 1

        future = time.time() + 72 * 3600
        make_engine(settings, store, now=lambda: future).run_cycle()

        # After eviction the book is no longer full; the engine may or may not
        # open a new position depending on signal, but the stale one is gone.
        closed = store.closed_options()
        assert closed, "stale option must be closed (freeing the slot)"


class TestCashReservePct:
    """Keep a configurable fraction of net worth in cash rather than deploying
    everything. As the portfolio grows, the absolute position budget grows too
    (proportional scaling). The reserve is enforced per new entry."""

    def test_cash_reserve_pct_normalises_percent(self):
        """OptionsConfig(cash_reserve_pct=40) must behave as 0.40, not 40.0."""
        from lmtrade.config import OptionsConfig
        assert OptionsConfig(cash_reserve_pct=40).cash_reserve_pct == pytest.approx(0.40), (
            "cash_reserve_pct=40 should be normalised to 0.40 (40 %)"
        )
        assert OptionsConfig(cash_reserve_pct=0.4).cash_reserve_pct == pytest.approx(0.40)
        # Edge values
        assert OptionsConfig(cash_reserve_pct=0).cash_reserve_pct == pytest.approx(0.0)
        assert OptionsConfig(cash_reserve_pct=1.0).cash_reserve_pct == pytest.approx(1.0)

    def test_fully_reserved_blocks_all_new_positions(self, settings, store):
        """cash_reserve_pct=1.0 means zero deployable capital → no positions opened."""
        settings.options.cash_reserve_pct = 1.0
        settings.options.enabled = True
        make_engine(settings, store).run_cycle()
        assert store.open_options() == [], (
            "with 100 % cash reserve no new option should be opened"
        )

    def test_budget_respects_reserve_cap(self, settings, store):
        """_position_budget is capped so deployed + new ≤ (1-reserve)*equity."""
        from unittest.mock import MagicMock
        from lmtrade.data.market import Quote
        from lmtrade.agents.fusion import Decision

        settings.options.cash_reserve_pct = 0.9   # only 10 % deployable
        settings.options.max_option_fraction = 0.5 # would want 50 % per trade

        engine = make_engine(settings, store)
        # equity = budget = 10, max_deployable = 10 * 0.1 = 1.0
        quote = MagicMock(spec=Quote)
        quote.symbol = "AAPL"
        quote.price = 150.0
        quote.history = [150.0] * 50
        decision = MagicMock(spec=Decision)
        decision.direction = "buy"
        decision.confidence = 1.0

        budget = engine._position_budget(quote, decision, None)
        # Without reserve: min(0.5 * 10, 10 - fee) = 5.0
        # With reserve:    min(0.5 * 10, 1.0,  10 - fee) = 1.0
        assert budget <= settings.budget * (1.0 - settings.options.cash_reserve_pct) + 0.2, (
            "budget must not exceed deployable_capital = equity * (1-cash_reserve_pct)"
        )
