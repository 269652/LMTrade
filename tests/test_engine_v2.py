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
        engine = make_engine(settings, store)
        cash_before = engine.broker.cash()
        for _ in range(5):
            engine.run_cycle()
            if store.open_options():
                break
        opts = store.open_options()
        assert opts, "engine should open at least one option position"
        assert opts[0]["underlying"] == "AAPL"
        assert opts[0]["kind"] in ("call", "put")
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

    def test_daily_analysis_runs_and_applies(self, settings, store):
        response = json.dumps({"risk": {"min_confidence": 0.7}, "notes": "ok"})
        engine = make_engine(settings, store,
                             analysis_caller=lambda p: (response, 0.0))
        engine.run_cycle()
        assert settings.risk.min_confidence == pytest.approx(0.7)
        assert any(a["kind"] == "insight" for a in store.recent_activity(20))


class TestEquityValuation:
    def test_equity_includes_option_marks(self, settings, store):
        engine = make_engine(settings, store)
        engine.run_cycle()
        curve = store.equity_curve(10)
        assert curve
        # cash spent on premium must be (approximately) recovered in equity mark
        last = curve[-1]
        assert last["equity"] >= last["cash"]


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
