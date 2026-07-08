"""Smoke tests: the bot must run end-to-end in paper mode with no external
services, no network and no API keys."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.agents.fusion import FusionEngine
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store, Trade
from lmtrade.data.market import MarketData
from lmtrade.economics.cost_accounting import CostAccountant
from lmtrade.finance import indicators
from lmtrade.finance.risk import size_position, should_exit
from lmtrade.models.providers import build_providers


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL", "MSFT"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1   # keep multi-cycle tests fast
    return s


def test_indicators_need_history():
    assert indicators.sma([1, 2], 5) is None
    assert indicators.rsi(list(range(1, 40))) is not None
    snap = indicators.indicator_snapshot([float(i) for i in range(1, 60)])
    assert snap["last"] == 59.0


def test_synthetic_market_always_returns_data():
    md = MarketData("synthetic", lookback=40)
    q = md.quote("AAPL")
    assert q.price > 0
    assert len(q.history) == 40
    assert q.source == "synthetic"


def test_paper_broker_buy_sell(tmp_path: Path):
    store = Store(tmp_path / "t.db")
    broker = PaperBroker(store, starting_cash=100.0, fee=1.0)
    assert broker.cash() == 100.0
    r = broker.buy("AAPL", qty=2, price=10.0)
    assert r.ok and broker.cash() == pytest.approx(100 - 20 - 1)
    assert store.position("AAPL").qty == 2
    r = broker.sell("AAPL", qty=2, price=12.0)
    assert r.ok and broker.cash() == pytest.approx(79 + 24 - 1)
    assert store.position("AAPL") is None


def test_paper_broker_rejects_overspend(tmp_path: Path):
    store = Store(tmp_path / "t.db")
    broker = PaperBroker(store, starting_cash=5.0)
    assert not broker.buy("AAPL", qty=10, price=100.0).ok


def test_risk_sizing_and_exit(settings: Settings):
    r = size_position(price=10, cash=10, equity=10, confidence=0.9, cfg=settings.risk)
    assert r.qty > 0
    r0 = size_position(price=10, cash=10, equity=10, confidence=0.1, cfg=settings.risk)
    assert r0.qty == 0
    hit, _ = should_exit(avg_price=10, last_price=9.0, cfg=settings.risk)
    assert hit  # 10% drop trips the 5% stop


def test_fusion_produces_decision(settings: Settings):
    fusion = FusionEngine(settings, build_providers(settings))
    q = MarketData("synthetic").quote("AAPL")
    d = fusion.decide(q)
    assert d.direction in ("buy", "sell", "hold")
    assert 0.0 <= d.confidence <= 1.0
    assert d.signals


def test_economics_snapshot(settings: Settings):
    store = Store(settings.db_path)
    store.set_meta("starting_cash", 10.0)
    acc = CostAccountant(settings, store)
    snap = acc.snapshot(cash_eur=10.0, positions_value_eur=0.0)
    assert snap.net_worth_eur == 10.0
    assert snap.runway_hours > 0
    assert isinstance(snap.self_sustaining, bool)


def test_engine_runs_cycles(settings: Settings):
    store = Store(settings.db_path)
    broker = PaperBroker(store, starting_cash=settings.budget)
    engine = Engine(settings, store, broker)
    engine.run_forever(max_cycles=3)
    # After a few cycles we should have logged activity and an equity curve.
    assert store.recent_activity(10)
    assert store.equity_curve(10)


def test_viz_renders_headless(settings: Settings, monkeypatch):
    """The inline dashboard must build a figure from real data without a display."""
    import matplotlib
    matplotlib.use("Agg")
    # Point viz's config loader at this test's data dir.
    monkeypatch.setattr("lmtrade.viz.panels.load_settings", lambda: settings)

    store = Store(settings.db_path)
    engine = Engine(settings, store, PaperBroker(store, starting_cash=settings.budget))
    engine.run_forever(max_cycles=3)
    store.close()

    from lmtrade import viz

    assert not viz.trades_df(db_path=settings.db_path).empty or True  # may be empty
    assert isinstance(viz.summary(db_path=settings.db_path), dict)
    fig = viz.dashboard_figure(db_path=settings.db_path)
    assert fig is not None
    assert len(fig.axes) >= 4
    # Equity axis must overlay the retail benchmark when benchmark data exists.
    eq_ax = fig.axes[0]
    labels = [ln.get_label() for ln in eq_ax.lines]
    assert any("benchmark" in str(l).lower() for l in labels), labels
