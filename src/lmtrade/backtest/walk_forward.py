"""Walk-forward backtester: pre-train the strategy genome population on
historical data instead of waiting for live paper trades.

Classic walk-forward: the data is split into rolling folds of
[train window | test window], stepping forward by the test size. In each fold
every genome trades the train window and its realized simulated P&L is
attributed via the optimizer (which then evolves); the fittest genome is then
evaluated out-of-sample on the unseen test window. Because the optimizer
persists to the Store, a backtest leaves the live engine starting with a
pre-trained population.

The trade simulation mirrors the live engine's options logic: signal -> long
call/put at synthetic Black-Scholes premium, exits on premium take-profit /
stop-loss / expiry. One bar = one trading day.
"""
from __future__ import annotations

import random

from ..config import Settings
from ..core.state import Store
from ..data.market import MarketData
from ..finance.options import bs_price, realized_iv
from ..strategies.optimizer import Genome, StrategyOptimizer

WARMUP_BARS = 40         # bars of history a strategy needs before signaling
DAYS_PER_YEAR = 365.0
OPTION_FEE = 0.1
MIN_SIGNAL_STRENGTH = 0.3


def fetch_history(symbol: str, bars: int, provider: str = "auto") -> list[float]:
    """Daily close history, `bars` long. yfinance when available (period sized
    to the request), deterministic synthetic random walk otherwise."""
    if provider != "synthetic":
        try:
            import yfinance as yf

            years = max(1, int(bars / 252) + 1)
            hist = yf.Ticker(symbol).history(period=f"{years}y", interval="1d")
            closes = [float(x) for x in hist["Close"].dropna().tolist()]
            if len(closes) >= bars:
                return closes[-bars:]
        except Exception:  # noqa: BLE001 — fall back to synthetic
            pass
    md = MarketData("synthetic", lookback=bars)
    return md.quote(symbol).history


def simulate_genome(genome: Genome, prices: list[float], settings: Settings) -> list[dict]:
    """Run one genome over a price window with the live engine's option rules.
    Returns the closed trades: [{kind, entry_bar, exit_bar, pnl, reason}]."""
    cfg = settings.options
    trades: list[dict] = []
    open_pos: dict | None = None
    trade_notional = settings.budget * cfg.max_option_fraction

    for i in range(WARMUP_BARS, len(prices)):
        spot = prices[i]
        history = prices[: i + 1]

        if open_pos is not None:
            days_held = i - open_pos["entry_bar"]
            t_left = max(0.0, (cfg.expiry_days - days_held) / DAYS_PER_YEAR)
            mark = bs_price(spot, open_pos["strike"], t_left,
                            open_pos["iv"], open_pos["kind"])
            entry = open_pos["entry_premium"]
            change = (mark - entry) / max(1e-9, entry)
            reason = None
            if change >= cfg.take_profit_pct:
                reason = "take_profit"
            elif change <= -cfg.stop_loss_pct:
                reason = "stop_loss"
            elif days_held >= cfg.expiry_days - 1:
                reason = "expiry"
            if reason:
                pnl = (mark - entry) * open_pos["contracts"] - 2 * OPTION_FEE
                trades.append({"kind": open_pos["kind"],
                               "entry_bar": open_pos["entry_bar"], "exit_bar": i,
                               "pnl": pnl, "reason": reason})
                open_pos = None
            continue

        direction, strength = genome.signal(history)
        if direction == "hold" or strength < MIN_SIGNAL_STRENGTH:
            continue
        kind = "call" if direction == "buy" else "put"
        iv = realized_iv(history[-60:])
        t0 = cfg.expiry_days / DAYS_PER_YEAR
        premium = max(0.01, bs_price(spot, spot, t0, iv, kind))
        contracts = trade_notional / premium
        open_pos = {"kind": kind, "strike": spot, "iv": iv,
                    "entry_premium": premium, "contracts": contracts,
                    "entry_bar": i}

    return trades


class WalkForward:
    def __init__(self, settings: Settings, store: Store,
                 rng: random.Random | None = None):
        self.settings = settings
        self.store = store
        self.rng = rng or random.Random()
        self.optimizer = StrategyOptimizer(
            store, population=settings.learning.population,
            epsilon=0.0,  # backtests exploit deterministically; live loop explores
            mutation_scale=settings.learning.mutation_scale, rng=self.rng)

    def run(self, symbols: list[str], bars: int = 500,
            train_bars: int = 150, test_bars: int = 50) -> dict:
        if bars < train_bars + test_bars:
            raise ValueError(
                f"need at least train+test bars ({train_bars + test_bars}), got {bars}")

        histories = {s: fetch_history(s, bars, self.settings.data.provider)
                     for s in symbols}
        folds = []
        total_test_pnl = 0.0
        start = 0
        fold_no = 0
        while start + train_bars + test_bars <= bars:
            fold_no += 1
            train_pnl = 0.0
            # In-sample: every genome trades the train window; results feed learning.
            for genome in self.optimizer.genomes():
                for sym in symbols:
                    window = histories[sym][start: start + train_bars]
                    trades = simulate_genome(genome, window, self.settings)
                    pnl = sum(t["pnl"] for t in trades)
                    train_pnl += pnl
                    if trades:
                        self.optimizer.record_result(genome.id, pnl)
            self.optimizer.evolve()

            # Out-of-sample: the fittest genome trades the unseen test window.
            best = self.optimizer.select()
            test_pnl = 0.0
            for sym in symbols:
                window = histories[sym][start + train_bars:
                                        start + train_bars + test_bars]
                # prepend warmup context from the train tail so signals can fire
                context = histories[sym][start + train_bars - WARMUP_BARS:
                                         start + train_bars + test_bars]
                trades = simulate_genome(best, context, self.settings)
                test_pnl += sum(t["pnl"] for t in trades
                                if t["entry_bar"] >= WARMUP_BARS)
            total_test_pnl += test_pnl
            folds.append({"fold": fold_no, "best_genome": best.id,
                          "best_strategy": best.strategy,
                          "train_pnl": round(train_pnl, 4),
                          "test_pnl": round(test_pnl, 4)})
            start += test_bars

        return {"folds": folds,
                "total_test_pnl": round(total_test_pnl, 4),
                "leaderboard": self.optimizer.leaderboard()}
