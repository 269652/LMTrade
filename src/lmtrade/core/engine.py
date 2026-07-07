"""The trading engine — one evaluation loop over the universe.

Per instrument, each cycle:
  1. Fetch a quote (live or synthetic).
  2. Check economics: if runway is below the floor, only manage exits.
  3. Manage open positions (stop-loss / take-profit).
  4. If allowed, run the fusion engine and act on the decision, sizing with the
     risk module and executing through the broker.
  5. Persist an equity snapshot and structured activity for the dashboard.
"""
from __future__ import annotations

import time

from ..agents.fusion import Decision, FusionEngine
from ..brokers.base import Broker
from ..config import Settings
from ..data.market import MarketData, Quote
from ..economics.cost_accounting import CostAccountant
from ..finance.risk import should_exit, size_position
from ..models.providers import build_providers
from .events import EventBus
from .state import Store, Trade


class Engine:
    def __init__(self, settings: Settings, store: Store, broker: Broker):
        self.settings = settings
        self.store = store
        self.broker = broker
        self.bus = EventBus(store)
        self.market = MarketData(settings.data.provider)
        self.fusion = FusionEngine(settings, build_providers(settings))
        self.accountant = CostAccountant(settings, store)
        self._stop = False
        store.set_meta("mode", broker.mode)
        store.set_meta("universe", settings.universe)

    # -- valuation helpers ----------------------------------------------------
    def _positions_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for pos in self.store.positions():
            px = prices.get(pos.symbol, pos.avg_price)
            total += pos.qty * px
        return total

    # -- one full cycle -------------------------------------------------------
    def run_cycle(self) -> None:
        prices: dict[str, float] = {}
        quotes: dict[str, Quote] = {}
        for symbol in self.settings.universe:
            q = self.market.quote(symbol)
            quotes[symbol] = q
            prices[symbol] = q.price

        cash = self.broker.cash()
        econ = self.accountant.snapshot(cash, self._positions_value(prices))
        self.store.set_meta("economics", econ.as_dict())
        self.bus.activity(
            "economics",
            f"net €{econ.net_worth_eur:.2f} | runway {econ.runway_hours:.1f}h | "
            f"{'SELF-SUSTAINING' if econ.self_sustaining else 'subsidised'}",
            detail=econ.as_dict(),
        )
        if econ.halt_trading:
            self.bus.warn(
                f"Runway {econ.runway_hours:.1f}h < floor "
                f"{self.settings.economics.min_runway_hours}h — halting new entries, "
                "managing exits only.",
                source="economics",
            )

        for symbol in self.settings.universe:
            try:
                self._evaluate(symbol, quotes[symbol], econ)
            except Exception as exc:  # noqa: BLE001
                self.bus.error(f"{symbol}: cycle error: {exc}")

        # Persist equity snapshot after actions.
        cash = self.broker.cash()
        equity = cash + self._positions_value(prices)
        fees = self.store.total_costs().get("fee", 0.0)
        self.store.record_equity(cash, equity, fees)

    def _evaluate(self, symbol: str, quote: Quote, econ) -> None:
        last = quote.price
        pos = self.store.position(symbol)

        # 1) Manage exits on any open position (always allowed).
        if pos:
            exit_now, why = should_exit(
                avg_price=pos.avg_price, last_price=last, cfg=self.settings.risk
            )
            if exit_now:
                res = self.broker.sell(symbol, pos.qty, last)
                if res.ok:
                    self.store.record_trade(Trade(
                        symbol, "sell", res.qty, res.price, res.fee,
                        self.broker.mode, why, None,
                    ))
                    self.bus.activity("trade", f"SELL {symbol} — {why}", symbol,
                                      {"qty": res.qty, "price": res.price})
                return

        # 2) New entries gated by economics + position cap.
        if econ.halt_trading:
            return
        if len(self.store.positions()) >= self.settings.loop.max_positions and not pos:
            return
        if not self.accountant.can_afford_inference(econ, est_usd=0.01):
            self.bus.warn(f"{symbol}: skipping inference — would breach runway floor",
                          source="economics")
            return

        # 3) Fuse signals into a decision.
        decision: Decision = self.fusion.decide(quote)
        self.accountant.record_inference("fusion", decision.inference_cost)
        self.bus.activity(
            "decision",
            f"{symbol}: {decision.direction.upper()} conf {decision.confidence:.2f}",
            symbol, decision.as_dict(),
        )

        # 4) Act.
        if decision.direction == "buy" and not pos:
            sizing = size_position(
                price=last, cash=self.broker.cash(), equity=econ.net_worth_eur,
                confidence=decision.confidence, cfg=self.settings.risk,
            )
            if sizing.qty <= 0:
                self.bus.activity("risk", f"{symbol}: no buy — {sizing.reason}", symbol)
                return
            res = self.broker.buy(symbol, sizing.qty, last)
            if res.ok:
                self.store.record_trade(Trade(
                    symbol, "buy", res.qty, res.price, res.fee, self.broker.mode,
                    decision.rationale, decision.confidence,
                ))
                self.bus.activity("trade", f"BUY {symbol} ×{res.qty:.4f} @ {res.price:.2f}",
                                  symbol, {"sizing": sizing.reason})
            else:
                self.bus.warn(f"{symbol}: buy rejected — {res.message}")
        elif decision.direction == "sell" and pos:
            res = self.broker.sell(symbol, pos.qty, last)
            if res.ok:
                self.store.record_trade(Trade(
                    symbol, "sell", res.qty, res.price, res.fee, self.broker.mode,
                    decision.rationale, decision.confidence,
                ))
                self.bus.activity("trade", f"SELL {symbol} — signal", symbol)

    # -- long-running loop ----------------------------------------------------
    def stop(self) -> None:
        self._stop = True

    def run_forever(self, max_cycles: int | None = None) -> None:
        self.bus.info(
            f"Engine start | mode={self.broker.mode} | universe={self.settings.universe} | "
            f"interval={self.settings.loop.interval_seconds}s"
        )
        n = 0
        while not self._stop:
            started = time.time()
            self.run_cycle()
            n += 1
            if max_cycles and n >= max_cycles:
                self.bus.info(f"Reached max_cycles={max_cycles}, stopping.")
                break
            elapsed = time.time() - started
            time.sleep(max(0.0, self.settings.loop.interval_seconds - elapsed))
