"""The trading engine — high-cadence evaluation loop.

Per cycle:
  1. Scheduled research: hourly Perplexity news per instrument, daily Claude
     strategy review (both no-ops without keys; both injectable for tests).
  2. Quotes for the universe + the retail benchmark (buy-and-hold SPY).
  3. Economics gate: below the GPU-runway floor, only exits are managed.
  4. Manage open positions: option marks vs premium take-profit/stop/expiry,
     equity stop-loss/take-profit. Closed-trade P&L feeds the strategy
     optimizer (evolutionary learning).
  5. New entries: the fusion engine (financial models + SLM + LLM + news
     sentiment) is combined with the active strategy genome's signal; with
     options enabled a buy becomes a long call and a sell a long put —
     defined-risk leverage, which a tiny account needs.
  6. Persist equity, benchmark and alpha snapshots for the dashboard.
"""
from __future__ import annotations

import time
from typing import Callable

from ..agents.fusion import Decision, FusionEngine
from ..brokers.base import Broker
from ..config import Settings
from ..data.market import MarketData, Quote
from ..economics.cost_accounting import CostAccountant
from ..finance.options import mark_option, synth_option
from ..finance.risk import should_exit, size_position
from ..models.base import Signal
from ..models.providers import build_providers
from ..research.daily import DailyAnalyst
from ..research.news import NewsService
from ..strategies.optimizer import StrategyOptimizer
from .events import EventBus
from .scheduler import Scheduler
from .state import Store, Trade

OPTION_FEE = 0.1   # per option order (paper): smaller than TR's equity fee,
                   # comparable to warrant spreads on tiny notionals
NEWS_MAX_AGE_S = 24 * 3600   # news older than this no longer influences decisions


class Engine:
    def __init__(
        self, settings: Settings, store: Store, broker: Broker,
        news_fetcher=None, analysis_caller=None, market=None,
        now: Callable[[], float] = time.time,
    ):
        self.settings = settings
        self.store = store
        self.broker = broker
        self.now = now
        self.bus = EventBus(store)
        self.market = market or MarketData(settings.data.provider,
                                           intraday=settings.data.intraday)
        self._last_good_price: dict[str, float] = {}
        self.fusion = FusionEngine(settings, build_providers(settings))
        self.accountant = CostAccountant(settings, store)
        self.scheduler = Scheduler(store, now=now)
        self.news = NewsService(store, settings, fetcher=news_fetcher, now=now)
        self.analyst = DailyAnalyst(store, settings, caller=analysis_caller)
        self.optimizer = (
            StrategyOptimizer(store, population=settings.learning.population,
                              epsilon=settings.learning.epsilon,
                              mutation_scale=settings.learning.mutation_scale)
            if settings.learning.enabled else None
        )
        self._closed_since_evolve = 0
        self._stop = False
        store.set_meta("mode", broker.mode)
        store.set_meta("universe", settings.universe)

    # ------------------------------------------------------------------ helpers
    def _is_trustworthy(self, quote: Quote) -> bool:
        """A quote is trustworthy if its source matches what the configured
        provider promises. When live data is expected (provider != synthetic)
        but MarketData silently fell back to synthetic (network hiccup, rate
        limiting, ...), the quote is a different price regime entirely and
        must never be used to mark or trade an existing/new position — mixing
        a real strike with a synthetic mark (or vice versa) produces
        nonsensical P&L. See test_data_source_safety.py for the incident this
        guards against."""
        if self.settings.data.provider == "synthetic":
            return True
        return quote.source != "synthetic"

    def _positions_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for pos in self.store.positions():
            total += pos.qty * prices.get(pos.symbol, pos.avg_price)
        return total

    def _options_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for o in self.store.open_options():
            spot = prices.get(o["underlying"])
            if spot is None:
                continue
            total += o["contracts"] * mark_option(
                spot, o["strike"], o["expiry_ts"], o["iv"], o["kind"])
        return total

    # ------------------------------------------------------------ research jobs
    def _run_scheduled_jobs(self) -> None:
        news_iv = self.settings.research.news_interval_minutes * 60
        if self.scheduler.due("news", news_iv):
            for symbol in self.settings.universe:
                item = self.news.get(symbol)
                if item:
                    self.bus.activity(
                        "signal", f"news {symbol}: {item['sentiment']}", symbol,
                        {"text": str(item.get("text", ""))[:300]})
        daily_iv = self.settings.research.daily_analysis_interval_hours * 3600
        if self.scheduler.due("daily_analysis", daily_iv):
            result = self.analyst.run()
            if result:
                self.bus.info(f"daily analysis applied: {result['applied']}",
                              source="research")

    # ---------------------------------------------------------------- benchmark
    def _update_benchmark(self, prices: dict[str, float]) -> None:
        sym = self.settings.benchmark.symbol
        price = prices.get(sym)
        if price is None or price <= 0:
            return
        entry = self.store.get_meta("benchmark_entry")
        if entry is None:
            entry = {"symbol": sym, "price": price}
            self.store.set_meta("benchmark_entry", entry)
        bench_equity = self.settings.budget * (price / entry["price"])
        self.store.record_benchmark(bench_equity)
        cash = self.broker.cash()
        bot_equity = cash + self._positions_value(prices) + self._options_value(prices)
        self.store.set_meta("alpha", round(bot_equity - bench_equity, 6))

    # ------------------------------------------------------------ option exits
    def _manage_options(self, prices: dict[str, float], tradeable: set[str]) -> None:
        cfg = self.settings.options
        for o in self.store.open_options():
            underlying = o["underlying"]
            if underlying not in tradeable:
                continue  # no fresh trustworthy quote this cycle — leave untouched
            spot = prices.get(underlying)
            if spot is None:
                continue
            mark = mark_option(spot, o["strike"], o["expiry_ts"], o["iv"], o["kind"])
            entry = max(1e-9, o["entry_premium"])
            change = (mark - entry) / entry
            hours_left = (o["expiry_ts"] - self.now()) / 3600.0
            # Every position carries explicit TP/SL levels set at open time —
            # those govern. Config-derived thresholds are only a fallback for
            # legacy rows persisted before stops were stored per-position.
            tp = o.get("tp_premium") or entry * (1 + cfg.take_profit_pct)
            sl = o.get("sl_premium")
            if sl is None:
                sl = entry * (1 - cfg.stop_loss_pct)
            reason = None
            if mark >= tp:
                reason = f"take-profit hit (mark {mark:.3f} >= TP {tp:.3f}, {change:+.0%})"
            elif mark <= sl:
                reason = f"stop-loss hit (mark {mark:.3f} <= SL {sl:.3f}, {change:+.0%})"
            elif hours_left <= cfg.min_hours_to_expiry:
                reason = f"expiry window ({hours_left:.1f}h left)"
            if reason is None:
                continue
            proceeds = mark * o["contracts"] - OPTION_FEE
            pnl = (mark - entry) * o["contracts"] - OPTION_FEE
            self.broker.adjust_cash(max(0.0, proceeds))
            self.store.record_cost("fee", OPTION_FEE, "options")
            self.store.close_option(o["id"], mark, pnl)
            self.store.record_trade(Trade(
                o["underlying"], "sell", o["contracts"], mark, OPTION_FEE,
                self.broker.mode, f"close {o['kind']} — {reason}", None))
            self.bus.activity(
                "trade",
                f"CLOSE {o['kind'].upper()} {o['underlying']} pnl {pnl:+.3f} — {reason}",
                o["underlying"], {"pnl": pnl})
            self._record_learning(o.get("genome_id"), pnl)

    def _record_learning(self, genome_id: str | None, pnl: float) -> None:
        if not (self.optimizer and genome_id):
            return
        self.optimizer.record_result(genome_id, pnl)
        self._closed_since_evolve += 1
        if self._closed_since_evolve >= self.settings.learning.evolve_every_trades:
            self._closed_since_evolve = 0
            mutant = self.optimizer.evolve()
            if mutant:
                self.bus.activity(
                    "learning",
                    f"evolved: new {mutant.strategy} genome {mutant.id}",
                    detail={"params": mutant.params})

    # ---------------------------------------------------------------- entries
    def _decide(self, quote: Quote) -> tuple[Decision, str | None]:
        extra: list[Signal] = []
        genome_id = None
        if self.optimizer:
            genome = self.optimizer.select()
            d, s = genome.signal(quote.history)
            extra.append(Signal("strategy", d, s,
                                f"{genome.strategy} {genome.params}", 0.0))
            genome_id = genome.id
        ctx: dict = {}
        latest = self.store.latest_news(quote.symbol)
        if latest and (self.now() - latest["ts"]) < NEWS_MAX_AGE_S:
            ctx["research"] = latest["text"]
            sent = latest.get("sentiment")
            if sent in ("bullish", "bearish"):
                extra.append(Signal("news", "buy" if sent == "bullish" else "sell",
                                    0.6, f"news sentiment {sent}", 0.0))
        # Daily market analysis (compiled by the Claude Routine, imported via
        # `lmtrade import-analysis`) — a directional bias per symbol, valid for
        # the window it declares.
        analysis = self.store.get_meta("market_analysis")
        if analysis:
            age = self.now() - float(analysis.get("ts", 0))
            valid_s = float(analysis.get("valid_hours", 24)) * 3600
            entry = (analysis.get("symbols") or {}).get(quote.symbol)
            if age < valid_s and entry and entry.get("bias") in ("bullish", "bearish"):
                direction = "buy" if entry["bias"] == "bullish" else "sell"
                conf = max(0.0, min(1.0, float(entry.get("confidence", 0.5))))
                extra.append(Signal("analysis", direction, conf,
                                    str(entry.get("notes", ""))[:200], 0.0))
        decision = self.fusion.decide(quote, extra_signals=extra, context_extra=ctx)
        self.accountant.record_inference("fusion", decision.inference_cost)
        return decision, genome_id

    def _enter_option(self, quote: Quote, decision: Decision, genome_id: str | None) -> None:
        cfg = self.settings.options
        kind = "call" if decision.direction == "buy" else "put"
        oq = synth_option(quote.symbol, quote.price, quote.history, kind,
                          expiry_days=cfg.expiry_days)
        cash = self.broker.cash()
        equity = cash + self._positions_value({quote.symbol: quote.price}) \
            + self._options_value({quote.symbol: quote.price})
        budget = min(cfg.max_option_fraction * equity * decision.confidence,
                     cash - OPTION_FEE)
        if budget <= 0.05:
            return
        contracts = budget / oq.premium
        cost = contracts * oq.premium + OPTION_FEE
        if not self.broker.adjust_cash(-cost):
            return
        self.store.record_cost("fee", OPTION_FEE, "options")
        # Explicit stops placed with the order: every position always carries
        # its own TP/SL, immune to later config changes.
        tp_premium = oq.premium * (1 + cfg.take_profit_pct)
        sl_premium = oq.premium * (1 - cfg.stop_loss_pct)
        self.store.open_option(quote.symbol, kind, oq.strike, oq.expiry_ts,
                               oq.iv, contracts, oq.premium, genome_id,
                               tp_premium=tp_premium, sl_premium=sl_premium)
        self.store.record_trade(Trade(
            quote.symbol, "buy", contracts, oq.premium, OPTION_FEE,
            self.broker.mode, f"open {kind} K={oq.strike} — {decision.rationale}",
            decision.confidence))
        self.bus.activity(
            "trade",
            f"OPEN {kind.upper()} {quote.symbol} K={oq.strike} ×{contracts:.3f} "
            f"@ {oq.premium:.3f} (IV {oq.iv:.0%})",
            quote.symbol, {"genome": genome_id, "delta": round(oq.delta, 3)})

    def _enter_equity(self, quote: Quote, decision: Decision, econ) -> None:
        pos = self.store.position(quote.symbol)
        if decision.direction == "buy" and not pos:
            sizing = size_position(
                price=quote.price, cash=self.broker.cash(),
                equity=econ.net_worth_eur, confidence=decision.confidence,
                cfg=self.settings.risk)
            if sizing.qty <= 0:
                return
            res = self.broker.buy(quote.symbol, sizing.qty, quote.price)
            if res.ok:
                self.store.record_trade(Trade(
                    quote.symbol, "buy", res.qty, res.price, res.fee,
                    self.broker.mode, decision.rationale, decision.confidence))
                self.bus.activity("trade",
                                  f"BUY {quote.symbol} ×{res.qty:.4f} @ {res.price:.2f}",
                                  quote.symbol)
        elif decision.direction == "sell" and pos:
            res = self.broker.sell(quote.symbol, pos.qty, quote.price)
            if res.ok:
                self.store.record_trade(Trade(
                    quote.symbol, "sell", res.qty, res.price, res.fee,
                    self.broker.mode, decision.rationale, decision.confidence))
                self.bus.activity("trade", f"SELL {quote.symbol} — signal", quote.symbol)

    # ------------------------------------------------------------------- cycle
    def run_cycle(self) -> None:
        self._run_scheduled_jobs()

        symbols = list(dict.fromkeys(
            self.settings.universe + [self.settings.benchmark.symbol]))
        quotes: dict[str, Quote] = {}
        prices: dict[str, float] = {}     # valuation prices: fresh, or last-known-good
        tradeable: set[str] = set()        # symbols with a FRESH trustworthy quote
        degraded: list[str] = []
        for symbol in symbols:
            q = self.market.quote(symbol)
            quotes[symbol] = q
            if self._is_trustworthy(q):
                prices[symbol] = q.price
                self._last_good_price[symbol] = q.price
                tradeable.add(symbol)
            else:
                degraded.append(symbol)
                if symbol in self._last_good_price:
                    prices[symbol] = self._last_good_price[symbol]
        if degraded:
            self.bus.warn(
                f"Live data unavailable this cycle for {degraded} (fell back to "
                "synthetic) — using last known price for valuation only; no "
                "new trades or exits on these symbols this cycle.",
                source="data")

        # exits always run (even when halted), but only for symbols with a
        # fresh trustworthy quote — never mark/close against a stale or
        # mismatched-source price.
        self._manage_options(prices, tradeable)
        for symbol in self.settings.universe:
            if symbol not in tradeable:
                continue
            pos = self.store.position(symbol)
            if pos:
                exit_now, why = should_exit(avg_price=pos.avg_price,
                                            last_price=prices[symbol],
                                            cfg=self.settings.risk)
                if exit_now:
                    res = self.broker.sell(symbol, pos.qty, prices[symbol])
                    if res.ok:
                        self.store.record_trade(Trade(
                            symbol, "sell", res.qty, res.price, res.fee,
                            self.broker.mode, why, None))
                        self.bus.activity("trade", f"SELL {symbol} — {why}", symbol)

        cash = self.broker.cash()
        total_value = self._positions_value(prices) + self._options_value(prices)
        econ = self.accountant.snapshot(cash, total_value)
        self.store.set_meta("economics", econ.as_dict())
        self._update_benchmark(prices)
        self.bus.activity(
            "economics",
            f"net €{econ.net_worth_eur:.2f} | runway {econ.runway_hours:.1f}h | "
            f"alpha {self.store.get_meta('alpha', 0):+0.3f} | "
            f"{'SELF-SUSTAINING' if econ.self_sustaining else 'subsidised'}",
            detail=econ.as_dict())

        if econ.sanity_breached:
            self.bus.error(
                f"SANITY BREACH: net worth €{econ.net_worth_eur:.2f} exceeds "
                f"{self.settings.economics.sanity_max_multiple}x starting budget — "
                "trading halted unconditionally. This indicates a valuation bug, "
                "not a real gain. Investigate before resuming.",
                source="economics")
        elif econ.halt_trading:
            self.bus.warn(
                f"Runway {econ.runway_hours:.1f}h < floor — exits only.",
                source="economics")
        else:
            open_count = len(self.store.open_options()) + len(self.store.positions())
            for symbol in self.settings.universe:
                if open_count >= self.settings.loop.max_positions:
                    break
                if symbol not in tradeable:
                    continue
                if self.store.position(symbol) or any(
                        o["underlying"] == symbol for o in self.store.open_options()):
                    continue
                if not self.accountant.can_afford_inference(econ, est_usd=0.01):
                    break
                decision, genome_id = self._decide(quotes[symbol])
                self.bus.activity(
                    "decision",
                    f"{symbol}: {decision.direction.upper()} conf {decision.confidence:.2f}",
                    symbol, decision.as_dict())
                if decision.direction == "hold" or \
                        decision.confidence < self.settings.risk.min_confidence:
                    continue
                if self.settings.options.enabled:
                    self._enter_option(quotes[symbol], decision, genome_id)
                else:
                    self._enter_equity(quotes[symbol], decision, econ)
                open_count += 1

        cash = self.broker.cash()
        equity = cash + self._positions_value(prices) + self._options_value(prices)
        fees = self.store.total_costs().get("fee", 0.0)
        self.store.record_equity(cash, equity, fees)

    # -------------------------------------------------------------- long loop
    def stop(self) -> None:
        self._stop = True

    def run_forever(self, max_cycles: int | None = None) -> None:
        self.bus.info(
            f"Engine start | mode={self.broker.mode} | universe={self.settings.universe} | "
            f"interval={self.settings.loop.interval_seconds}s | "
            f"options={'on' if self.settings.options.enabled else 'off'} | "
            f"learning={'on' if self.optimizer else 'off'}")
        n = 0
        while not self._stop:
            started = time.time()
            try:
                self.run_cycle()
            except Exception as exc:  # noqa: BLE001 — loop must survive one bad cycle
                self.bus.error(f"cycle error: {exc}")
            n += 1
            if max_cycles and n >= max_cycles:
                self.bus.info(f"Reached max_cycles={max_cycles}, stopping.")
                break
            elapsed = time.time() - started
            time.sleep(max(0.0, self.settings.loop.interval_seconds - elapsed))
