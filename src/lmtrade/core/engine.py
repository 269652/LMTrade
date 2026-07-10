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
from ..finance.knockouts import (
    is_knocked_out,
    knockout_price,
    strike_after_financing,
)
from ..finance.options import mark_option
from ..finance.risk import should_exit, size_position
from ..finance.sizing import kelly_fraction, regime, vol_scale
from ..models.base import Signal
from ..models.providers import build_providers
from ..research.daily import DailyAnalyst
from ..research.news import NewsService
from ..strategies.optimizer import StrategyOptimizer
from .control import LOW_BALANCE_EUR
from .events import EventBus
from .scheduler import Scheduler
from .state import Store, Trade

OPTION_FEE = 0.1   # per option order (paper): smaller than TR's equity fee,
                   # comparable to warrant spreads on tiny notionals
NEWS_MAX_AGE_S = 24 * 3600   # news older than this no longer influences decisions
ANALYSIS_RETRY_S = 15 * 60   # when analysis is missing/unusable, retry this often
                             # (not every cycle) so a provider outage doesn't hammer


class _SharedResearchStore:
    """Wraps the active book's store so NEWS and the compiled MARKET ANALYSIS
    are read/written through BOTH the paper and live books — they are
    process-wide research, not book-specific trading state, and fetching them
    twice (once per book, whenever each becomes 'active') wastes real web-
    search/LLM calls. Everything else (trades, costs, activity) passes
    through to the active book only: a single real API call must be billed
    once, not mirrored as if it happened twice."""

    _SHARED_META_KEYS = ("market_analysis", "provider_warnings")

    def __init__(self, primary: Store, secondary: Store | None):
        self._primary = primary
        self._secondary = secondary

    def __getattr__(self, name):
        return getattr(self._primary, name)

    def latest_news(self, symbol: str):
        row = self._primary.latest_news(symbol)
        if row is not None:
            return row
        return self._secondary.latest_news(symbol) if self._secondary else None

    def add_news(self, symbol: str, text: str, sentiment: str, ts=None) -> None:
        self._primary.add_news(symbol, text, sentiment, ts=ts)
        if self._secondary is not None:
            self._secondary.add_news(symbol, text, sentiment, ts=ts)

    def get_meta(self, key: str, default=None):
        if key not in self._SHARED_META_KEYS:
            return self._primary.get_meta(key, default)
        v = self._primary.get_meta(key, None)
        if v is None and self._secondary is not None:
            v = self._secondary.get_meta(key, None)
        return v if v is not None else default

    def set_meta(self, key: str, value) -> None:
        self._primary.set_meta(key, value)
        if key in self._SHARED_META_KEYS and self._secondary is not None:
            self._secondary.set_meta(key, value)


class Engine:
    def __init__(
        self, settings: Settings, store: Store, broker: Broker,
        news_fetcher=None, analysis_caller=None, market=None,
        tr_derivatives="auto",
        now: Callable[[], float] = time.time,
        fallback_store: Store | None = None,
    ):
        self.settings = settings
        self.store = store
        self.broker = broker
        self.fallback_store = fallback_store  # paper store for fallback analysis in live mode
        self.now = now
        self.bus = EventBus(store)
        self.market = market or MarketData(settings.data.provider,
                                           intraday=settings.data.intraday)
        if tr_derivatives == "auto":
            # Auto-build from settings + env; returns None without TR creds,
            # keeping TR login strictly optional. Pass tr_derivatives=None
            # explicitly to force the synthetic-options path.
            from ..brokers.tr_derivatives import build_tr_derivatives

            tr_derivatives = build_tr_derivatives(settings)
        self.tr_derivatives = tr_derivatives
        self._last_good_price: dict[str, float] = {}
        self.fusion = FusionEngine(settings, build_providers(settings))
        self.accountant = CostAccountant(settings, store)
        self.scheduler = Scheduler(store, now=now)
        research_store = _SharedResearchStore(store, fallback_store)
        self.news = NewsService(research_store, settings, fetcher=news_fetcher, now=now)
        self.analyst = DailyAnalyst(research_store, settings, caller=analysis_caller)
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

    def _mark_position(self, o: dict, spot: float) -> float:
        """Current per-unit mark for an open position, by instrument type."""
        if (o.get("instrument_type") or "option") == "knockout":
            if is_knocked_out(spot, o["barrier"], o["kind"]):
                return 0.0
            days_held = max(0.0, (self.now() - o["opened_ts"]) / 86400.0)
            eff_strike = strike_after_financing(o["strike"], o["kind"], days_held)
            return knockout_price(spot, eff_strike, o["ratio"] or 1.0, o["kind"])
        return mark_option(spot, o["strike"], o["expiry_ts"], o["iv"], o["kind"])

    def _options_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for o in self.store.open_options():
            spot = prices.get(o["underlying"])
            if spot is None:
                continue
            total += o["contracts"] * self._mark_position(o, spot)
        return total

    def _persist_option_marks(self, prices: dict[str, float]) -> None:
        """Store the current mark + unrealized P&L for each open option so the
        dashboard can show live profit/loss (it's read-only over the Store and
        has no market feed of its own). Keyed by option id as a string."""
        self._persist_option_marks_for(self.store, prices)

    def _persist_option_marks_for(self, store: Store, prices: dict[str, float]) -> None:
        marks: dict[str, dict] = {}
        for o in store.open_options():
            spot = prices.get(o["underlying"])
            # Fall back to last-known price if no fresh quote this cycle
            if spot is None:
                spot = self._last_good_price.get(o["underlying"])
            if spot is None:
                continue
            mark = self._mark_position(o, spot)
            value = o["contracts"] * mark
            cost = o["contracts"] * o["entry_premium"]
            marks[str(o["id"])] = {
                "mark_premium": round(mark, 4),
                "value": round(value, 4),
                "unrealized_pnl": round(value - cost, 4),
                "spot": round(spot, 4),
            }
        store.set_meta("open_option_marks", marks)

    def _value_fallback_book(self, prices: dict[str, float], tr_cash: float | None) -> None:
        """Keep the INACTIVE book (paper<->live) valued too, using this
        cycle's already-fetched prices — no extra API calls. The dashboard's
        live view must always show accurate cash/net worth even while the
        engine is trading the OTHER book (e.g. live view before arming: the
        engine simulates on paper, but real TR positions synced into the live
        book still need fresh marks to price correctly)."""
        fb = self.fallback_store
        if fb is None:
            return
        self._persist_option_marks_for(fb, prices)
        if tr_cash is not None:
            fb.set_meta("tr_account_cash", tr_cash)
            if fb.get_meta("tr_baseline_net_worth") is None:
                fb.set_meta("tr_baseline_net_worth", tr_cash)

    def _get_analysis_with_fallback(self) -> dict | None:
        """Get daily market analysis from store, falling back to paper store if live analysis
        is missing (useful when live mode doesn't have analysis but paper run does)."""
        analysis = self.store.get_meta("market_analysis")
        if analysis is None and self.fallback_store is not None:
            analysis = self.fallback_store.get_meta("market_analysis")
        return analysis

    def _analysis_current(self, analysis: dict | None) -> bool:
        """True when a stored market analysis exists and is still inside its own
        validity window (ts + valid_hours). Used to reuse a fresh analysis
        across restarts instead of rebuilding it every launch. A malformed or
        timestamp-less analysis counts as not current so it gets refreshed."""
        if not isinstance(analysis, dict):
            return False
        ts = analysis.get("ts")
        if ts is None:
            return False
        valid_h = analysis.get(
            "valid_hours", self.settings.research.daily_analysis_interval_hours)
        try:
            return (self.now() - float(ts)) < float(valid_h) * 3600
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------ research jobs
    def _run_scheduled_jobs(self) -> None:
        # The news pass fires on the (shorter) retry cadence; get()/get_many
        # only actually refetch a symbol whose cache is missing or older than
        # news_interval_minutes, so healthy symbols aren't re-fetched — only
        # failed/corrupted ones retry sooner (default every 10 min).
        news_iv = min(self.settings.research.news_interval_minutes,
                      self.settings.research.news_retry_minutes) * 60
        if self.scheduler.due("news", news_iv):
            universe = self.settings.universe
            provider = self.settings.research.news_provider
            self.bus.info(
                f"[research] fetching news for {len(universe)} symbols "
                f"(provider={provider}) — this can take a few minutes with a "
                f"live web-search provider...", source="research")

            def _news_progress(symbol: str, item: dict | None, done: int, total: int) -> None:
                # Fired as each symbol completes (in the calling thread), so
                # the run shows live progress instead of appearing hung.
                if item:
                    self.bus.activity(
                        "signal", f"news {done}/{total} {symbol}: {item['sentiment']}",
                        symbol, {"text": str(item.get("text", ""))[:300]})
                else:
                    self.bus.info(f"[research] news {done}/{total} {symbol}: no data",
                                  source="research")

            self.news.get_many(universe, max_workers=min(8, len(universe)),
                               on_progress=_news_progress)
            self.bus.info("[research] news fetch complete", source="research")
        daily_iv = self.settings.research.daily_analysis_interval_hours * 3600
        # Reuse an up-to-date analysis already in the DB instead of rebuilding
        # it on every (re)start: a stored analysis is current while it's inside
        # its own validity window (ts + valid_hours). Only when it's missing or
        # stale do we (re)compile — gated by the scheduler so a provider outage
        # retries periodically rather than every cycle. When it IS current we
        # still advance the daily stamp so a fresh scheduler (post-restart)
        # doesn't consider it overdue and thrash.
        existing = self.store.get_meta("market_analysis")
        if self._analysis_current(existing):
            self.scheduler.due("daily_analysis", daily_iv)
        elif self.scheduler.due("daily_analysis", ANALYSIS_RETRY_S):
            provider = self.settings.research.analysis_provider
            self.bus.info(
                f"[research] compiling daily market analysis (provider={provider}) "
                f"for {len(self.settings.universe)} symbols...", source="research")
            analysis = self.analyst.compile_analysis(self.settings.universe, self.now())
            if analysis:
                self.bus.info(
                    f"[research] daily analysis compiled: {len(analysis['symbols'])} "
                    f"symbols", source="research")
            else:
                self.bus.warn("[research] daily analysis could not be compiled "
                              "(provider unavailable or empty) — will retry.",
                              source="research")
            # The risk-parameter review is a separate, best-effort step.
            result = self.analyst.run()
            if result:
                self.bus.info(f"[research] risk review applied: {result['applied']}",
                              source="research")

    def _update_realized_mark(self, net_worth: float) -> None:
        """Maintain `last_realized_net_worth`: the net worth locked in at the
        most recent REALIZED (closed) trade. The dashboard colors net worth
        red while the live mark-to-market value sits below this, green at or
        above — a realized high-water mark that resets each time a position
        closes (win or loss)."""
        if self.store.get_meta("last_realized_net_worth") is None:
            self.store.set_meta(
                "last_realized_net_worth",
                float(self.store.get_meta("starting_cash", self.settings.budget)))
        closed = self.store.closed_options_count()
        if closed > int(self.store.get_meta("realized_close_count", 0)):
            # A trade just realized — reset the mark to the current net worth.
            self.store.set_meta("last_realized_net_worth", round(net_worth, 6))
            self.store.set_meta("realized_close_count", closed)

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
            is_ko = (o.get("instrument_type") or "option") == "knockout"
            mark = self._mark_position(o, spot)
            entry = max(1e-9, o["entry_premium"])
            change = (mark - entry) / entry
            hours_left = (o["expiry_ts"] - self.now()) / 3600.0
            hours_held = (self.now() - o["opened_ts"]) / 3600.0
            # Every position carries explicit TP/SL levels set at open time —
            # those govern. Config-derived thresholds are only a fallback for
            # legacy rows persisted before stops were stored per-position.
            tp = o.get("tp_premium") or entry * (1 + cfg.take_profit_pct)
            sl = o.get("sl_premium")
            if sl is None:
                sl = entry * (1 - cfg.stop_loss_pct)

            expiry_hit = hours_left <= cfg.min_hours_to_expiry
            held_long_enough = hours_held >= cfg.min_hold_hours
            reason = None
            if is_ko and is_knocked_out(spot, o["barrier"], o["kind"]):
                # A knockout is an involuntary event: the certificate IS dead
                # the moment the barrier is touched — overrides every gate.
                mark = 0.0
                change = -1.0
                reason = (f"KNOCKED OUT (spot {spot:.2f} touched barrier "
                          f"{o['barrier']:.2f}) — total loss of premium")
            elif expiry_hit:
                # A hard constraint of the option itself — overrides min_hold.
                reason = f"expiry window ({hours_left:.1f}h left)"
            elif cfg.max_hold_hours > 0 and hours_held >= cfg.max_hold_hours:
                reason = f"max hold time reached ({hours_held:.1f}h)"
            elif held_long_enough and mark >= tp:
                reason = f"take-profit hit (mark {mark:.3f} >= TP {tp:.3f}, {change:+.0%})"
            elif held_long_enough and mark <= sl:
                reason = f"stop-loss hit (mark {mark:.3f} <= SL {sl:.3f}, {change:+.0%})"
            if reason is None:
                continue
            proceeds = mark * o["contracts"] - OPTION_FEE
            pnl = (mark - entry) * o["contracts"] - OPTION_FEE
            self.broker.adjust_cash(max(0.0, proceeds))
            self.store.record_cost("fee", OPTION_FEE, "options")
            self.store.close_option(o["id"], mark, pnl)

            # Profit stash: a configured fraction of realized PROFIT (never
            # losses) is moved out of tradeable cash into a reserve. Still
            # counted in net worth/alpha — just protected from being re-risked.
            stash_pct = self.settings.economics.profit_stash_pct
            if pnl > 0 and stash_pct > 0:
                stash = pnl * stash_pct
                if self.broker.adjust_cash(-stash):
                    self.store.add_reserve(stash)

            self.store.record_trade(Trade(
                o["underlying"], "sell", o["contracts"], mark, OPTION_FEE,
                self.broker.mode, f"close {o['kind']} — {reason}", None))
            self.bus.activity(
                "trade",
                f"CLOSE {o['kind'].upper()} {o['underlying']} pnl {pnl:+.3f} — {reason}",
                o["underlying"], {"pnl": pnl})
            self._record_learning(o.get("genome_id"), pnl)

        # Stale eviction: close positions that have been sideways for too long
        # to free a slot for a stronger incoming signal.  Runs after the normal
        # TP/SL/expiry loop so already-closed options are never double-processed
        # (close_option removes them from open_options()).  Only fires on
        # positions older than stale_hours with pnl below stale_max_profit_pct —
        # winners are always held; the normal SL cuts big losers; only dead-
        # weight "stuck" positions are cleaned up here.
        if self.settings.loop.stale_evict_enabled:
            stale_h = self.settings.loop.stale_hours
            max_profit = self.settings.loop.stale_max_profit_pct
            now = self.now()
            for o in list(self.store.open_options()):
                spot = prices.get(o["underlying"])
                if spot is None:
                    continue
                age_h = (now - o["opened_ts"]) / 3600.0
                if age_h < stale_h:
                    continue
                entry = max(1e-9, float(o["entry_premium"]))
                mark = self._mark_position(o, spot)
                pnl_pct = (mark - entry) / entry
                if pnl_pct >= max_profit:
                    continue   # winning position — don't evict
                contracts = float(o["contracts"])
                pnl = (mark - entry) * contracts - OPTION_FEE
                proceeds = max(0.0, mark * contracts - OPTION_FEE)
                self.broker.adjust_cash(proceeds)
                self.store.record_cost("fee", OPTION_FEE, "options")
                self.store.close_option(o["id"], mark, pnl)
                stash_pct = self.settings.economics.profit_stash_pct
                if pnl > 0 and stash_pct > 0:
                    stash = pnl * stash_pct
                    if self.broker.adjust_cash(-stash):
                        self.store.add_reserve(stash)
                self.store.record_trade(Trade(
                    o["underlying"], "sell", contracts, mark, OPTION_FEE,
                    self.broker.mode,
                    f"close {o['kind']} — stale ({age_h:.0f}h, {pnl_pct:+.1%}): "
                    "freeing slot for stronger signal",
                    None))
                self.bus.activity(
                    "trade",
                    f"STALE CLOSE {o['kind'].upper()} {o['underlying']} "
                    f"age={age_h:.0f}h pnl={pnl_pct:+.1%} → freed slot",
                    o["underlying"], {"pnl": round(pnl, 4), "stale": True})
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
            # Market context for cross-sectional families (xsmom, rel_value):
            # the whole cycle's histories, set by run_cycle. Kept on self (not
            # a parameter) so tests stubbing _decide(quote) stay valid.
            market_ctx = {
                "symbol": quote.symbol,
                "histories": getattr(self, "_cycle_histories", {}),
                "benchmark": self.settings.benchmark.symbol,
            }
            d, s = genome.signal(quote.history, market_ctx)
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
        # the window it declares. Falls back to paper store if live analysis
        # is missing (e.g. Claude reached limits but paper run succeeded).
        analysis = self._get_analysis_with_fallback()
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

    def _genome_stats(self, genome_id: str | None):
        if not (self.optimizer and genome_id):
            return None
        for g in self.optimizer.genomes():
            if g.id == genome_id:
                return g
        return None

    def _position_budget(self, quote: Quote, decision: Decision,
                         genome_id: str | None) -> float:
        """Premium budget for a new position, layering the quant sizing rules:
        fractional Kelly from the genome's empirical edge (when proven),
        volatility targeting, and the storm-regime haircut. Falls back to the
        plain confidence-scaled fraction when there isn't enough data.

        The cash reserve cap (options.cash_reserve_pct) further limits the
        budget: at most (1 - reserve_pct) of current equity may be deployed in
        open positions at any time. As the portfolio grows the deployable bucket
        grows proportionally, so the engine automatically uses larger positions
        after profitable runs without any manual parameter changes."""
        cfg = self.settings.options
        s = self.settings.sizing
        cash = self.broker.cash()
        # Use all last-known prices for a portfolio-wide equity estimate,
        # not just the current quote — options on OTHER symbols have value too.
        all_prices = {**self._last_good_price, quote.symbol: quote.price}
        positions_value = self._positions_value(all_prices)
        options_value = self._options_value(all_prices)
        equity = cash + positions_value + options_value

        fraction = cfg.max_option_fraction * decision.confidence
        g = self._genome_stats(genome_id)
        if (s.kelly_enabled and g is not None
                and (g.wins + g.losses) >= s.kelly_min_trades):
            kf = kelly_fraction(g.win_rate, g.avg_win, g.avg_loss) \
                * s.kelly_fraction_of_full
            fraction = min(cfg.max_option_fraction, kf)
        fraction *= vol_scale(quote.history, s.vol_target_annual)
        if s.regime_filter_enabled and regime(quote.history) == "storm":
            fraction *= s.storm_size_factor

        # Cash reserve: cap the budget so total deployed capital never exceeds
        # (1 - cash_reserve_pct) * equity.  Already-deployed capital is
        # subtracted first; the remainder is available for this new position.
        deployed = positions_value + options_value
        max_deployable = equity * (1.0 - cfg.cash_reserve_pct)
        available = max(0.0, max_deployable - deployed)

        return min(fraction * equity, available, cash - OPTION_FEE)

    def _enter_knockout(self, quote: Quote, decision: Decision,
                        genome_id: str | None) -> bool:
        """Open a real-ISIN TR knockout (paper: simulated fill; armed live: a
        real order). Returns False when no TR client / no suitable instrument
        — the caller skips this cycle rather than fabricate a synthetic one."""
        tr = self.tr_derivatives
        if tr is None or not tr.available():
            return False
        ko = tr.find_knockout(quote.symbol, decision.direction, quote.price,
                              self.settings.tr.target_leverage)
        if ko is None or ko.price <= 0:
            return False
        cfg = self.settings.options
        budget = self._position_budget(quote, decision, genome_id)
        if budget <= 0.05:
            return True   # handled (deliberately no trade), don't fall back
        contracts = budget / ko.price

        # Real execution: when the broker is an ARMED live broker, place a
        # REAL market order for the knockout certificate. Opening a long
        # knockout (call or put variant) is always a BUY of the certificate.
        # Real fills happen at TR; we don't touch local simulated cash (the
        # live view's balances come from the real TR account).
        from .control import ControlState
        
        control = ControlState.load(self.settings.control_path)
        broker_armed = getattr(self.broker, "armed", False)
        has_place_order = hasattr(self.broker, "place_order")
        # The broker is only ever constructed armed when the control plane is
        # live+armed (see cli._build_engine_for_control), so an armed broker IS
        # the routing signal for real execution. The runtime low-balance guard
        # below is re-checked against the live balance on every order.
        armed_real = broker_armed and has_place_order

        if armed_real:
            # Runtime low-balance guard (second arming switch): below
            # LOW_BALANCE_EUR the flat ~1 EUR fee is a >1% drag, so real orders
            # are blocked unless the user has explicitly armed the second guard.
            # Checked live against the REAL account balance every order, so it
            # engages the moment net worth drops under the threshold.
            net_worth = self.broker.cash()
            if control.low_balance_blocks(net_worth):
                self.bus.warn(
                    f"LIVE order blocked [{ko.isin}]: balance "
                    f"{'unknown' if net_worth is None else f'€{net_worth:.2f}'} "
                    f"is under €{LOW_BALANCE_EUR:.0f} and the low-balance guard "
                    f"is not armed.", source="engine")
                return True
            size = int(contracts)   # whole certificates (sellFractions off)
            if size < 1:
                self.bus.warn(
                    f"LIVE knockout {ko.isin}: budget too small for one "
                    f"certificate at {ko.price:.2f} — skipping.", source="engine")
                return True
            res = self.broker.place_order(ko.isin, "buy", float(size))
            if not res.ok:
                self.bus.warn(f"LIVE knockout order rejected [{ko.isin}]: "
                              f"{res.message}", source="engine")
                return True   # never fall back to synthetic once live-armed
            # Real fill: cash is the REAL TR account (no local ledger to debit).
            # The live view's balances come from TR; we only record the fee for
            # cost accounting and then book the position locally.
            contracts = float(size)
            self.store.record_cost("fee", OPTION_FEE, "options")
        else:
            cost = contracts * ko.price + OPTION_FEE
            if not self.broker.adjust_cash(-cost):
                return True
            self.store.record_cost("fee", OPTION_FEE, "options")
        tp_premium = ko.price * (1 + cfg.take_profit_pct)
        sl_premium = ko.price * (1 - cfg.stop_loss_pct)
        # KOs are open-ended: expiry far out; the barrier is the real risk.
        expiry_ts = self.now() + 365 * 86400.0
        self.store.open_option(
            quote.symbol, ko.kind, ko.strike, expiry_ts, 0.0, contracts,
            ko.price, genome_id, tp_premium=tp_premium, sl_premium=sl_premium,
            instrument_type="knockout", barrier=ko.barrier, ratio=ko.ratio,
            isin=ko.isin)
        self.store.record_trade(Trade(
            quote.symbol, "buy", contracts, ko.price, OPTION_FEE,
            self.broker.mode,
            f"open {ko.kind} {ko.isin or 'synthetic'} K={ko.strike} "
            f"barrier={ko.barrier} {ko.leverage:.1f}x — {decision.rationale}",
            decision.confidence))
        self.bus.activity(
            "trade",
            f"OPEN {ko.kind.upper()} {quote.symbol} [{ko.isin or 'synthetic'}] "
            f"K={ko.strike} barrier={ko.barrier} {ko.leverage:.1f}x "
            f"×{contracts:.3f} @ {ko.price:.3f}",
            quote.symbol, {"genome": genome_id, "isin": ko.isin,
                           "leverage": ko.leverage})
        return True

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
        # Concurrent, bounded fetch — a large universe fetched sequentially
        # would make cycle time grow linearly with universe size. The cap
        # also keeps request bursts against Yahoo modest (see
        # data/market.py: this is what previously triggered rate limiting).
        if hasattr(self.market, "quotes_concurrent"):
            quotes: dict[str, Quote] = self.market.quotes_concurrent(
                symbols, max_workers=min(8, len(symbols)))
        else:
            quotes = {s: self.market.quote(s) for s in symbols}
        # Cycle-wide histories for cross-sectional strategies (see _decide).
        self._cycle_histories = {s: q.history for s, q in quotes.items()}
        prices: dict[str, float] = {}     # valuation prices: fresh, or last-known-good
        tradeable: set[str] = set()        # symbols with a FRESH trustworthy quote
        degraded: list[str] = []
        for symbol, q in quotes.items():
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
        self._persist_option_marks(prices)
        # Live TR account cash for the dashboard, refreshed at most every few
        # minutes rather than every cycle — each fetch opens a websocket, so
        # doing it per-cycle hammers TR (and multiplies any 401). None-safe:
        # no-op without a real authenticated TR client.
        tr_cash = None
        if self.tr_derivatives is not None and self.scheduler.due("tr_cash", 300):
            tr_cash = self.tr_derivatives.account_cash()
            if tr_cash is not None:
                self.store.set_meta("tr_account_cash", tr_cash)
                # First real balance we ever see becomes the live P&L baseline
                # (positions are ~empty at that point), so live P&L reflects the
                # change since going live rather than a meaningless delta vs the
                # paper starting budget.
                if self.store.get_meta("tr_baseline_net_worth") is None:
                    self.store.set_meta("tr_baseline_net_worth", tr_cash)
        # Keep the OTHER book valued too — the dashboard's live view must
        # always show accurate cash/net worth even while this engine is
        # actively trading the other book (e.g. unarmed live: paper trades,
        # but real TR positions synced into the live book still need marks).
        self._value_fallback_book(prices, tr_cash)
        econ = self.accountant.snapshot(cash, total_value, self.store.reserve_balance())
        self.store.set_meta("economics", econ.as_dict())
        self._update_realized_mark(econ.net_worth_eur)
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
            slots = self.settings.loop.max_positions - open_count
            if slots <= 0:
                # Book full — the decision/entry step is skipped this cycle.
                # Say so, otherwise the run looks stuck (only economics lines)
                # when it's actually just holding a full position book.
                self.bus.info(
                    f"position book full ({open_count}/{self.settings.loop.max_positions}) "
                    f"— no free slots, holding existing positions this cycle",
                    source="engine")
            candidates: list[tuple[Decision, str | None]] = []
            for symbol in self.settings.universe:
                if slots <= 0:
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
                # Storm regime: demand extra conviction — measured edges are
                # the first casualty when the volatility regime flips.
                min_conf = self.settings.risk.min_confidence
                if (self.settings.sizing.regime_filter_enabled
                        and regime(quotes[symbol].history) == "storm"):
                    min_conf += self.settings.sizing.storm_extra_confidence
                if decision.direction == "hold" or decision.confidence < min_conf:
                    continue
                candidates.append((decision, genome_id))

            # Rank by confidence so limited slots go to the strongest signals
            # across the whole universe, not just whichever symbols happened
            # to come first in the list.
            candidates.sort(key=lambda c: c[0].confidence, reverse=True)
            take = max(0, slots)
            cap = self.settings.loop.max_new_positions_per_cycle
            if cap > 0:
                take = min(take, cap)
            for decision, genome_id in candidates[:take]:
                quote = quotes[decision.symbol]
                if self.settings.options.enabled:
                    # ONLY real TR knockout instruments are ever traded — in
                    # paper AND live — real ISIN/strike/barrier/price from the
                    # TR catalog. Paper mode simulates the fill locally; live
                    # mode places a real order. There is no synthetic
                    # Black-Scholes fallback in either mode: a fabricated local
                    # position (no TR client, session down, or no suitable
                    # instrument) is a phantom the TR app never sees.
                    self.bus.info(
                        f"[execution] entering {decision.symbol} {decision.direction} "
                        f"in {self.broker.mode} mode (broker.armed={getattr(self.broker, 'armed', False)})",
                        source="engine")
                    if not self._enter_knockout(quote, decision, genome_id):
                        self.bus.warn(
                            f"{decision.symbol}: no real TR knockout tradeable "
                            f"this cycle — skipping (synthetic instruments are "
                            f"disabled).", source="engine")
                else:
                    self._enter_equity(quote, decision, econ)

        cash = self.broker.cash()
        equity = cash + self._positions_value(prices) + self._options_value(prices)
        fees = self.store.total_costs().get("fee", 0.0)
        self.store.record_equity(cash, equity, fees)

    # -------------------------------------------------------------- long loop
    def stop(self) -> None:
        self._stop = True

    def run_forever(self, max_cycles: int | None = None,
                    rebuild_when=None) -> None:
        """Run cycles until stopped or max_cycles. `rebuild_when()` is checked
        after each cycle; when it returns True the loop returns so the caller
        can rebuild the engine for a new book/broker (the paper/live toggle
        or arm/disarm changed). Returning — not raising — keeps `finally`
        cleanup in the CLI simple."""
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
            if rebuild_when is not None and rebuild_when():
                self.bus.info("Control changed (paper/live/armed) — reconfiguring.")
                break
            elapsed = time.time() - started
            time.sleep(max(0.0, self.settings.loop.interval_seconds - elapsed))
