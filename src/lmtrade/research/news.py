"""Hourly market-news service backed by Perplexity.

Fetches web-grounded news + sentiment per instrument at most once per
configured interval (default hourly), caches into the Store's news table, and
serves the cache in between. The fetcher is injectable so tests run offline;
by default it uses the PerplexityProvider (which degrades to no-op without a
key)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..config import Settings, secret
from ..core.state import Store
from ..models.providers import ClaudeCLIProvider, PerplexityProvider

# fetcher(symbol) -> (text, cost_usd)
Fetcher = Callable[[str], tuple[str, float]]


def _is_valid_news(text: str) -> bool:
    """A genuine research response carries the SENTIMENT marker its prompt
    demands. Anything without it is broken (CLI/shell noise) and must not be
    cached/served as news."""
    return bool(text) and "sentiment" in text.lower()


def _parse_sentiment(text: str) -> str:
    low = (text or "").lower()
    if "sentiment: bullish" in low or ("bullish" in low and "bearish" not in low):
        return "bullish"
    if "sentiment: bearish" in low or ("bearish" in low and "bullish" not in low):
        return "bearish"
    return "neutral"


class NewsService:
    def __init__(
        self, store: Store, settings: Settings,
        fetcher: Fetcher | None = None, now: Callable[[], float] = time.time,
    ):
        self.store = store
        self.settings = settings
        self.now = now
        if fetcher is not None:
            self.fetcher: Fetcher | None = fetcher
        elif settings.research.news_provider == "claude_cli":
            cli = ClaudeCLIProvider(settings)
            self.fetcher = cli.research if cli.available() else None
        elif secret("PERPLEXITY_API_KEY"):
            provider = PerplexityProvider(settings)
            self.fetcher = provider.research
        else:
            self.fetcher = None

    def get(self, symbol: str) -> dict | None:
        """Return {text, sentiment, ts} for the symbol — cached if fresh AND
        valid, refetched if stale/missing/corrupt, None if unavailable.

        Corrupt cache guard: a real research response always carries the
        SENTIMENT marker its prompt demands. An entry lacking it is broken
        (e.g. CLI/shell error output stored as "neutral" by an older build) —
        we never serve it and refetch instead, so bad neutrals self-heal
        rather than persisting for the whole freshness window."""
        max_age = self.settings.research.news_interval_minutes * 60
        cached = self.store.latest_news(symbol)
        cached_ok = bool(cached and _is_valid_news(cached.get("text", "")))
        if cached_ok and (self.now() - cached["ts"]) < max_age:
            return cached
        fallback = cached if cached_ok else None   # never fall back to garbage
        if self.fetcher is None:
            return fallback
        try:
            text, cost = self.fetcher(symbol)
        except Exception:  # noqa: BLE001 — news must never take down the engine
            return fallback
        if not text:
            return fallback
        if cost > 0:
            self.store.record_cost("inference", cost, "perplexity")
        sentiment = _parse_sentiment(text)
        self.store.add_news(symbol, text, sentiment, ts=self.now())
        return {"symbol": symbol, "text": text, "sentiment": sentiment, "ts": self.now()}

    def get_many(
        self, symbols: list[str], max_workers: int = 8,
        on_progress: Callable[[str, dict | None, int, int], None] | None = None,
    ) -> dict[str, dict | None]:
        """Fetch news for many symbols in parallel, bounded by max_workers —
        a sequential per-symbol loop is impractical once the fetcher has real
        per-call latency (e.g. a claude CLI subprocess doing a web search)
        across a large universe. A failing symbol resolves to whatever get()
        would have returned (cache or None) without affecting the others.

        `on_progress(symbol, item, done, total)` is invoked (in the calling
        thread, so it's safe to log/write from) as each symbol completes, so
        callers can show live progress instead of the fetch looking hung for
        minutes. The returned dict preserves the input symbol order."""
        if not symbols:
            return {}
        total = len(symbols)
        results: dict[str, dict | None] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, total))) as pool:
            futures = {pool.submit(self.get, s): s for s in symbols}
            done = 0
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    results[symbol] = fut.result()
                except Exception:  # noqa: BLE001 — one symbol must not sink the batch
                    results[symbol] = None
                done += 1
                if on_progress is not None:
                    on_progress(symbol, results[symbol], done, total)
        return {s: results.get(s) for s in symbols}
