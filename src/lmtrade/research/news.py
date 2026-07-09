"""Hourly market-news service backed by Perplexity.

Fetches web-grounded news + sentiment per instrument at most once per
configured interval (default hourly), caches into the Store's news table, and
serves the cache in between. The fetcher is injectable so tests run offline;
by default it uses the PerplexityProvider (which degrades to no-op without a
key)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..config import Settings, secret
from ..core.state import Store
from ..models.providers import ClaudeCLIProvider, PerplexityProvider

# fetcher(symbol) -> (text, cost_usd)
Fetcher = Callable[[str], tuple[str, float]]


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
        """Return {text, sentiment, ts} for the symbol — cached if fresh,
        refetched if stale, None if news is unavailable entirely."""
        max_age = self.settings.research.news_interval_minutes * 60
        cached = self.store.latest_news(symbol)
        if cached and (self.now() - cached["ts"]) < max_age:
            return cached
        if self.fetcher is None:
            return cached  # possibly None: no source at all
        try:
            text, cost = self.fetcher(symbol)
        except Exception:  # noqa: BLE001 — news must never take down the engine
            return cached
        if not text:
            return cached
        if cost > 0:
            self.store.record_cost("inference", cost, "perplexity")
        sentiment = _parse_sentiment(text)
        self.store.add_news(symbol, text, sentiment, ts=self.now())
        return {"symbol": symbol, "text": text, "sentiment": sentiment, "ts": self.now()}

    def get_many(self, symbols: list[str], max_workers: int = 8) -> dict[str, dict | None]:
        """Fetch news for many symbols in parallel, bounded by max_workers —
        a sequential per-symbol loop is impractical once the fetcher has real
        per-call latency (e.g. a claude CLI subprocess doing a web search)
        across a large universe. A failing symbol resolves to whatever get()
        would have returned (cache or None) without affecting the others."""
        if not symbols:
            return {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(symbols)))) as pool:
            results = list(pool.map(self.get, symbols))
        return dict(zip(symbols, results))
