"""Market data providers.

`auto` uses yfinance when available and falls back to a deterministic synthetic
random-walk generator, so the bot always runs — offline, in CI, or on a fresh
Vast.ai box before keys are set.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass


@dataclass
class Quote:
    symbol: str
    price: float
    history: list[float]      # recent close prices, oldest -> newest
    source: str


class MarketData:
    def __init__(self, provider: str = "auto", lookback: int = 60, intraday: bool = False):
        self.lookback = lookback
        self.intraday = intraday
        self.provider = self._resolve(provider)

    def _resolve(self, provider: str) -> str:
        if provider == "synthetic":
            return "synthetic"
        try:
            import yfinance  # noqa: F401

            return "yfinance"
        except Exception:
            return "synthetic"

    def quote(self, symbol: str) -> Quote:
        if self.provider == "yfinance":
            try:
                return self._yf_quote(symbol)
            except Exception:
                pass  # fall through to synthetic on any data hiccup
        return self._synthetic_quote(symbol)

    # -- yfinance -------------------------------------------------------------
    def _yf_quote(self, symbol: str) -> Quote:
        import yfinance as yf

        if self.intraday:
            hist = yf.Ticker(symbol).history(period="1d", interval="1m")
            if hist.empty:  # market closed / no intraday bars -> daily fallback
                hist = yf.Ticker(symbol).history(period="3mo", interval="1d")
        else:
            hist = yf.Ticker(symbol).history(period="3mo", interval="1d")
        closes = [float(x) for x in hist["Close"].dropna().tolist()][-self.lookback:]
        if not closes:
            raise ValueError("no data")
        return Quote(symbol, closes[-1], closes, "yfinance")

    # -- synthetic ------------------------------------------------------------
    def _synthetic_quote(self, symbol: str) -> Quote:
        """Deterministic-ish random walk seeded by symbol + coarse time, so the
        series evolves between loops but is reproducible within a bucket
        (10s buckets intraday, 60s otherwise)."""
        seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % 10_000
        base = 50 + (seed % 200)
        bucket = 10 if self.intraday else 60
        t0 = int(time.time() // bucket)
        closes: list[float] = []
        price = float(base)
        for i in range(self.lookback):
            # pseudo-random step from a hash of (symbol, tick)
            h = int(hashlib.sha256(f"{symbol}:{t0 - self.lookback + i}".encode()).hexdigest(), 16)
            step = ((h % 2000) / 1000.0 - 1.0)          # -1..1
            drift = 0.02 * math.sin((t0 + i) / 7.0)
            price = max(1.0, price * (1 + 0.01 * step + drift * 0.01))
            closes.append(round(price, 4))
        return Quote(symbol, closes[-1], closes, "synthetic")
