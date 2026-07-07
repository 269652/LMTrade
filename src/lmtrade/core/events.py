"""Thin façade the rest of the engine uses to emit logs and activity. Writes to
both the Python logger (console) and the Store (dashboard)."""
from __future__ import annotations

from ..logging_setup import get_logger
from .state import Store

log = get_logger("lmtrade")


class EventBus:
    def __init__(self, store: Store):
        self.store = store

    def info(self, message: str, source: str = "engine") -> None:
        log.info("%s", message)
        self.store.add_log("INFO", message, source)

    def warn(self, message: str, source: str = "engine") -> None:
        log.warning("%s", message)
        self.store.add_log("WARNING", message, source)

    def error(self, message: str, source: str = "engine") -> None:
        log.error("%s", message)
        self.store.add_log("ERROR", message, source)

    def activity(
        self, kind: str, summary: str, symbol: str | None = None, detail: dict | None = None
    ) -> None:
        """Record a structured activity item (decision/signal/trade/risk/economics)."""
        self.store.add_activity(kind, summary, symbol, detail)
        log.info("[%s] %s", kind, summary)
