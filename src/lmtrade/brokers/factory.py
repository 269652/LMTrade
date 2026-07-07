"""Select and construct the broker for the configured mode."""
from __future__ import annotations

from ..config import Settings
from ..core.state import Store
from .base import Broker
from .paper import PaperBroker


def build_broker(settings: Settings, store: Store) -> Broker:
    if settings.mode == "live":
        # Imported lazily so a paper-only install needn't have pytr.
        from .trade_republic import TradeRepublicBroker

        return TradeRepublicBroker(store, settings)
    return PaperBroker(store, starting_cash=settings.budget)
