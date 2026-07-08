"""Paper broker — simulated fills against live/synthetic prices, cash and
positions persisted in the Store. This is the default and the only mode that
touches no real money."""
from __future__ import annotations

import time

from ..core.state import Position, Store, Trade
from .base import Broker, OrderResult

# Trade Republic charges a flat 1 EUR external fee on most orders; model it so
# paper P&L is honest about the drag it puts on a 10 EUR account.
DEFAULT_FEE = 1.0


class PaperBroker(Broker):
    mode = "paper"

    def __init__(self, store: Store, starting_cash: float, fee: float = DEFAULT_FEE):
        self.store = store
        self.fee = fee
        if store.get_meta("cash") is None:
            store.set_meta("cash", starting_cash)
            store.set_meta("starting_cash", starting_cash)

    def cash(self) -> float:
        return float(self.store.get_meta("cash", 0.0))

    def _set_cash(self, value: float) -> None:
        self.store.set_meta("cash", round(value, 6))

    def price(self, symbol: str) -> float:  # not used directly; engine passes prices
        pos = self.store.position(symbol)
        return pos.avg_price if pos else 0.0

    def adjust_cash(self, delta: float) -> bool:
        """Credit (positive) or debit (negative) cash directly — used by the
        options book for premiums and proceeds. Debits that would overdraw are
        rejected."""
        new = self.cash() + delta
        if new < -1e-9:
            return False
        self._set_cash(new)
        return True

    def buy(self, symbol: str, qty: float, price: float) -> OrderResult:
        cost = qty * price + self.fee
        if cost > self.cash() + 1e-9:
            return OrderResult(False, symbol, "buy", qty, price, self.fee,
                               "insufficient cash")
        self._set_cash(self.cash() - cost)

        existing = self.store.position(symbol)
        if existing:
            total_qty = existing.qty + qty
            avg = (existing.avg_price * existing.qty + price * qty) / total_qty
            self.store.upsert_position(Position(symbol, total_qty, avg, existing.opened_ts))
        else:
            self.store.upsert_position(Position(symbol, qty, price, time.time()))

        self.store.record_cost("fee", self.fee, "trade_republic")
        return OrderResult(True, symbol, "buy", qty, price, self.fee, "filled")

    def sell(self, symbol: str, qty: float, price: float) -> OrderResult:
        existing = self.store.position(symbol)
        if not existing or existing.qty < qty - 1e-9:
            return OrderResult(False, symbol, "sell", qty, price, self.fee,
                               "no position / insufficient qty")
        proceeds = qty * price - self.fee
        self._set_cash(self.cash() + proceeds)

        remaining = existing.qty - qty
        self.store.upsert_position(
            Position(symbol, remaining, existing.avg_price, existing.opened_ts)
        )
        self.store.record_cost("fee", self.fee, "trade_republic")
        return OrderResult(True, symbol, "sell", qty, price, self.fee, "filled")
