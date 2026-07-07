"""Trade Republic live adapter (opt-in, LMTRADE_MODE=live).

⚠️  Trade Republic has NO official trading API. This adapter targets the
unofficial, reverse-engineered `pytr` client, which uses TR's private mobile
API. Using it may violate Trade Republic's Terms of Service and can get your
account locked. It requires phone + PIN login and app-based 2FA. Enable only if
you accept that risk.

The methods below are intentionally guarded: without `pytr` installed and
credentials present, construction fails loudly rather than silently doing
nothing. Order placement is left as a clearly-marked integration point so live
trading is a deliberate, reviewed step — not an accident.
"""
from __future__ import annotations

from ..config import Settings, secret
from ..core.state import Store
from .base import Broker, OrderResult


class TradeRepublicBroker(Broker):
    mode = "live"

    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings
        self.phone = secret("TR_PHONE")
        self.pin = secret("TR_PIN")
        if not (self.phone and self.pin):
            raise RuntimeError(
                "Live mode needs TR_PHONE and TR_PIN in the environment. "
                "Refusing to start live trading without credentials."
            )
        try:
            from pytr.account import Account  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "pytr not installed. Install the optional extra: "
                "pip install 'lmtrade[traderepublic]'. Live trading is unofficial "
                "and against Trade Republic ToS — proceed at your own risk."
            ) from exc
        self._account = None  # lazily logged-in pytr session

    def _login(self):
        if self._account is not None:
            return self._account
        from pytr.account import Account  # type: ignore

        acc = Account(phone_no=self.phone, pin=self.pin)
        # NOTE: pytr login triggers a 2FA prompt (app / SMS). In an unattended
        # deployment you must complete the device-reset pairing once and reuse
        # the stored cookie. See docs/TRADE_REPUBLIC.md.
        acc.login()
        self._account = acc
        return acc

    def cash(self) -> float:
        acc = self._login()
        try:
            # pytr exposes cash via the portfolio websocket payload; the exact
            # field is left as an integration point tied to your pytr version.
            return float(getattr(acc, "cash", 0.0) or 0.0)
        except Exception:
            return 0.0

    def price(self, symbol: str) -> float:
        return 0.0  # engine passes live prices from the data layer

    def buy(self, symbol: str, qty: float, price: float) -> OrderResult:
        return self._place(symbol, "buy", qty, price)

    def sell(self, symbol: str, qty: float, price: float) -> OrderResult:
        return self._place(symbol, "sell", qty, price)

    def _place(self, symbol: str, side: str, qty: float, price: float) -> OrderResult:
        # Deliberate guard: real order submission is not wired by default. Fill
        # this in against your reviewed pytr version and remove the guard only
        # when you have tested against a funded account and accepted the risk.
        return OrderResult(
            False, symbol, side, qty, price, 1.0,
            "LIVE order placement not enabled — see brokers/trade_republic.py",
        )
