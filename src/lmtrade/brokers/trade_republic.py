"""Trade Republic LIVE execution adapter (real money).

⚠️  Trade Republic has NO official trading API. This adapter drives the
unofficial, reverse-engineered `pytr` client over TR's private mobile API.
Using it may violate Trade Republic's Terms of Service and can get your account
locked. It requires phone + PIN login and app-based 2FA. Fills are REAL and
IRREVERSIBLE. Enable only if you accept that risk.

Defense in depth: even when this broker is constructed, it refuses to place an
order unless `armed=True` was passed (the engine passes `control.live_armed`,
which is only ever true when the dashboard's paper/live toggle is on *live*
AND the arm+confirm guard has been satisfied). So a real order requires the
control plane, the engine routing, AND this flag to all agree.
"""
from __future__ import annotations

from typing import Any, Callable

from ..config import Settings, secret
from ..core.state import Store
from ..logging_setup import get_logger
from .base import Broker, OrderResult
from .tr_derivatives import _parse_cash

log = get_logger("lmtrade.tr")

# TR's default trading venue for retail; LS Exchange (Lang & Schwarz).
DEFAULT_EXCHANGE = "LSX"


class TradeRepublicBroker(Broker):
    mode = "live"

    def __init__(self, store: Store, settings: Settings, *, armed: bool = False,
                 api_factory: Callable[[], Any] | None = None):
        self.store = store
        self.settings = settings
        self.armed = bool(armed)
        self._api_factory = api_factory
        self.phone = secret("TR_PHONE")
        self.pin = secret("TR_PIN")
        if not (self.phone and self.pin):
            raise RuntimeError(
                "Live mode needs TR_PHONE and TR_PIN in the environment. "
                "Refusing to start live trading without credentials."
            )
        self._api = None
        self._next_retry = 0.0

    def _invalidate(self) -> None:
        """Drop the session and back off, so a refreshed cookie (after the
        user re-runs `pytr login`) is picked up on a later attempt instead of
        disabling live execution permanently."""
        import time
        self._api = None
        self._next_retry = time.time() + 60.0

    # -- session -------------------------------------------------------------
    def _make_api(self) -> Any:
        if self._api_factory is not None:
            return self._api_factory()
        from pytr import api as pytr_api  # type: ignore

        from .tr_derivatives import _patch_ws_max_size
        _patch_ws_max_size(pytr_api.websockets)
        return pytr_api.TradeRepublicApi(
            phone_no=self.phone, pin=self.pin, save_cookies=True)

    def _login(self):
        import time
        if self._api is not None:
            return self._api
        if time.time() < self._next_retry:
            return None   # backoff after a recent failure
        try:
            api = self._make_api()
            if not api.resume_websession():
                log.warning("TR live session not resumable — re-run `pytr login "
                            "--store_credentials`; the bot will pick it up on "
                            "its next attempt.")
                self._invalidate()
                return None
            self._api = api
        except Exception as exc:  # noqa: BLE001
            log.warning("TR live session unavailable (%s) — will retry.", exc)
            self._invalidate()
        return self._api

    # -- balances ------------------------------------------------------------
    def cash(self) -> float:
        """Real TR account cash (account currency). 0.0 if unreachable."""
        api = self._login()
        if api is None:
            return 0.0
        try:
            import asyncio

            async def _q() -> Any:
                sub_id = await api.cash()
                _, _, payload = await api.recv()
                await api.unsubscribe(sub_id)
                return payload

            return _parse_cash(asyncio.get_event_loop().run_until_complete(_q())) or 0.0
        except Exception as exc:  # noqa: BLE001
            log.warning("TR cash fetch failed (%s) — dropping session.", exc)
            self._invalidate()
            return 0.0

    def price(self, symbol: str) -> float:
        return 0.0  # engine passes live prices from the data layer

    # -- execution -----------------------------------------------------------
    def place_order(self, isin: str, side: str, size: float,
                    exchange: str = DEFAULT_EXCHANGE) -> OrderResult:
        """Place a REAL market order for `isin` (a knockout/warrant/equity ISIN).
        Refuses unless armed. `size` is the number of certificates/shares."""
        if not self.armed:
            return OrderResult(
                False, isin, side, size, 0.0, 1.0,
                "LIVE order blocked: broker not armed. Arm + confirm live "
                "execution in the dashboard first.")
        api = self._login()
        if api is None:
            return OrderResult(False, isin, side, size, 0.0, 1.0,
                               "LIVE order blocked: TR session unavailable.")
        try:
            import asyncio

            async def _q() -> Any:
                # good-for-day market order, no fractional certificates.
                sub_id = await api.market_order(isin, exchange, side, size,
                                                "gfd", False)
                _, _, payload = await api.recv()
                await api.unsubscribe(sub_id)
                return payload

            payload = asyncio.get_event_loop().run_until_complete(_q())
            errors = (payload or {}).get("errors")
            if errors:
                log.warning("TR rejected %s %s x%s: %s", side, isin, size, errors)
                return OrderResult(False, isin, side, size, 0.0, 1.0,
                                   f"TR rejected order: {errors}")
            log.info("LIVE order placed: %s %s x%s on %s", side, isin, size, exchange)
            return OrderResult(True, isin, side, size, 0.0, 1.0, "live order placed")
        except Exception as exc:  # noqa: BLE001
            log.warning("TR order error (%s %s x%s): %s", side, isin, size, exc)
            return OrderResult(False, isin, side, size, 0.0, 1.0,
                               f"TR order error: {exc}")

    def buy(self, symbol: str, qty: float, price: float) -> OrderResult:
        # Equity-by-ticker isn't the bot's live path (it trades knockouts by
        # ISIN via place_order); resolve-then-place could be added if needed.
        return self.place_order(symbol, "buy", qty)

    def sell(self, symbol: str, qty: float, price: float) -> OrderResult:
        return self.place_order(symbol, "sell", qty)
