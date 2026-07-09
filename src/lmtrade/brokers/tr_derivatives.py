"""Optional Trade Republic derivatives catalog adapter.

When TR credentials (TR_PHONE/TR_PIN) are configured AND tr.use_derivatives
is on, paper trading selects REAL Trade Republic knockout certificates (real
ISINs from the issuer catalog) instead of synthetic instruments — fills are
still simulated (paper), but on instruments that actually exist on TR, priced
with the knockout model in finance/knockouts.py against live underlying data.

Without credentials the factory returns None and the engine falls back to the
synthetic Black-Scholes options layer, exactly as before — TR login stays
strictly optional.

Honest operational caveats, written down so nobody is surprised later:
- pytr is an UNOFFICIAL client of TR's private mobile API (against ToS; see
  docs/TRADE_REPUBLIC.md). Logging in triggers the same 2FA as the app.
- TR's API is websocket-based. Corporate/sandboxed proxies frequently do not
  pass websockets (the environment this was developed in explicitly does
  not), so the live client is written defensively: any failure at any stage
  degrades to unavailable, never crashes the engine, and the synthetic
  fallback takes over.
- The derivative-search payload shape differs across pytr versions; the
  parsing here is deliberately tolerant (missing fields -> skip instrument).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..config import Settings, secret
from ..finance.knockouts import MAX_LEVERAGE, MIN_LEVERAGE
from ..logging_setup import get_logger

log = get_logger("lmtrade.tr")


@dataclass
class TRDerivativeQuote:
    isin: str
    underlying: str
    kind: str            # ko_call | ko_put
    strike: float
    barrier: float
    ratio: float
    price: float         # ask per certificate
    leverage: float
    issuer: str = ""


class TRDerivativesBase:
    """Interface: find the best-fitting real KO instrument for a signal."""

    def available(self) -> bool:
        raise NotImplementedError

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        raise NotImplementedError

    def find_knockout(self, underlying: str, direction: str, spot: float,
                      target_leverage: float) -> TRDerivativeQuote | None:
        """Best instrument = tradeable leverage closest to target, within the
        sane retail band."""
        kind = "ko_call" if direction == "buy" else "ko_put"
        candidates = [q for q in self.search(underlying, direction)
                      if q.kind == kind and q.price > 0
                      and MIN_LEVERAGE <= q.leverage <= MAX_LEVERAGE]
        if not candidates:
            return None
        return min(candidates, key=lambda q: abs(q.leverage - target_leverage))


class FakeTRDerivatives(TRDerivativesBase):
    """Offline stand-in for tests and dry runs: a hand-fed catalog."""

    def __init__(self, catalog: dict[str, list[TRDerivativeQuote]]):
        self._catalog = catalog

    def available(self) -> bool:
        return True

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        return list(self._catalog.get(underlying, []))


class PytrDerivatives(TRDerivativesBase):
    """Live client over pytr. Lazily logs in on first use; every failure mode
    (missing pytr, login/2FA failure, websocket blocked, payload drift)
    degrades to unavailable rather than raising into the engine.

    `api_factory` is an injection point for tests (see FakeAsyncTRApi in
    test_tr_derivatives.py) so this glue code can be verified without the
    optional pytr dependency installed and without ever touching TR's real
    (websocket) API. Left unset, it lazily constructs the real
    pytr.api.TradeRepublicApi.
    """

    def __init__(self, phone: str, pin: str,
                 api_factory: Callable[[], Any] | None = None):
        self._phone = phone
        self._pin = pin
        self._api_factory = api_factory
        self._api = None
        self._failed = False

    def _make_api(self) -> Any:
        if self._api_factory is not None:
            return self._api_factory()
        from pytr.api import TradeRepublicApi  # type: ignore

        return TradeRepublicApi(phone_no=self._phone, pin=self._pin, save_cookies=True)

    def _login(self):
        if self._api is not None or self._failed:
            return self._api
        try:
            api = self._make_api()
            # pytr>=0.4 dropped the old `.login()` method. The correct
            # non-interactive flow (mirroring pytr's own `account.login()`
            # reference implementation) is: try to resume the session cookie
            # saved by a prior interactive `pytr login`. If that fails there
            # is no cached session to resume, and the fresh-pairing flow
            # (initiate_weblogin -> wait for a 2FA code -> complete_weblogin)
            # needs interactive input we must never block on here — so we
            # degrade instead, same as any other unavailable-provider case.
            if not api.resume_websession():
                log.warning(
                    "TR session not resumable — no cached session cookie found "
                    "(or it expired). Run `pytr login -n \"<TR_PHONE>\" -p "
                    "\"<TR_PIN>\" --store_credentials` once interactively (the "
                    "--store_credentials flag is required, or nothing persists "
                    "to disk; the phone number must match TR_PHONE "
                    "character-for-character), then restart LMTrade. Falling "
                    "back to synthetic instruments for now.")
                self._failed = True
                return None
            self._api = api
        except Exception as exc:  # noqa: BLE001
            log.warning("TR derivatives unavailable (%s) — falling back to "
                        "synthetic instruments.", exc)
            self._failed = True
        return self._api

    def available(self) -> bool:
        return self._login() is not None

    async def _resolve_isin(self, api: Any, underlying: str) -> str | None:
        """TR's derivative search takes an ISIN, not a ticker — look the
        underlying up first via TR's own instrument search."""
        sub_id = await api.search(underlying, asset_type="stock")
        _, _, payload = await api.recv()
        await api.unsubscribe(sub_id)
        for result in (payload or {}).get("results", []):
            isin = result.get("isin")
            if isin:
                return isin
        return None

    async def _fetch_derivatives(self, api: Any, isin: str) -> list[dict]:
        # "knockout" is the best-effort productCategory value (TR's own app
        # terminology) — this environment cannot reach TR's live websocket
        # API to verify the exact request/response schema against a real
        # account. debug-logging the raw payload here so a live run can
        # confirm/correct it quickly if this comes back empty in practice.
        sub_id = await api.search_derivative(isin, "knockout")
        _, _, payload = await api.recv()
        await api.unsubscribe(sub_id)
        log.debug("TR search_derivative(%s, knockout) -> %r", isin, payload)
        return (payload or {}).get("results", [])

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        api = self._login()
        if api is None:
            return []
        try:
            import asyncio

            async def _query() -> list[dict]:
                isin = await self._resolve_isin(api, underlying)
                if isin is None:
                    return []
                return await self._fetch_derivatives(api, isin)

            items = asyncio.get_event_loop().run_until_complete(_query())
            out: list[TRDerivativeQuote] = []
            for item in items:
                try:
                    out.append(TRDerivativeQuote(
                        isin=item["isin"], underlying=underlying,
                        # Every result from this query is labeled with the
                        # requested direction rather than an actual call/put
                        # field from the response (unconfirmed field name) —
                        # a pre-existing simplification, not new here.
                        kind="ko_call" if direction == "buy" else "ko_put",
                        strike=float(item["strike"]),
                        barrier=float(item.get("barrier", item["strike"])),
                        ratio=float(item.get("ratio", 1.0) or 1.0),
                        price=float(item.get("ask", 0) or 0),
                        leverage=float(item.get("leverage", 0) or 0),
                        issuer=str(item.get("issuerDisplayName", ""))))
                except (KeyError, TypeError, ValueError):
                    continue  # tolerate payload drift per-instrument
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("TR derivative search failed (%s).", exc)
            return []


def build_tr_derivatives(settings: Settings) -> TRDerivativesBase | None:
    """Factory honoring the opt-in contract: credentials present AND
    tr.use_derivatives enabled, else None (synthetic fallback)."""
    if not settings.tr.use_derivatives:
        return None
    phone, pin = secret("TR_PHONE"), secret("TR_PIN")
    if not (phone and pin):
        return None
    return PytrDerivatives(phone, pin)
