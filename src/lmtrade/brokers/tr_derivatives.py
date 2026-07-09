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

# TR returns the full knockout catalog for an underlying in a single
# websocket frame — for a liquid name (AAPL etc.) that's well over the
# websockets library's default 1 MiB max_size, which pytr never raises, so
# the frame is rejected with a 1009 "message too big". 32 MiB is generous
# headroom for even the largest catalog while still bounding memory.
WS_MAX_SIZE = 32 * 1024 * 1024


def _patch_ws_max_size(ws_module: Any) -> None:
    """Wrap the `connect` attribute of pytr's imported `websockets` module so
    every socket pytr opens defaults to WS_MAX_SIZE instead of the 1 MiB
    library default. Necessary because pytr calls websockets.connect() with
    no max_size and reconnects on each search, so we cannot just pre-open one
    socket ourselves. Idempotent, and leaves an explicit max_size untouched."""
    connect = getattr(ws_module, "connect", None)
    if connect is None or getattr(connect, "_lmtrade_patched", False):
        return

    def patched_connect(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("max_size", WS_MAX_SIZE)
        return connect(*args, **kwargs)

    patched_connect._lmtrade_patched = True  # type: ignore[attr-defined]
    ws_module.connect = patched_connect


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
        from pytr import api as pytr_api  # type: ignore

        # Raise pytr's websocket message cap before any connection is opened,
        # so a >1 MiB knockout catalog frame isn't rejected (see WS_MAX_SIZE).
        _patch_ws_max_size(pytr_api.websockets)
        return pytr_api.TradeRepublicApi(
            phone_no=self._phone, pin=self._pin, save_cookies=True)

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

    # Best-effort productCategory value — TR's backend rejected the plain
    # "knockout" as a schema-invalid payload against a live account (seen as
    # a JSON_PARSE_ERROR/"validation failed" from TR's own MAPPER service).
    # This environment has no live TR access to verify the exact value, so
    # this is a single educated guess: TR's other subscription type names are
    # camelCase compounds (portfolioAggregateHistory, instrumentSuitability,
    # timelineDetailV2), matching this value's shape better than the flat
    # lowercase one did.
    PRODUCT_CATEGORY = "knockOutProduct"

    async def _fetch_derivatives(self, api: Any, isin: str) -> list[dict]:
        sub_id = await api.search_derivative(isin, self.PRODUCT_CATEGORY)
        _, _, payload = await api.recv()
        await api.unsubscribe(sub_id)
        log.debug("TR search_derivative(%s, %s) -> %r", isin, self.PRODUCT_CATEGORY, payload)
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
