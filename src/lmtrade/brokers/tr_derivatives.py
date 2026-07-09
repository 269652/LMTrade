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
    degrades to unavailable rather than raising into the engine."""

    def __init__(self, phone: str, pin: str):
        self._phone = phone
        self._pin = pin
        self._api = None
        self._failed = False

    def _login(self):
        if self._api is not None or self._failed:
            return self._api
        try:
            from pytr.api import TradeRepublicApi  # type: ignore

            api = TradeRepublicApi(phone_no=self._phone, pin=self._pin,
                                   save_cookies=True)
            # Reuses the stored session cookie when present; a fresh pairing
            # requires interactive 2FA once (run `pytr login` manually).
            api.login()
            self._api = api
        except Exception as exc:  # noqa: BLE001
            log.warning("TR derivatives unavailable (%s) — falling back to "
                        "synthetic instruments.", exc)
            self._failed = True
        return self._api

    def available(self) -> bool:
        return self._login() is not None

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        api = self._login()
        if api is None:
            return []
        try:
            import asyncio

            async def _query():
                # TR's derivative search: knock-out products for an underlying,
                # long or short. Field names tolerant to pytr/API versions.
                sub_id = await api.derivative_search(
                    underlying, product_category="knockOutProduct",
                    direction="long" if direction == "buy" else "short")
                _, _, payload = await api.recv()
                await api.unsubscribe(sub_id)
                return payload

            payload = asyncio.get_event_loop().run_until_complete(_query())
            out: list[TRDerivativeQuote] = []
            for item in (payload or {}).get("results", []):
                try:
                    out.append(TRDerivativeQuote(
                        isin=item["isin"], underlying=underlying,
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
