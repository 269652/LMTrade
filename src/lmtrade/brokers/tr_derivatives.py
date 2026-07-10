"""Trade Republic derivatives catalog adapter.

When TR credentials (TR_PHONE/TR_PIN) are configured AND tr.use_derivatives
is on, both paper and live trading select REAL Trade Republic instruments
(real ISINs from the issuer catalog: knockout certificates/Turbos, or vanilla
put/call warrants when no Turbo exists for a symbol — the common case) — in
paper mode the fill is simulated, in armed-live mode a real order is placed.
There is no synthetic-instrument fallback: without a working TR client,
options trading simply does not open new positions (see engine.py).

Two-stage lookup, because TR's derivative search can return thousands of
instruments per symbol with no price attached at all: search() resolves
metadata-only candidates (real ISIN/strike/barrier/leverage/expiry) for every
product category, find_knockout() picks the single best leverage-fit
candidate from that pool, and ONLY THEN fetches a live price for that one
instrument via a separate ticker(isin) subscription.

Honest operational caveats, written down so nobody is surprised later:
- pytr is an UNOFFICIAL client of TR's private mobile API (against ToS; see
  docs/TRADE_REPUBLIC.md). Logging in triggers the same 2FA as the app.
- TR's API is websocket-based. Corporate/sandboxed proxies frequently do not
  pass websockets (the environment this was developed in explicitly does
  not), so the live client is written defensively: any failure at any stage
  degrades to unavailable rather than crashing the engine.
- The derivative-SEARCH field mapping (isin/optionType/strike/barrier/size/
  leverage/expiry) is CONFIRMED against a live account. The ticker(isin)
  PRICE payload shape is still a best-effort guess (see
  PytrDerivatives._parse_ticker_price) — run scripts/diagnose_tr.py against
  a real account to verify/correct it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
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


# ---------------------------------------------------------------------------
# Shared pytr session. TR allows ONE active websocket per session cookie: when
# the live broker and the derivatives client each opened their own
# TradeRepublicApi, the second connection was rejected with HTTP 401 (observed
# live: "Connected." followed immediately by a 401 on the next connection).
# Everything in this process must share a single session per phone number.
_SHARED_APIS: dict[str, Any] = {}


def _new_pytr_api(phone: str, pin: str) -> Any:
    from pytr import api as pytr_api  # type: ignore

    _patch_ws_max_size(pytr_api.websockets)
    return pytr_api.TradeRepublicApi(phone_no=phone, pin=pin, save_cookies=True)


def get_shared_api(phone: str, pin: str) -> Any:
    api = _SHARED_APIS.get(phone)
    if api is None:
        api = _new_pytr_api(phone, pin)
        _SHARED_APIS[phone] = api
    return api


def drop_shared_api(phone: str) -> None:
    """Forget the shared session (stale cookie / websocket error) so the next
    get re-creates and re-resumes it."""
    _SHARED_APIS.pop(phone, None)


# Live incident: ticker(isin) on some instruments never produced a single
# frame — api.recv() just sat there. Since every caller reaches _recv_for via
# asyncio.get_event_loop().run_until_complete() (a synchronous, BLOCKING
# call), an unbounded recv() doesn't just fail that one lookup — it freezes
# the entire engine cycle (or a diagnostic script) forever. Every frame wait
# is now bounded.
RECV_TIMEOUT_S = 15.0


async def _recv_for(api: Any, sub_id: Any, max_frames: int = 10,
                    timeout_s: float = RECV_TIMEOUT_S) -> Any:
    """Receive until the frame for OUR subscription arrives. With the shared
    websocket, recv() can hand back another consumer's frame first — matching
    on subscription id prevents e.g. an order path consuming a ticker frame
    (or vice versa) and misreading it as its own response. Each individual
    frame wait is bounded by timeout_s (TR does not always answer every
    subscription type)."""
    import asyncio

    for _ in range(max_frames):
        try:
            rid, _, payload = await asyncio.wait_for(api.recv(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"no frame received within {timeout_s:g}s waiting for "
                f"subscription {sub_id}") from None
        if str(rid) == str(sub_id):
            return payload
    raise RuntimeError(f"no response for subscription {sub_id} "
                       f"within {max_frames} frames")


def _resume_once(api: Any) -> bool:
    """resume_websession() exactly once per shared session — a second resume
    on an already-live session can rotate tokens under the open websocket."""
    if getattr(api, "_lmtrade_resumed", False):
        return True
    ok = bool(api.resume_websession())
    if ok:
        api._lmtrade_resumed = True  # noqa: SLF001
    return ok


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
    kind: str            # ko_call | ko_put (knockouts) or call | put (vanilla warrants)
    strike: float
    barrier: float | None   # None for vanilla warrants — no knockout barrier
    ratio: float
    # Ask per certificate. TR's derivative SEARCH results (search_derivative)
    # carry no price field at all — confirmed against a live account, not
    # guessed. A freshly-searched quote is 0.0 here; PytrDerivatives.
    # find_knockout() fetches the real price via a separate ticker(isin)
    # subscription for the ONE candidate it selects (not per-candidate —
    # search results can run into the thousands) before returning it.
    price: float
    leverage: float
    issuer: str = ""
    expiry_ts: float | None = None   # vanilla warrants only; knockouts are open-ended


def _parse_cash(payload: Any) -> float | None:
    """TR's `cash` subscription returns per-currency balances,
    e.g. [{"currencyId": "EUR", "amount": 123.45}]. Prefer EUR; fall back to
    the first parseable amount; None if the shape isn't recognized."""
    entries = payload if isinstance(payload, list) else None
    if entries is None:
        return None
    best: float | None = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            amount = float(e.get("amount"))
        except (TypeError, ValueError):
            continue
        if str(e.get("currencyId", "")).upper() == "EUR":
            return amount
        if best is None:
            best = amount
    return best


def _parse_ts(raw: Any) -> float | None:
    """Tolerant timestamp parser for a vanilla warrant's expiry — TR's exact
    field/format for this is unverified (no live account to check against),
    so this accepts the plausible shapes: epoch milliseconds, epoch seconds,
    or an ISO 8601 date/datetime string. None (not a guessed default) on
    anything unrecognized — the caller decides the fallback."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # TR's other timestamped fields are epoch-ms; a value that large is
        # almost certainly milliseconds, not seconds (seconds would be ~5
        # digits for any near-term expiry vs. this century's ~13-digit ms).
        return raw / 1000.0 if raw > 10_000_000_000 else float(raw)
    if isinstance(raw, str):
        import datetime as _dt
        try:
            return _dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


class TRDerivativesBase:
    """Interface: find the best-fitting real KO instrument for a signal."""

    def available(self) -> bool:
        raise NotImplementedError

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        raise NotImplementedError

    def account_cash(self) -> float | None:
        """Live TR account cash balance (account currency). None unless a real
        authenticated TR client overrides this — paper/fake clients have no
        real account."""
        return None

    def portfolio(self) -> list[dict] | None:
        """Real TR portfolio positions [{isin, size, avg_price}] — None when
        unavailable (no real account / session down), NEVER an empty list on
        failure: [] means 'the account truly holds nothing' and callers may
        reconcile (delete local rows) against it."""
        return None

    # A "buy" wants a call-direction instrument, "sell" a put-direction one —
    # across BOTH product types this client can return (knockouts use
    # ko_call/ko_put, vanilla warrants use call/put).
    _CALL_KINDS = frozenset({"ko_call", "call"})
    _PUT_KINDS = frozenset({"ko_put", "put"})

    def find_knockout(self, underlying: str, direction: str, spot: float,
                      target_leverage: float) -> TRDerivativeQuote | None:
        """Best instrument = tradeable leverage closest to target, within the
        sane retail band. Candidates span every product type search() found
        (real Turbos/knockouts AND vanilla put/call warrants) — whichever
        actually exists for this underlying and best fits the target.

        No price filter here: search() results carry no price at all (see
        TRDerivativeQuote.price) for a real PytrDerivatives client — pricing
        happens for the ONE selected candidate, in PytrDerivatives'
        find_knockout() override, via a separate ticker fetch. A fully fake
        client (FakeTRDerivatives, tests) already bakes a real price into
        every quote it hands out, so this is a no-op filter for it."""
        wanted = self._CALL_KINDS if direction == "buy" else self._PUT_KINDS
        candidates = [q for q in self.search(underlying, direction)
                      if q.kind in wanted
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
                 api_factory: Callable[[], Any] | None = None,
                 now: Callable[[], float] = time.time,
                 retry_backoff_s: float = 60.0):
        self._phone = phone
        self._pin = pin
        self._api_factory = api_factory
        self._api = None
        self._now = now
        self._retry_backoff_s = retry_backoff_s
        self._next_retry = 0.0   # earliest time to re-attempt a failed login

    def _invalidate(self) -> None:
        """Drop the session and schedule a re-login after a backoff. Used both
        when a login fails and when a live call hits a stale-session error
        (e.g. HTTP 401 on the websocket) — the session is NOT abandoned
        permanently, so once `pytr login` refreshes the cookie the next
        attempt re-resumes and the bot recovers without a restart."""
        self._api = None
        self._next_retry = self._now() + self._retry_backoff_s
        drop_shared_api(self._phone)   # stale for every consumer of the session

    def _make_api(self) -> Any:
        if self._api_factory is not None:
            return self._api_factory()
        return get_shared_api(self._phone, self._pin)

    def _login(self):
        if self._api is not None:
            return self._api
        if self._now() < self._next_retry:
            return None   # in backoff after a recent failure — don't hammer TR
        try:
            api = self._make_api()
            # pytr>=0.4 dropped the old `.login()` method. The correct
            # non-interactive flow (mirroring pytr's own `account.login()`
            # reference implementation) is: try to resume the session cookie
            # saved by a prior interactive `pytr login`. If that fails there
            # is no cached session to resume, and the fresh-pairing flow needs
            # interactive 2FA we must never block on here — so we back off and
            # retry, picking up a refreshed cookie automatically once the user
            # re-runs `pytr login`. Resumed at most once per shared session —
            # re-resuming under an open websocket rotates tokens and 401s it.
            if not _resume_once(api):
                log.warning(
                    "TR session not resumable (expired or no cookie). Re-run "
                    "`pytr login -n \"<TR_PHONE>\" -p \"<TR_PIN>\" "
                    "--store_credentials`; the bot will pick up the refreshed "
                    "session on its next attempt. Using synthetic instruments "
                    "meanwhile.")
                self._invalidate()
                return None
            self._api = api
        except Exception as exc:  # noqa: BLE001
            log.warning("TR derivatives unavailable (%s) — retrying later, "
                        "synthetic instruments meanwhile.", exc)
            self._invalidate()
        return self._api

    def available(self) -> bool:
        return self._login() is not None

    async def _resolve_isin(self, api: Any, underlying: str) -> str | None:
        """TR's derivative search takes an ISIN, not a ticker — look the
        underlying up first via TR's own instrument search."""
        sub_id = await api.search(underlying, asset_type="stock")
        payload = await _recv_for(api, sub_id)
        await api.unsubscribe(sub_id)
        for result in (payload or {}).get("results", []):
            isin = result.get("isin")
            if isin:
                return isin
        return None

    # Best-effort productCategory values — TR's backend rejected the plain
    # "knockout" as a schema-invalid payload against a live account (seen as
    # a JSON_PARSE_ERROR/"validation failed" from TR's own MAPPER service).
    # This environment has no live TR access to verify the exact values, so
    # these are educated guesses: TR's other subscription type names are
    # camelCase compounds (portfolioAggregateHistory, instrumentSuitability,
    # timelineDetailV2), matching this shape better than flat lowercase ones.
    #
    # Live incident: MOST symbols return zero knockOutProduct instruments —
    # TR simply doesn't list a Turbo for them, not a bug in this client. TR
    # also sells vanilla put/call warrants (no knockout barrier) under a
    # separate category; search() tries both and merges whatever exists.
    PRODUCT_CATEGORY = "knockOutProduct"
    VANILLA_PRODUCT_CATEGORY = "vanillaWarrant"

    async def _fetch_derivatives(self, api: Any, isin: str, category: str) -> list[dict]:
        sub_id = await api.search_derivative(isin, category)
        payload = await _recv_for(api, sub_id)
        await api.unsubscribe(sub_id)
        log.debug("TR search_derivative(%s, %s) -> %r", isin, category, payload)
        return (payload or {}).get("results", [])

    # optionType -> our kind label. CONFIRMED against a live account (not a
    # guess): knockOutProduct items carry "long"/"short"; vanillaWarrant
    # items carry "call"/"put" directly.
    _KNOCKOUT_OPTION_TYPES = {"long": "ko_call", "short": "ko_put"}
    _VANILLA_OPTION_TYPES = {"call", "put"}

    @staticmethod
    def _parse_knockout_item(item: dict, underlying: str) -> TRDerivativeQuote:
        # strike/barrier/leverage/optionType are REQUIRED (item["..."], no
        # .get(..., 0) default). Live incident: a wrong field-name guess for
        # ask/leverage silently defaulted both to 0, so every instrument
        # "parsed" (no KeyError) yet failed find_knockout()'s tradeability
        # filter — thousands logged as 'usable', zero ever tradeable, no
        # diagnostic ever fired because nothing LOOKED broken. Requiring
        # these fields turns a wrong guess back into a loud, correctly-
        # counted parse failure instead of a quiet no-op.
        kind = PytrDerivatives._KNOCKOUT_OPTION_TYPES.get(item["optionType"])
        if kind is None:
            raise ValueError(f"unrecognized optionType {item.get('optionType')!r}")
        return TRDerivativeQuote(
            isin=item["isin"], underlying=underlying,
            kind=kind,
            strike=float(item["strike"]),
            barrier=float(item["barrier"]),
            # TR's search results carry no field literally named "ratio" —
            # "size" is the closest candidate (confirmed present on every
            # live instrument seen; UNVERIFIED as the correct interpretation
            # of a subscription ratio — if ongoing mark-to-market P&L looks
            # wrong, check this first). Entry cost itself doesn't depend on
            # it: contracts = budget / price, and price comes from a real
            # ticker fetch, not this.
            ratio=float(item["size"]),
            price=0.0,   # not in search results; find_knockout() prices the winner
            leverage=float(item["leverage"]),
            issuer=str(item.get("issuerDisplayName", "")))

    @staticmethod
    def _parse_vanilla_item(item: dict, underlying: str) -> TRDerivativeQuote:
        # See _parse_knockout_item for why strike/leverage/optionType are
        # required rather than silently defaulted.
        kind = item["optionType"]
        if kind not in PytrDerivatives._VANILLA_OPTION_TYPES:
            raise ValueError(f"unrecognized optionType {item.get('optionType')!r}")
        expiry_raw = (item.get("expiry") or item.get("expiryDate")
                     or item.get("maturityDate") or item.get("maturity"))
        return TRDerivativeQuote(
            isin=item["isin"], underlying=underlying,
            kind=kind,
            strike=float(item["strike"]),
            barrier=None,     # vanilla warrant: no knockout barrier
            ratio=float(item["size"]),   # see _parse_knockout_item's ratio note
            price=0.0,
            leverage=float(item["leverage"]),
            issuer=str(item.get("issuerDisplayName", "")),
            expiry_ts=_parse_ts(expiry_raw))

    # (category, parser, human label for diagnostics)
    _CATEGORIES = (
        (PRODUCT_CATEGORY, "_parse_knockout_item", "knockout"),
        (VANILLA_PRODUCT_CATEGORY, "_parse_vanilla_item", "vanilla warrant"),
    )

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        """Metadata-only candidates (real ISIN/strike/barrier/leverage/
        expiry) — NO price. TR's search results carry no price field at all;
        find_knockout() fetches one for the single candidate it selects."""
        api = self._login()
        if api is None:
            return []
        try:
            import asyncio

            async def _query() -> list[tuple[str, str, list[dict]]]:
                isin = await self._resolve_isin(api, underlying)
                if isin is None:
                    return []
                out = []
                for category, parser_name, label in self._CATEGORIES:
                    items = await self._fetch_derivatives(api, isin, category)
                    out.append((parser_name, label, items))
                return out

            by_category = asyncio.get_event_loop().run_until_complete(_query())
            out: list[TRDerivativeQuote] = []
            for parser_name, label, items in by_category:
                parser = getattr(self, parser_name)
                parsed: list[TRDerivativeQuote] = []
                first_error: Exception | None = None
                for item in items:
                    try:
                        parsed.append(parser(item, underlying))
                    except (KeyError, TypeError, ValueError) as exc:
                        if first_error is None:
                            first_error = exc
                        continue  # tolerate payload drift per-instrument
                if items and not parsed:
                    # TR returned instruments but our field mapping matched
                    # none — this symbol will simply have no tradeable
                    # instrument in this category this cycle (no synthetic
                    # fallback). Dump the real field names so the mapping
                    # above can be corrected against a live account; without
                    # TR access from the dev environment this is
                    # unverifiable in code.
                    log.warning(
                        "TR %s search for %s: %d instrument(s) returned but "
                        "NONE parsed (first error: %r). Actual fields on the "
                        "first item: %s. The response field mapping in "
                        "tr_derivatives.py needs updating for your pytr/TR "
                        "version.", label, underlying, len(items), first_error,
                        sorted(items[0].keys()))
                elif parsed:
                    log.info("TR %s search for %s: %d raw -> %d usable instrument(s).",
                             label, underlying, len(items), len(parsed))
                else:
                    log.debug("TR %s search for %s: no instruments returned.",
                             label, underlying)
                out.extend(parsed)
            return out
        except Exception as exc:  # noqa: BLE001
            # A stale-session error (e.g. HTTP 401 on the websocket) surfaces
            # here — drop the session so the next call re-resumes with a
            # (possibly refreshed) cookie instead of failing forever.
            log.warning("TR derivative search failed (%s) — dropping session, "
                        "will re-login.", exc)
            self._invalidate()
            return []

    async def _fetch_ticker(self, api: Any, isin: str, exchange: str = "LSX") -> Any:
        sub_id = await api.ticker(isin, exchange)
        payload = await _recv_for(api, sub_id)
        await api.unsubscribe(sub_id)
        log.debug("TR ticker(%s, %s) -> %r", isin, exchange, payload)
        return payload

    async def _fetch_price_for_order(self, api: Any, isin: str, order_type: str,
                                     exchange: str = "LSX") -> Any:
        sub_id = await api.price_for_order(isin, exchange, order_type)
        payload = await _recv_for(api, sub_id)
        await api.unsubscribe(sub_id)
        log.debug("TR priceForOrder(%s, %s, %s) -> %r", isin, exchange, order_type, payload)
        return payload

    @staticmethod
    def _parse_ticker_price(payload: Any) -> float | None:
        """The tradeable price out of a ticker(isin) or priceForOrder payload.
        UNVERIFIED against a live account (unlike the search-result field
        mapping, which is now confirmed) — tries the most plausible shapes
        for an executable buy price in order (nested {"ask": {"price": ...}}
        / {"price": {"price": ...}}, flat {"ask": ...} / {"price": ...},
        then last/bid as a fallback), and returns None rather than guessing
        further on a shape none of these match. The caller logs the raw
        payload on None so the real shape can be corrected here quickly."""
        if not isinstance(payload, dict):
            return None
        for outer, inner in (("price", "price"), ("ask", "price"),
                             ("last", "price"), ("bid", "price")):
            node = payload.get(outer)
            if isinstance(node, dict):
                try:
                    return float(node[inner])
                except (KeyError, TypeError, ValueError):
                    continue
        for key in ("price", "ask", "last", "bid"):
            val = payload.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def find_knockout(self, underlying: str, direction: str, spot: float,
                      target_leverage: float) -> TRDerivativeQuote | None:
        """Metadata-based selection (base class), THEN a live price fetch for
        the ONE winning candidate — search results run into the thousands
        per symbol, so pricing is deliberately deferred until a single
        instrument has already been chosen, not attempted per-candidate.

        Live incident: ticker(isin) — a passive streaming quote — produced
        ZERO frames (not even an error) for a real instrument on a real
        account; TR's own app doesn't rely on that topic to price an order.
        priceForOrder is TR's purpose-built pre-order pricing subscription
        and is tried FIRST; ticker() is kept only as a fallback for when
        priceForOrder itself doesn't answer."""
        candidate = super().find_knockout(underlying, direction, spot, target_leverage)
        if candidate is None:
            return None
        api = self._login()
        if api is None:
            return None
        import asyncio

        order_type = "sell" if direction == "sell" else "buy"
        price: float | None = None
        payload: Any = None
        try:
            payload = asyncio.get_event_loop().run_until_complete(
                self._fetch_price_for_order(api, candidate.isin, order_type))
            price = self._parse_ticker_price(payload)
        except Exception as exc:  # noqa: BLE001
            log.debug("TR priceForOrder fetch failed for %s (%s) — falling "
                     "back to ticker().", candidate.isin, exc)
        if price is None or price <= 0:
            try:
                payload = asyncio.get_event_loop().run_until_complete(
                    self._fetch_ticker(api, candidate.isin))
            except Exception as exc:  # noqa: BLE001
                log.warning("TR ticker fetch failed for %s (%s) — dropping "
                            "session, will re-login.", candidate.isin, exc)
                self._invalidate()
                return None
            price = self._parse_ticker_price(payload)
        if price is None:
            log.warning(
                "TR priceForOrder/ticker(%s) returned an unrecognized "
                "payload shape — no price extracted. Raw payload: %r. The "
                "mapping in PytrDerivatives._parse_ticker_price needs "
                "updating for your account.", candidate.isin, payload)
            return None
        if price <= 0:
            return None
        return replace(candidate, price=price)

    def account_cash(self) -> float | None:
        api = self._login()
        if api is None:
            return None
        try:
            import asyncio

            async def _query() -> Any:
                sub_id = await api.cash()
                payload = await _recv_for(api, sub_id)
                await api.unsubscribe(sub_id)
                return payload

            return _parse_cash(asyncio.get_event_loop().run_until_complete(_query()))
        except Exception as exc:  # noqa: BLE001
            # Drop the (stale) session so the next attempt re-resumes from the
            # cookie file — otherwise a websocket 401 recurs every cycle
            # against the same dead session and never picks up a refreshed
            # `pytr login`.
            log.warning("TR account cash fetch failed (%s) — dropping session, "
                        "will re-login.", exc)
            self._invalidate()
            return None

    # TR has served the holdings list under different subscription topics
    # across backend/account versions. Newer accounts REMOVED the classic
    # "compactPortfolio" and "portfolio" topics (both rejected live with
    # BAD_SUBSCRIPTION_TYPE "Unknown topic type: compactPortfolio") and now
    # deliver holdings via "compactPortfolioByType". We subscribe by raw topic
    # (not pytr's named methods) so topics pytr has no wrapper for are still
    # reachable, and try each in order — a rejected TOPIC is not a dead
    # SESSION.
    # Ordered most-current first: newer accounts serve holdings ONLY via
    # compactPortfolioByType and reject the classic topics with
    # BAD_SUBSCRIPTION_TYPE, so trying the modern one first means those
    # accounts never emit the (noisy but harmless) "Unknown topic type:
    # compactPortfolio" error. Older accounts fall through to the classic ones.
    _PORTFOLIO_TOPICS = (
        "compactPortfolioByType",   # current TR app; newer accounts accept only this
        "compactPortfolio",         # classic; still present on older accounts
        "portfolio",                # legacy
        "portfolioStatus",          # last-resort fallback
    )

    @staticmethod
    def _is_unknown_topic(exc: Exception) -> bool:
        """True when TR rejected the SUBSCRIPTION TYPE (wrong/removed topic
        name), which is recoverable by trying another topic — as opposed to a
        session failure (401/websocket drop) that must invalidate the session.
        pytr's TradeRepublicError carries the error payload on `.error` and
        also in its args, so check both."""
        blob = f"{getattr(exc, 'error', '')} {exc}"
        return "BAD_SUBSCRIPTION_TYPE" in blob or "Unknown topic type" in blob

    @staticmethod
    def _iter_positions(payload: Any) -> "list[dict]":
        """Flatten a holdings payload to a list of position dicts. Flat topics
        put them under "positions"; compactPortfolioByType groups them under
        "categories":[{"positions":[...]}]. Handle both."""
        items: list[dict] = []
        if not isinstance(payload, dict):
            return items
        if isinstance(payload.get("positions"), list):
            items.extend(p for p in payload["positions"] if isinstance(p, dict))
        for cat in payload.get("categories", []) or []:
            if isinstance(cat, dict) and isinstance(cat.get("positions"), list):
                items.extend(p for p in cat["positions"] if isinstance(p, dict))
        return items

    @classmethod
    def _parse_portfolio(cls, payload: Any) -> list[dict]:
        out: list[dict] = []
        for item in cls._iter_positions(payload):
            isin = item.get("instrumentId") or item.get("isin")
            try:
                size = float(item.get("netSize") or item.get("size")
                             or item.get("amount") or 0)
                avg = float(item.get("averageBuyIn") or item.get("avg_price") or 0)
            except (TypeError, ValueError):
                continue
            if isin and size > 0:
                out.append({"isin": str(isin), "size": size, "avg_price": avg})
        return out

    def portfolio(self) -> list[dict] | None:
        """Real TR portfolio. Tries each known holdings subscription topic
        until one is accepted; field names are parsed tolerantly across
        account versions and unparseable entries are skipped. None (not []) on
        any failure, so callers never reconcile against phantom-empty data."""
        api = self._login()
        if api is None:
            return None
        import asyncio

        loop = asyncio.get_event_loop()
        last_topic_error: Exception | None = None
        for topic in self._PORTFOLIO_TOPICS:
            try:
                async def _query(_t: str = topic) -> Any:
                    sub_id = await api.subscribe({"type": _t})
                    payload = await _recv_for(api, sub_id)
                    await api.unsubscribe(sub_id)
                    return payload

                payload = loop.run_until_complete(_query())
            except Exception as exc:  # noqa: BLE001
                if self._is_unknown_topic(exc):
                    # Wrong/removed topic for this backend — keep the session,
                    # try the next candidate.
                    last_topic_error = exc
                    log.info("TR portfolio topic %r not supported — trying next.",
                             topic)
                    continue
                log.warning("TR portfolio fetch failed (%s) — dropping session.", exc)
                self._invalidate()
                return None
            return self._parse_portfolio(payload)
        log.warning("TR portfolio: no supported holdings topic (tried %s; last "
                    "error: %r). Position reconciliation skipped — the bot "
                    "continues, but TR positions won't auto-import.",
                    ", ".join(self._PORTFOLIO_TOPICS), last_topic_error)
        return None


def build_tr_derivatives(settings: Settings) -> TRDerivativesBase | None:
    """Factory honoring the opt-in contract: credentials present AND
    tr.use_derivatives enabled, else None (synthetic fallback)."""
    if not settings.tr.use_derivatives:
        return None
    phone, pin = secret("TR_PHONE"), secret("TR_PIN")
    if not (phone and pin):
        return None
    return PytrDerivatives(phone, pin)
