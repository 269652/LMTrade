#!/usr/bin/env python3
"""Local Trade Republic diagnostic tool.

Exercises each step of the pytr integration IN ISOLATION — session resume,
cash, every known holdings-topic variant, ISIN search, knockout derivative
search — and prints a clear pass/fail per step with the RAW payload on
failure, so a broken link in the chain (session, subscription topic, field
mapping) can be pinpointed without wading through the bot's live logs.

READ-ONLY: this script never places an order and never touches the
`option_positions`/`positions` tables. Real order placement is exercised
through the dashboard's arm + double-confirm flow, which already carries the
safety guards a diagnostic script should not casually bypass.

Usage (from the repo root; loads TR_PHONE/TR_PIN from the environment or a
local .env, and requires a session already paired via
`pytr login -n "<phone>" -p "<pin>" --store_credentials`):

    python scripts/diagnose_tr.py [SYMBOL]

SYMBOL defaults to AAPL. Requires real network access to Trade Republic —
this will NOT work behind a websocket-blocking proxy.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmtrade.config import secret  # noqa: E402


def _load_credentials() -> tuple[str | None, str | None]:
    """TR_PHONE/TR_PIN from the shell environment OR a .env file at the repo
    root. secret() only reads os.environ — loading .env is a side effect of
    load_settings(), which this standalone script never calls — so without
    this, credentials that exist ONLY in .env (never exported to the shell)
    read as 'not set' even though the file genuinely has them. Imported
    inside the function (not at module top) so a monkeypatched REPO_ROOT in
    tests is honored."""
    from lmtrade.config import REPO_ROOT, _load_dotenv

    _load_dotenv(REPO_ROOT / ".env")
    return secret("TR_PHONE"), secret("TR_PIN")


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _info(msg: str) -> None:
    print(f"  \033[36mi\033[0m {msg}")


def _section(title: str) -> None:
    print(f"\n\033[1m=== {title} ===\033[0m")


def _dump(label: str, payload) -> None:
    try:
        text = json.dumps(payload, indent=2, default=str)
    except Exception:  # noqa: BLE001
        text = repr(payload)
    if len(text) > 2000:
        text = text[:2000] + "\n  … (truncated)"
    print(f"  {label}:")
    for line in text.splitlines():
        print(f"    {line}")


async def _query(api, sub_id_coro, recv_for) -> tuple[bool, object]:
    """Subscribe, receive one matching frame, unsubscribe. Returns
    (ok, payload_or_error)."""
    try:
        sub_id = await sub_id_coro
        payload = await recv_for(api, sub_id)
        await api.unsubscribe(sub_id)
        return True, payload
    except Exception as exc:  # noqa: BLE001
        return False, exc


# Substrings that hint a raw field is the one the bot's parser is looking
# for, regardless of whether it's under the name currently guessed in
# tr_derivatives.py. Live incident this caught: TR's real field names for
# ask price and leverage were NOT "ask"/"leverage" — the parser silently
# defaulted both to 0.0 (item.get(field, 0)), so every instrument "parsed"
# successfully yet failed the price>0/leverage-band tradeability check, and
# nothing ever looked broken until this diagnostic dumped the raw payload.
_FIELD_HINTS = ("lever", "ask", "bid", "price", "strike", "barrier", "ratio",
                "expir", "matur", "isin")


def _find_candidate_fields(item: dict) -> dict:
    return {k: v for k, v in item.items()
            if any(hint in str(k).lower() for hint in _FIELD_HINTS)}


def _product_categories(isin_search_result: dict) -> list[str]:
    """TR's OWN authoritative list of derivative product types for an
    underlying, straight from its instrument search result
    (derivativeProductCategories) — the real answer to 'does TR offer plain
    options for this symbol', not a guess at category names to try."""
    if not isinstance(isin_search_result, dict):
        return []
    return list(isin_search_result.get("derivativeProductCategories") or [])


def _diagnose_derivative_items(items: list, symbol: str, parser_name: str) -> object | None:
    """For a sample of raw instruments: dump the full raw JSON, run the
    bot's ACTUAL parser against it (showing what it extracts or why it
    fails), and call out fields whose NAME hints at strike/leverage/barrier/
    ratio/expiry/optionType — even under a different name than currently
    guessed — so the real field mapping is visible without a round trip.
    Finishes with a raw -> parsed -> in-leverage-band funnel so a silently-
    wrong field mapping is immediately obvious. Search results carry NO
    price (confirmed against a live account) — pricing is a separate step,
    see the ticker diagnostic this return value feeds. Returns the first
    successfully-parsed, in-band TRDerivativeQuote (for the ticker probe),
    or None."""
    from lmtrade.brokers.tr_derivatives import PytrDerivatives
    from lmtrade.finance.knockouts import MAX_LEVERAGE, MIN_LEVERAGE

    parser = getattr(PytrDerivatives, parser_name)
    sample = items[:3]
    for i, item in enumerate(sample):
        print(f"\n  --- instrument {i + 1}/{len(sample)} (of {len(items)} total) ---")
        _dump("raw", item)
        if not isinstance(item, dict):
            _fail(f"item is not an object ({type(item).__name__}) — cannot parse.")
            continue
        candidates = _find_candidate_fields(item)
        if candidates:
            _dump("fields whose NAME hints strike/leverage/barrier/ratio/"
                  "expiry/optionType (compare against tr_derivatives.py's "
                  "parsing even if the name differs from what's currently "
                  "guessed)", candidates)
        try:
            q = parser(item, symbol)
            _ok(f"parser extracts: kind={q.kind} leverage={q.leverage} "
                f"strike={q.strike} barrier={q.barrier} ratio={q.ratio}")
        except (KeyError, TypeError, ValueError) as exc:
            _fail(f"parser failed on this item: {exc!r} — a required field "
                  "is missing or under a different name (see candidates above).")

    parsed_ok = in_band = 0
    first_tradeable = None
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            q = parser(item, symbol)
        except (KeyError, TypeError, ValueError):
            continue
        parsed_ok += 1
        if MIN_LEVERAGE <= q.leverage <= MAX_LEVERAGE:
            in_band += 1
            if first_tradeable is None:
                first_tradeable = q
    funnel = (f"{len(items)} raw -> {parsed_ok} parsed -> {in_band} in the "
             f"{MIN_LEVERAGE:g}x-{MAX_LEVERAGE:g}x leverage band")
    if in_band > 0:
        _ok(funnel)
    elif parsed_ok > 0:
        _fail(funnel + " — parses but every instrument falls outside the "
              "leverage band. Either genuinely no tradeable leverage exists "
              "for this symbol/category right now, or 'leverage' is under "
              "the wrong field (see candidates above).")
    else:
        _fail(funnel)
    return first_tradeable


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    _section("Credentials")
    phone, pin = _load_credentials()
    if not (phone and pin):
        _fail("TR_PHONE / TR_PIN not set (env or .env). Nothing else can run.")
        return 1
    masked = phone[:4] + "*" * max(0, len(phone) - 4)
    _ok(f"TR_PHONE={masked}  TR_PIN=({len(pin)} chars)")

    try:
        import pytr  # noqa: F401
    except ImportError:
        _fail("pytr is not installed — `pip install -e '.[dev]'`.")
        return 1
    _ok(f"pytr importable (version {getattr(pytr, '__version__', 'unknown')})")

    from lmtrade.brokers.tr_derivatives import (
        PytrDerivatives,
        _new_pytr_api,
        _recv_for,
        _resume_once,
    )

    _section("Session")
    try:
        api = _new_pytr_api(phone, pin)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Could not construct the pytr client: {exc!r}")
        return 1
    _ok("pytr.TradeRepublicApi constructed")

    if not _resume_once(api):
        _fail(
            "resume_websession() failed — no valid cached session cookie. "
            "Run: pytr login -n \"<TR_PHONE>\" -p \"<TR_PIN>\" --store_credentials"
        )
        return 1
    _ok("Session resumed from cached cookie")

    loop = asyncio.get_event_loop()

    _section("Cash")
    ok, result = loop.run_until_complete(
        _query(api, api.cash(), _recv_for))
    if ok:
        _ok("cash subscription answered")
        _dump("raw payload", result)
    else:
        _fail(f"cash subscription failed: {result!r}")

    _section("Portfolio (holdings) topics")
    _info(f"Trying each known topic in order: {', '.join(PytrDerivatives._PORTFOLIO_TOPICS)}")
    any_topic_ok = False
    for topic in PytrDerivatives._PORTFOLIO_TOPICS:

        async def _sub(t=topic):
            return await api.subscribe({"type": t})

        ok, result = loop.run_until_complete(_query(api, _sub(), _recv_for))
        if ok:
            any_topic_ok = True
            _ok(f"'{topic}' accepted")
            _dump(f"raw payload ({topic})", result)
        else:
            blob = f"{getattr(result, 'error', '')} {result}"
            if "BAD_SUBSCRIPTION_TYPE" in blob or "Unknown topic type" in blob:
                _fail(f"'{topic}' rejected (BAD_SUBSCRIPTION_TYPE — not this account's topic)")
            else:
                _fail(f"'{topic}' errored: {result!r}")
    if not any_topic_ok:
        _fail(
            "NO portfolio topic was accepted — position sync/reconciliation "
            "cannot work on this account. File the raw errors above."
        )

    _section(f"ISIN search ({symbol})")
    ok, result = loop.run_until_complete(
        _query(api, api.search(symbol, asset_type="stock"), _recv_for))
    isin = None
    matched_result = None
    if ok:
        _ok("search subscription answered")
        _dump("raw payload", result)
        for r in (result or {}).get("results", []):
            if r.get("isin"):
                isin = r["isin"]
                matched_result = r
                break
        if isin:
            _ok(f"Resolved {symbol} -> ISIN {isin}")
        else:
            _fail(f"No ISIN found for {symbol} in the results above.")
    else:
        _fail(f"search failed: {result!r}")

    if matched_result is not None:
        _section(f"Derivative product categories TR advertises for {symbol}")
        advertised = _product_categories(matched_result)
        known = {c for c, _, _ in PytrDerivatives._CATEGORIES}
        if advertised:
            _dump("derivativeProductCategories (from the ISIN search result "
                  "above — TR's own authoritative list, not a guess)",
                  advertised)
            unqueried = [c for c in advertised if c not in known]
            if unqueried:
                _fail(f"TR advertises {unqueried} for {symbol} but this bot "
                      f"does NOT query {'it' if len(unqueried) == 1 else 'them'} "
                      f"yet — add to PytrDerivatives._CATEGORIES in "
                      f"tr_derivatives.py if this should be tradeable.")
            else:
                _ok(f"This bot already queries every category TR advertises "
                    f"for {symbol}: {sorted(known)}.")
            if "vanillaOption" not in advertised and "option" not in advertised:
                _info("No plain/listed-option category advertised for this "
                      "symbol — 'vanillaWarrant' (Optionsschein) is the "
                      "closest TR product to a plain option: same "
                      "call/put/strike/expiry shape, but it's an issuer "
                      "certificate (Société Générale etc.), not an "
                      "exchange-listed contract, so external option-chain "
                      "data (yfinance etc.) won't have this exact ISIN.")
        else:
            _fail(f"No derivativeProductCategories field on the {symbol} "
                  f"search result — cannot tell what TR offers for it from "
                  f"this payload alone.")

    priced_candidate = None
    if isin:
        # Every product category the bot searches — a symbol with no
        # knockout/Turbo may still have a vanilla put/call warrant, and
        # that's common, not a bug. Reports raw results for each so a
        # rejected/guessed category name is immediately visible.
        for category, parser_name, label in PytrDerivatives._CATEGORIES:
            _section(f"{label.title()} derivative search ({symbol} / {isin}, category={category!r})")
            ok, result = loop.run_until_complete(
                _query(api, api.search_derivative(isin, category), _recv_for))
            if not ok:
                blob = f"{getattr(result, 'error', '')} {result}"
                if "BAD_SUBSCRIPTION_TYPE" in blob or "Unknown topic type" in blob:
                    _fail(f"category {category!r} rejected by this account")
                else:
                    _fail(f"search_derivative failed: {result!r}")
                continue
            items = (result or {}).get("results", [])
            _ok(f"search_derivative subscription answered — {len(items)} raw instrument(s)")
            if not items:
                _dump("raw payload (empty results)", result)
                continue
            candidate = _diagnose_derivative_items(items, symbol, parser_name)
            if priced_candidate is None:
                priced_candidate = candidate
    else:
        _info("Skipping derivative search — no ISIN resolved above.")

    if priced_candidate:
        _section(f"Live pricing ({priced_candidate.isin})")
        _info("Search results carry NO price at all (confirmed against a live "
              "account) — the bot fetches it separately, per selected "
              "instrument. priceForOrder (TR's own pre-order pricing "
              "subscription) is now tried FIRST — a live incident showed "
              "ticker(isin) can produce ZERO frames (not even an error) for "
              "some instruments. ticker() is queried too, as the fallback "
              "the bot itself falls back to. Both payload SHAPES are "
              "unverified; the parser tries several plausible shapes.")

        _section(f"priceForOrder({priced_candidate.isin}, 'LSX', 'buy')")
        ok, result = loop.run_until_complete(
            _query(api, api.price_for_order(priced_candidate.isin, "LSX", "buy"), _recv_for))
        if ok:
            _ok("priceForOrder subscription answered")
            _dump("raw payload", result)
            price = PytrDerivatives._parse_ticker_price(result)
            if price is not None:
                _ok(f"parser extracts price={price}")
            else:
                _fail("parser could not extract a price from this payload — "
                      "update PytrDerivatives._parse_ticker_price to match "
                      "the shape dumped above.")
        else:
            blob = f"{getattr(result, 'error', '')} {result}"
            if "BAD_SUBSCRIPTION_TYPE" in blob or "Unknown topic type" in blob:
                _fail("'priceForOrder' topic rejected by this account")
            elif isinstance(result, TimeoutError):
                _fail(f"priceForOrder timed out: {result}")
            else:
                _fail(f"priceForOrder fetch failed: {result!r}")

        _section(f"ticker({priced_candidate.isin}) [fallback]")
        ok, result = loop.run_until_complete(
            _query(api, api.ticker(priced_candidate.isin), _recv_for))
        if ok:
            _ok("ticker subscription answered")
            _dump("raw payload", result)
            price = PytrDerivatives._parse_ticker_price(result)
            if price is not None:
                _ok(f"parser extracts price={price}")
            else:
                _fail("parser could not extract a price from this payload — "
                      "update PytrDerivatives._parse_ticker_price to match "
                      "the shape dumped above.")
        else:
            blob = f"{getattr(result, 'error', '')} {result}"
            if "BAD_SUBSCRIPTION_TYPE" in blob or "Unknown topic type" in blob:
                _fail("'ticker' topic rejected by this account")
            elif isinstance(result, TimeoutError):
                _fail(f"ticker timed out: {result}")
            else:
                _fail(f"ticker fetch failed: {result!r}")
    elif isin:
        _info("Skipping ticker price check — no in-leverage-band instrument "
              "found above to price.")

    _section("Summary")
    print(
        "  This tool never places an order. If everything above passed but "
        "the bot still isn't filling: check the ORDER CONFIRMATION path "
        "(brokers/trade_republic.py) — TR sometimes acknowledges a "
        "submission with a warnings-only payload (no error, no order id) "
        "that must be resubmitted with warningsShown, and any response "
        "with neither an order id nor a recognized warning is logged as "
        "'unconfirmed' and treated as NOT placed. Check the bot's own logs "
        "around a real order attempt for that exact message."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
