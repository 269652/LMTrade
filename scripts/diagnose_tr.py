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


def _diagnose_derivative_items(items: list, symbol: str, parser_name: str) -> None:
    """For a sample of raw instruments: dump the full raw JSON, run the
    bot's ACTUAL parser against it (showing what it extracts or why it
    fails), and call out fields whose NAME hints at strike/ask/leverage/
    barrier/ratio/expiry — even under a different name than currently
    guessed — so the real field mapping is visible without a round trip.
    Finishes with a raw -> parsed -> tradeable funnel so a 'parses fine but
    never tradeable' mismatch (wrong ask/leverage field, defaulting to 0) is
    immediately obvious rather than looking like a healthy 'usable' count."""
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
            _dump("fields whose NAME hints strike/ask/leverage/barrier/ratio/"
                  "expiry (compare against tr_derivatives.py's parsing even "
                  "if the name differs from what's currently guessed)",
                  candidates)
        try:
            q = parser(item, symbol, "buy")
            _ok(f"parser extracts: price={q.price} leverage={q.leverage} "
                f"strike={q.strike} barrier={q.barrier}")
        except (KeyError, TypeError, ValueError) as exc:
            _fail(f"parser failed on this item: {exc!r} — a required field "
                  "is missing or under a different name (see candidates above).")

    parsed_ok = tradeable = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            q = parser(item, symbol, "buy")
        except (KeyError, TypeError, ValueError):
            continue
        parsed_ok += 1
        if q.price > 0 and MIN_LEVERAGE <= q.leverage <= MAX_LEVERAGE:
            tradeable += 1
    funnel = (f"{len(items)} raw -> {parsed_ok} parsed -> {tradeable} tradeable "
             f"(price>0 and {MIN_LEVERAGE:g}x <= leverage <= {MAX_LEVERAGE:g}x)")
    if tradeable > 0:
        _ok(funnel)
    elif parsed_ok > 0:
        _fail(funnel + " — parses but NEVER passes the tradeability filter. "
              "Compare the candidate fields dumped above against "
              "tr_derivatives.py's _parse_knockout_item/_parse_vanilla_item "
              "('ask' and 'leverage' are the current guesses) and fix the "
              "mapping if the real field has a different name.")
    else:
        _fail(funnel)


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
    if ok:
        _ok("search subscription answered")
        _dump("raw payload", result)
        for r in (result or {}).get("results", []):
            if r.get("isin"):
                isin = r["isin"]
                break
        if isin:
            _ok(f"Resolved {symbol} -> ISIN {isin}")
        else:
            _fail(f"No ISIN found for {symbol} in the results above.")
    else:
        _fail(f"search failed: {result!r}")

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
            _diagnose_derivative_items(items, symbol, parser_name)
    else:
        _info("Skipping derivative search — no ISIN resolved above.")

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
