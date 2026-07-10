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


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    _section("Credentials")
    phone = secret("TR_PHONE")
    pin = secret("TR_PIN")
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
        _section(f"Knockout derivative search ({symbol} / {isin})")
        ok, result = loop.run_until_complete(
            _query(api, api.search_derivative(isin, PytrDerivatives.PRODUCT_CATEGORY), _recv_for))
        if ok:
            items = (result or {}).get("results", [])
            _ok(f"search_derivative subscription answered — {len(items)} instrument(s)")
            if items:
                _dump("first raw instrument (check field names against "
                      "tr_derivatives.py's parsing)", items[0])
            else:
                _dump("raw payload (empty results)", result)
        else:
            _fail(f"search_derivative failed: {result!r}")
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
