"""Test that noisy third-party HTTP loggers (httpx/httpcore) are quieted so
per-request "HTTP Request: GET ..." lines don't spam the console every engine
cycle. Written before the fix per strict TDD."""
from __future__ import annotations

import logging

from lmtrade.logging_setup import setup_logging


def test_httpx_and_httpcore_are_quieted():
    """httpx logs an INFO line per HTTP request; at the engine's cadence
    (a Yahoo Finance fetch per symbol per cycle) that spams the console.
    Levels must be set explicitly via setLevel(), not left to inherit from
    root — basicConfig() is a no-op once any handler exists on root (e.g.
    under pytest's own log-capture plugin, or if another library configured
    logging first), so relying on inheritance is fragile."""
    import lmtrade.logging_setup as mod
    mod._CONFIGURED = False

    setup_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_idempotent_second_call_is_a_noop():
    """setup_logging() must be safe to call repeatedly (every get_logger()
    call triggers it) without raising or duplicating handlers."""
    import lmtrade.logging_setup as mod
    mod._CONFIGURED = False

    setup_logging()
    handlers_after_first = len(logging.getLogger().handlers)
    setup_logging()
    assert len(logging.getLogger().handlers) == handlers_after_first
