"""Central logging setup. Logs go to stderr (rich) and are also mirrored into the
state store so the web dashboard can display them."""
from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler

_CONFIGURED = False


def _force_utf8_streams() -> None:
    """Make stdout/stderr tolerate non-ASCII (€, →, emoji in banners/logs).
    A legacy Windows console defaults to cp1252, which raises
    UnicodeEncodeError on those glyphs and can crash a log write mid-run.
    reconfigure() (Python 3.7+) switches the stream to UTF-8 and, failing
    that, replaces unencodable chars instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — best effort; never block startup
            pass


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _force_utf8_streams()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Quiet noisy third-party HTTP client loggers (one INFO line per request —
    # the engine fetches market data every cycle, which spams the console).
    # Set explicitly rather than relying on root inheritance: basicConfig()
    # is a no-op once root already has a handler (e.g. under a test runner's
    # log-capture plugin, or if another library configured logging first).
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
