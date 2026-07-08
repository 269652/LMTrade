"""Central logging setup. Logs go to stderr (rich) and are also mirrored into the
state store so the web dashboard can display them."""
from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
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
