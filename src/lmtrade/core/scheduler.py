"""Interval scheduler for periodic jobs (hourly news, daily analysis).

Last-run timestamps persist in the Store, so schedules survive restarts. The
clock is injectable for deterministic tests.
"""
from __future__ import annotations

import time
from typing import Callable

from .state import Store

_KEY = "scheduler_last_run"


class Scheduler:
    def __init__(self, store: Store, now: Callable[[], float] = time.time):
        self.store = store
        self.now = now

    def due(self, name: str, interval_seconds: float) -> bool:
        """True if `name` has never run or its interval has elapsed. Marks the
        job as run when it returns True — call it once per dispatch site."""
        last_runs: dict = self.store.get_meta(_KEY, {})
        last = last_runs.get(name)
        current = self.now()
        if last is not None and (current - float(last)) < interval_seconds:
            return False
        last_runs[name] = current
        self.store.set_meta(_KEY, last_runs)
        return True
