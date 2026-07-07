"""SQLite-backed state store shared by the engine and the web dashboard.

Everything the dashboard shows (trades, positions, logs, activity feed, equity
and cost history) is persisted here so the web process can read a consistent
snapshot without touching the engine's memory.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL,          -- buy | sell
    qty        REAL NOT NULL,
    price      REAL NOT NULL,
    fee        REAL NOT NULL DEFAULT 0,
    mode       TEXT NOT NULL,          -- paper | live
    reason     TEXT,                   -- fused rationale
    confidence REAL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol    TEXT PRIMARY KEY,
    qty       REAL NOT NULL,
    avg_price REAL NOT NULL,
    opened_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    source  TEXT,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activity (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    kind    TEXT NOT NULL,             -- decision | signal | trade | risk | economics | system
    symbol  TEXT,
    summary TEXT NOT NULL,
    detail  TEXT                       -- JSON blob
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts     REAL PRIMARY KEY,
    cash   REAL NOT NULL,
    equity REAL NOT NULL,             -- cash + mark-to-market positions
    fees   REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS costs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,           -- gpu | inference | fee
    provider TEXT,
    amount   REAL NOT NULL            -- USD
);
"""


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    opened_ts: float


@dataclass
class Trade:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0
    mode: str = "paper"
    reason: str | None = None
    confidence: float | None = None
    ts: float = field(default_factory=time.time)


class Store:
    """Thread-safe SQLite wrapper. One connection guarded by a lock — fine for
    the modest write volume of a small bot, and keeps the web read path simple."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- meta -----------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # -- logs -----------------------------------------------------------------
    def add_log(self, level: str, message: str, source: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO logs(ts,level,source,message) VALUES(?,?,?,?)",
                (time.time(), level, source, message),
            )
            self._conn.commit()

    def recent_logs(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- activity -------------------------------------------------------------
    def add_activity(
        self, kind: str, summary: str, symbol: str | None = None, detail: dict | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO activity(ts,kind,symbol,summary,detail) VALUES(?,?,?,?,?)",
                (time.time(), kind, symbol, summary, json.dumps(detail or {})),
            )
            self._conn.commit()

    def recent_activity(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d["detail"]) if d["detail"] else {}
            except json.JSONDecodeError:
                d["detail"] = {}
            out.append(d)
        return out

    # -- trades ---------------------------------------------------------------
    def record_trade(self, trade: Trade) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO trades(ts,symbol,side,qty,price,fee,mode,reason,confidence) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    trade.ts, trade.symbol, trade.side, trade.qty, trade.price,
                    trade.fee, trade.mode, trade.reason, trade.confidence,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_trades(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- positions ------------------------------------------------------------
    def upsert_position(self, pos: Position) -> None:
        with self._lock:
            if pos.qty <= 1e-9:
                self._conn.execute("DELETE FROM positions WHERE symbol=?", (pos.symbol,))
            else:
                self._conn.execute(
                    "INSERT INTO positions(symbol,qty,avg_price,opened_ts) VALUES(?,?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, "
                    "avg_price=excluded.avg_price",
                    (pos.symbol, pos.qty, pos.avg_price, pos.opened_ts),
                )
            self._conn.commit()

    def positions(self) -> list[Position]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM positions").fetchall()
        return [Position(**dict(r)) for r in rows]

    def position(self, symbol: str) -> Position | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM positions WHERE symbol=?", (symbol,)
            ).fetchone()
        return Position(**dict(row)) if row else None

    # -- equity & costs -------------------------------------------------------
    def record_equity(self, cash: float, equity: float, fees: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO equity_curve(ts,cash,equity,fees) VALUES(?,?,?,?) "
                "ON CONFLICT(ts) DO NOTHING",
                (time.time(), cash, equity, fees),
            )
            self._conn.commit()

    def equity_curve(self, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM equity_curve ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def record_cost(self, kind: str, amount: float, provider: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO costs(ts,kind,provider,amount) VALUES(?,?,?,?)",
                (time.time(), kind, provider, amount),
            )
            self._conn.commit()

    def total_costs(self) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, SUM(amount) AS total FROM costs GROUP BY kind"
            ).fetchall()
        return {r["kind"]: r["total"] for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
