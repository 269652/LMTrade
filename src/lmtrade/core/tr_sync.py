"""Reconcile the LIVE book against the real Trade Republic account.

Run at live-mode startup (and after a db reset): the live book must mirror
TR reality — real cash, real positions, nothing else.

- Cash: refresh the tr_account_cash meta (and the live P&L baseline).
- Options/knockouts: a local open row whose ISIN is not in the real TR
  portfolio is a PHANTOM (opened by an earlier bug, or closed/knocked out
  on TR's side while the bot was down) — deleted, not closed, so it never
  pollutes realized P&L. Synthetic rows (no ISIN) never belong in the live
  book at all.
- Equity rows: same rule — removed unless TR holds them; TR positions the
  book doesn't track are imported (symbol = ISIN, qty/avg from TR) so they
  are visible and counted in net worth.

If the TR portfolio is UNAVAILABLE (session down -> portfolio() is None),
nothing is deleted — reconciliation only acts on positive knowledge.
"""
from __future__ import annotations

from ..logging_setup import get_logger
from .state import Position, Store

log = get_logger("lmtrade.tr")


def sync_tr_portfolio(store: Store, tr) -> dict:
    """Reconcile `store` (the LIVE book) against the real TR account via the
    client `tr` (anything with account_cash() and portfolio()). Returns a
    summary dict {cash, imported, removed_options, removed_positions}."""
    summary = {"cash": None, "imported": 0, "removed_options": 0,
               "removed_positions": 0}

    cash = tr.account_cash()
    if cash is not None:
        store.set_meta("tr_account_cash", cash)
        if store.get_meta("tr_baseline_net_worth") is None:
            store.set_meta("tr_baseline_net_worth", cash)
        summary["cash"] = cash

    portfolio = tr.portfolio()
    if portfolio is None:
        log.warning("TR portfolio unavailable — skipping live-book "
                    "reconciliation (nothing deleted without data).")
        return summary

    tr_by_isin = {p["isin"]: p for p in portfolio}

    # Phantom option/knockout rows: not present in the real account.
    for o in store.open_options():
        isin = o.get("isin")
        if isin not in tr_by_isin:
            store.delete_option(o["id"])
            summary["removed_options"] += 1
            log.info("Reconciled away phantom live position %s (%s) — not in "
                     "the real TR portfolio.", o["underlying"], isin or "no ISIN")

    # Pending live orders: a process crash between placing a real order and
    # confirming it (mark_option_open) leaves a 'pending' row. TR's own
    # portfolio is authoritative — if the fill really happened, promote it;
    # if TR never has it, it's a phantom, same treatment as a stale open row.
    for o in store.pending_options():
        isin = o.get("isin")
        if isin in tr_by_isin:
            store.mark_option_open(o["id"])
            log.info("Reconciled pending order %s (%s) as confirmed — TR "
                     "portfolio holds it.", o["underlying"], isin)
        else:
            store.delete_option(o["id"])
            summary["removed_options"] += 1
            log.info("Reconciled away phantom pending order %s (%s) — not in "
                     "the real TR portfolio.", o["underlying"], isin or "no ISIN")

    tracked = {o.get("isin") for o in store.open_options()}

    # Equity rows: drop what TR doesn't hold, import what it does.
    for pos in store.positions():
        if pos.symbol not in tr_by_isin:
            store.upsert_position(Position(pos.symbol, 0.0, pos.avg_price,
                                           pos.opened_ts))
            summary["removed_positions"] += 1
    for isin, p in tr_by_isin.items():
        if isin in tracked or store.position(isin) is not None:
            continue
        store.upsert_position(Position(isin, p["size"], p["avg_price"], 0.0))
        summary["imported"] += 1
        log.info("Imported TR position %s x%s @ %s into the live book.",
                 isin, p["size"], p["avg_price"])

    return summary
