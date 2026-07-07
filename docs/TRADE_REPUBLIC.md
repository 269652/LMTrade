# Trade Republic integration (live mode)

> ⚠️ **Read this before enabling live trading.**

## There is no official API

Trade Republic does **not** publish a trading API. The only programmatic access
is via **reverse-engineered clients** such as [`pytr`](https://github.com/pytr-org/pytr),
which speak TR's private mobile app protocol.

**Consequences you accept by enabling live mode:**

- It likely **violates Trade Republic's Terms of Service**.
- Your account can be **flagged, frozen, or closed**.
- The private API can change without notice and break execution mid-trade.
- There is **no support** and no guarantee orders fill as intended.

For these reasons LMTrade defaults to **paper mode**, and the live adapter ships
with order submission **disabled by a deliberate guard** in
`src/lmtrade/brokers/trade_republic.py`. Enabling it is a conscious, reviewed
step — not a config flip.

## If you still want to enable it

1. Install the optional extra:
   ```bash
   pip install 'lmtrade[traderepublic]'
   ```
2. Put credentials in `.env`:
   ```
   LMTRADE_MODE=live
   TR_PHONE=+49...
   TR_PIN=1234
   ```
3. Complete the **one-time 2FA pairing**. `pytr` triggers an app/SMS challenge on
   first login; you must run an interactive login once and persist the cookie so
   the unattended bot can reuse the session. See pytr's docs for `pytr login`.
4. Review and implement the guarded `_place()` method against **your** installed
   `pytr` version (field names and order endpoints vary by version). Test with
   the **smallest possible** order first.
5. Only remove the guard once you've verified execution against a funded account
   and accept the risk.

## Recommended safer alternatives

- **Stay in paper mode** with live market data to validate the strategy first.
- Use a broker that offers an **official API** (e.g. Alpaca, Interactive
  Brokers) — the `Broker` interface in `brokers/base.py` is designed so you can
  drop in a compliant adapter without touching the engine.

## The €10 reality

Trade Republic charges a flat **€1 external fee** on most orders. On a €10
account that is a **10% round-trip drag** — the paper broker models this fee so
your simulated P&L is honest. A strategy must clear that hurdle *and* the GPU
bill before it is genuinely self-sustaining.
