"""Daily strategy review by a frontier Claude model.

Once a day the analyst compiles the bot's recent performance (trades, P&L,
strategy leaderboard, news) into a report, asks Claude for bounded parameter
adjustments, applies them inside hard safety clamps, and records the insight
for the dashboard. The API caller is injectable so tests run offline; without
an ANTHROPIC_API_KEY the analyst is a graceful no-op.
"""
from __future__ import annotations

import json
import re
from typing import Callable

import httpx

from ..config import Settings, secret
from ..core.state import Store
from ..models.providers import ClaudeCLIProvider

# caller(prompt) -> (response_text, cost_usd)
Caller = Callable[[str], tuple[str, float]]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Hard clamps: whatever the model says, risk parameters stay in this envelope.
CLAMPS = {
    "stop_loss_pct": (0.02, 0.15),
    "take_profit_pct": (0.03, 0.30),
    "min_confidence": (0.40, 0.90),
    "max_position_fraction": (0.10, 0.60),
}


def _anthropic_caller(settings: Settings) -> Caller | None:
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None

    def call(prompt: str) -> tuple[str, float]:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": settings.model.cloud_model, "max_tokens": 700,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60.0,
        )
        r.raise_for_status()
        blocks = r.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text, settings.economics.inference_cost.get("cloud", 0.004)

    return call


def _claude_cli_caller(settings: Settings) -> Caller | None:
    provider = ClaudeCLIProvider(settings)
    return provider.ask if provider.available() else None


class DailyAnalyst:
    def __init__(self, store: Store, settings: Settings, caller: Caller | None = None):
        self.store = store
        self.settings = settings
        if caller is not None:
            self.caller = caller
        elif settings.research.analysis_provider == "claude_cli":
            self.caller = _claude_cli_caller(settings)
        else:
            self.caller = _anthropic_caller(settings)

    # -- report building ---------------------------------------------------------
    def build_report(self) -> str:
        trades = self.store.recent_trades(30)
        closed_opts = self.store.closed_options(30)
        news = self.store.recent_news(10)
        econ = self.store.get_meta("economics", {})
        genomes = self.store.get_meta("genomes", [])
        risk = self.settings.risk.model_dump()
        return json.dumps({
            "economics": econ,
            "current_risk_params": risk,
            "recent_equity_trades": trades[:15],
            "recent_option_trades": closed_opts[:15],
            "strategy_genomes": genomes,
            "recent_news": [{"symbol": n["symbol"], "sentiment": n["sentiment"]}
                            for n in news],
        }, default=str)

    def _prompt(self) -> str:
        return (
            "You are the daily strategy reviewer for a small automated paper-trading "
            "bot (~10 EUR account, options + equities, must eventually cover its GPU "
            "costs). If you have web search available, use it to ground your review "
            "in today's actual market conditions rather than prior/training "
            "knowledge. Here is its state and recent performance as JSON:\n\n"
            f"{self.build_report()}\n\n"
            "Suggest conservative adjustments to its risk parameters for the next "
            "24h. Respond ONLY with JSON of the form: "
            '{"risk": {"stop_loss_pct": float, "take_profit_pct": float, '
            '"min_confidence": float, "max_position_fraction": float}, '
            '"notes": "one-paragraph reasoning"} — include only the keys you want '
            "to change."
        )

    # -- run -----------------------------------------------------------------------
    def run(self) -> dict | None:
        """Execute the daily review. Returns applied adjustments, or None if the
        analyst is unavailable or the response was unusable."""
        if self.caller is None:
            return None
        try:
            text, cost = self.caller(self._prompt())
        except Exception:  # noqa: BLE001 — analysis must never crash the engine
            return None
        if cost > 0:
            self.store.record_cost("inference", cost, "claude")

        m = _JSON_RE.search(text or "")
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

        applied: dict[str, float] = {}
        for key, value in (obj.get("risk") or {}).items():
            if key not in CLAMPS:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            lo, hi = CLAMPS[key]
            v = max(lo, min(hi, v))
            setattr(self.settings.risk, key, v)
            applied[key] = v

        notes = str(obj.get("notes", ""))[:600]
        self.store.add_activity(
            "insight",
            f"daily review: {len(applied)} risk params adjusted",
            detail={"applied": applied, "notes": notes},
        )
        return {"applied": applied, "notes": notes}
