"""Hybrid decision fusion.

Gathers a Signal from every provider in the stack, weights them per config, and
produces a single Decision. This is where "LLMs + SLMs + financial models" are
actually combined into one action.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..data.market import Quote
from ..finance.indicators import indicator_snapshot
from ..models.base import ModelProvider, Signal
from ..models.providers import PerplexityProvider


@dataclass
class Decision:
    symbol: str
    direction: str                 # buy | sell | hold
    confidence: float              # 0..1
    rationale: str
    signals: list[Signal] = field(default_factory=list)
    inference_cost: float = 0.0    # total USD spent producing this decision

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "inference_cost": round(self.inference_cost, 6),
            "signals": [
                {
                    "provider": s.provider,
                    "direction": s.direction,
                    "confidence": round(s.confidence, 3),
                    "rationale": s.rationale,
                    "cost_usd": round(s.cost_usd, 6),
                }
                for s in self.signals
            ],
        }


class FusionEngine:
    def __init__(self, settings: Settings, providers: dict[str, ModelProvider]):
        self.settings = settings
        self.providers = providers
        self.weights = settings.model.weights

    def decide(self, quote: Quote) -> Decision:
        context: dict = {"indicators": indicator_snapshot(quote.history)}

        # Research first (Perplexity) so its notes feed the SLM/cloud prompts.
        cost = 0.0
        perp = self.providers.get("perplexity")
        if isinstance(perp, PerplexityProvider) and perp.available():
            research, rcost = perp.research(quote.symbol)
            context["research"] = research
            cost += rcost

        signals: list[Signal] = []
        for name, provider in self.providers.items():
            try:
                sig = provider.analyze(quote.symbol, context)
            except Exception as exc:  # noqa: BLE001
                sig = Signal(name, "hold", 0.5, f"error: {exc}", 0.0)
            signals.append(sig)
            cost += sig.cost_usd

        # Weighted vote. A `hold` is an ABSTENTION, not a vote — it must not
        # dilute the conviction of the providers that did take a side, otherwise
        # every unavailable provider (no key/host) would drag the decision toward
        # inaction. Only directional signals contribute to the denominator.
        num = 0.0
        den = 0.0
        for s in signals:
            if s.direction == "hold":
                continue
            w = self.weights.get(s.provider, 1.0)
            num += w * s.signed()
            den += w * max(1e-6, s.confidence)
        net = num / den if den else 0.0        # -1..1

        direction = "buy" if net > 0.1 else "sell" if net < -0.1 else "hold"
        confidence = min(1.0, abs(net))
        agree = [s.provider for s in signals if s.direction == direction]
        rationale = (
            f"net={net:+.2f} via {len(signals)} providers; "
            f"agree: {', '.join(agree) or 'none'}"
        )
        return Decision(quote.symbol, direction, confidence, rationale, signals, cost)
