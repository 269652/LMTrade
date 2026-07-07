"""Model-provider interface. Every provider — a financial heuristic, a local
SLM, a cloud LLM, or Perplexity research — returns the same `Signal` so the
fusion engine can treat them uniformly and price their inference cost."""
from __future__ import annotations

from dataclasses import dataclass, field

DIRECTIONS = ("buy", "sell", "hold")


@dataclass
class Signal:
    provider: str
    direction: str                 # buy | sell | hold
    confidence: float              # 0..1
    rationale: str = ""
    cost_usd: float = 0.0          # marginal inference cost of producing it
    meta: dict = field(default_factory=dict)

    def signed(self) -> float:
        """+conf for buy, -conf for sell, 0 for hold."""
        if self.direction == "buy":
            return self.confidence
        if self.direction == "sell":
            return -self.confidence
        return 0.0


class ModelProvider:
    name: str = "base"

    def available(self) -> bool:
        """Whether this provider can actually run (keys present, host up …)."""
        return True

    def analyze(self, symbol: str, context: dict) -> Signal:
        raise NotImplementedError
