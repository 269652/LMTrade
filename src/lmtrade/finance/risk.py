"""Position sizing and risk guardrails. Deliberately conservative — the whole
account is ~10 EUR, so a single fat-fingered size can wipe it out."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import RiskConfig


@dataclass
class SizingResult:
    qty: float
    notional: float
    reason: str


def size_position(
    *,
    price: float,
    cash: float,
    equity: float,
    confidence: float,
    cfg: RiskConfig,
) -> SizingResult:
    """Confidence-scaled fractional sizing, capped by max_position_fraction and
    available cash. Fractional share quantities are allowed (TR supports them)."""
    if price <= 0:
        return SizingResult(0.0, 0.0, "invalid price")
    if confidence < cfg.min_confidence:
        return SizingResult(0.0, 0.0, f"confidence {confidence:.2f} < {cfg.min_confidence}")

    # Fraction of equity to allocate, scaled by how far above the threshold we are.
    span = max(1e-6, 1.0 - cfg.min_confidence)
    scale = (confidence - cfg.min_confidence) / span      # 0..1
    fraction = cfg.max_position_fraction * scale
    notional = min(equity * fraction, cash)
    if notional < 1e-3:
        return SizingResult(0.0, 0.0, "notional below minimum")
    qty = notional / price
    return SizingResult(qty, notional, f"alloc {fraction:.0%} of equity @ conf {confidence:.2f}")


def should_exit(
    *, avg_price: float, last_price: float, cfg: RiskConfig
) -> tuple[bool, str]:
    """Hard stop-loss / take-profit check for an open long position."""
    if avg_price <= 0:
        return False, ""
    change = (last_price - avg_price) / avg_price
    if change <= -cfg.stop_loss_pct:
        return True, f"stop-loss hit ({change:.1%})"
    if change >= cfg.take_profit_pct:
        return True, f"take-profit hit ({change:.1%})"
    return False, ""
