"""Fusion confidence calibration.

Regression suite for a real defect: the weighted vote computed
net = Σ w·(±conf) / Σ w·conf, so each voter's confidence CANCELLED — a lone
directional voter always produced |net| = 1.0 and the decision printed
conf 1.00 regardless of the voter's actual conviction (observed live: every
decision at 1.00/0.00). That saturates every downstream gate that consumes
confidence (risk.min_confidence, storm extra-conviction, confidence-scaled
sizing, slot ranking by confidence).

Correct behavior: net = Σ w·(±conf) / Σ w over DIRECTIONAL voters only —
confidence propagates, holds stay abstentions. Written before the fix (TDD).
"""
from __future__ import annotations

import pytest

from lmtrade.agents.fusion import FusionEngine
from lmtrade.config import Settings
from lmtrade.data.market import Quote
from lmtrade.models.base import ModelProvider, Signal


class Scripted(ModelProvider):
    """Provider that returns a fixed signal."""

    def __init__(self, name: str, direction: str, confidence: float):
        self.name = name
        self._sig = (direction, confidence)

    def analyze(self, symbol: str, context: dict) -> Signal:
        d, c = self._sig
        return Signal(self.name, d, c, "scripted", 0.0)


def make_engine(providers: dict[str, ModelProvider],
                weights: dict[str, float] | None = None) -> FusionEngine:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"])
    s.model.stack = list(providers.keys())
    s.model.weights = weights or {name: 1.0 for name in providers}
    return FusionEngine(s, providers)


QUOTE = Quote("AAPL", 100.0, [100.0] * 60, "synthetic")


class TestLoneVoterCalibration:
    def test_lone_voter_confidence_propagates(self):
        eng = make_engine({"a": Scripted("a", "buy", 0.6)})
        d = eng.decide(QUOTE)
        assert d.direction == "buy"
        assert d.confidence == pytest.approx(0.6, abs=1e-6)   # NOT 1.0

    def test_lone_weak_voter_below_threshold_stays_weak(self):
        eng = make_engine({"a": Scripted("a", "sell", 0.2)})
        d = eng.decide(QUOTE)
        assert d.direction == "sell"
        assert d.confidence == pytest.approx(0.2, abs=1e-6)

    def test_lone_voter_barely_directional_is_hold(self):
        # |net| = 0.05 is under the 0.1 direction threshold.
        eng = make_engine({"a": Scripted("a", "buy", 0.05)})
        d = eng.decide(QUOTE)
        assert d.direction == "hold"


class TestAggregation:
    def test_agreement_averages_confidence(self):
        eng = make_engine({"a": Scripted("a", "buy", 0.6),
                           "b": Scripted("b", "buy", 0.8)})
        d = eng.decide(QUOTE)
        assert d.direction == "buy"
        assert d.confidence == pytest.approx(0.7, abs=1e-6)

    def test_full_disagreement_cancels_to_hold(self):
        eng = make_engine({"a": Scripted("a", "buy", 0.9),
                           "b": Scripted("b", "sell", 0.9)})
        d = eng.decide(QUOTE)
        assert d.direction == "hold"
        assert d.confidence == pytest.approx(0.0, abs=1e-6)

    def test_weights_tilt_the_vote(self):
        eng = make_engine({"a": Scripted("a", "buy", 0.6),
                           "b": Scripted("b", "sell", 0.6)},
                          weights={"a": 3.0, "b": 1.0})
        d = eng.decide(QUOTE)
        # net = (3*0.6 - 1*0.6) / 4 = 0.3
        assert d.direction == "buy"
        assert d.confidence == pytest.approx(0.3, abs=1e-6)

    def test_hold_is_abstention_not_dilution(self):
        eng = make_engine({"a": Scripted("a", "buy", 0.8),
                           "h1": Scripted("h1", "hold", 0.5),
                           "h2": Scripted("h2", "hold", 0.5)})
        d = eng.decide(QUOTE)
        assert d.direction == "buy"
        assert d.confidence == pytest.approx(0.8, abs=1e-6)   # holds don't drag

    def test_no_directional_votes_is_hold(self):
        eng = make_engine({"h": Scripted("h", "hold", 0.5)})
        d = eng.decide(QUOTE)
        assert d.direction == "hold"
        assert d.confidence == pytest.approx(0.0, abs=1e-6)
