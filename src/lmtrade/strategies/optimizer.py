"""Evolutionary strategy learner.

Maintains a population of parameter genomes across the strategy library. The
engine asks `select()` for the genome to trade with (epsilon-greedy: mostly the
best fitness, sometimes explore), attributes each closed trade's realized P&L
back via `record_result()`, and periodically calls `evolve()` — which replaces
the worst performer with a mutated copy of the best. Over weeks of paper
trading this is how the bot "learns" which strategies and parameters work.

Everything is persisted in the Store (meta key "genomes") so learning survives
restarts and is visible to the dashboard.
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import asdict, dataclass, field

from ..core.state import Store
from .library import STRATEGIES, signal_for

GENOMES_KEY = "genomes"


@dataclass
class Genome:
    id: str
    strategy: str
    params: dict
    trades: int = 0
    pnl: float = 0.0
    # Win/loss breakdown for Kelly sizing (defaults keep genomes persisted
    # before these fields existed loading cleanly).
    wins: int = 0
    losses: int = 0
    win_sum: float = 0.0     # sum of winning P&Ls (positive)
    loss_sum: float = 0.0    # sum of |losing P&Ls| (positive)

    @property
    def fitness(self) -> float:
        """Average P&L per trade; unproven genomes get a small optimistic prior
        so they get explored before being written off."""
        if self.trades == 0:
            return 0.01
        return self.pnl / self.trades

    @property
    def win_rate(self) -> float:
        settled = self.wins + self.losses
        return self.wins / settled if settled else 0.0

    @property
    def avg_win(self) -> float:
        return self.win_sum / self.wins if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / self.losses if self.losses else 0.0

    def signal(self, history: list[float]) -> tuple[str, float]:
        return signal_for(self.strategy, history, self.params)


class StrategyOptimizer:
    def __init__(
        self, store: Store, population: int = 8, epsilon: float = 0.2,
        mutation_scale: float = 0.3, rng: random.Random | None = None,
        ucb_c: float = 2.0,
    ):
        self.store = store
        self.population = population
        self.epsilon = epsilon
        self.mutation_scale = mutation_scale
        self.ucb_c = ucb_c
        self.rng = rng or random.Random()
        if store.get_meta(GENOMES_KEY) is None:
            self._save(self._seed_population())

    # -- persistence ------------------------------------------------------------
    def genomes(self) -> list[Genome]:
        raw = self.store.get_meta(GENOMES_KEY, [])
        return [Genome(**g) for g in raw]

    def _save(self, genomes: list[Genome]) -> None:
        self.store.set_meta(GENOMES_KEY, [asdict(g) for g in genomes])

    # -- population -------------------------------------------------------------
    def _seed_population(self) -> list[Genome]:
        names = list(STRATEGIES)
        out: list[Genome] = []
        for i in range(self.population):
            name = names[i % len(names)]
            params = dict(STRATEGIES[name].default_params)
            if i >= len(names):        # later seeds start mutated for diversity
                params = self._mutate_params(name, params)
            out.append(Genome(id=uuid.uuid4().hex[:8], strategy=name, params=params))
        return out

    def _mutate_params(self, strategy: str, params: dict) -> dict:
        spec = STRATEGIES[strategy]
        out = dict(params)
        for key, (lo, hi) in spec.bounds.items():
            if self.rng.random() < 0.7:
                cur = float(out.get(key, (lo + hi) / 2))
                jitter = (hi - lo) * self.mutation_scale * (self.rng.random() * 2 - 1)
                val = max(lo, min(hi, cur + jitter))
                out[key] = int(round(val)) if isinstance(lo, int) else round(val, 5)
        return out

    # -- learning API -------------------------------------------------------------
    def select(self) -> Genome:
        """UCB1 selection (Auer, Cesa-Bianchi & Fischer 2002), with a small
        epsilon of pure random exploration retained on top.

        Epsilon-greedy alone had a winner-takes-all pathology observed live:
        the first genome to beat the 0.01 optimistic prior won every exploit
        pick, and 20% exploration split across the rest (further filtered by
        confidence gates and slot limits) never completed a trade for them —
        one genome had all the trades, seven had zero. UCB1's exploration
        bonus c*sqrt(ln N / n) guarantees untried genomes are sampled first
        and under-sampled ones are revisited at a logarithmic rate."""
        genomes = self.genomes()
        if self.rng.random() < self.epsilon:
            return self.rng.choice(genomes)
        untried = [g for g in genomes if g.trades == 0]
        if untried:
            return self.rng.choice(untried)
        total = sum(g.trades for g in genomes)
        return max(genomes, key=lambda g: g.fitness
                   + self.ucb_c * math.sqrt(math.log(max(2, total)) / g.trades))

    def record_result(self, genome_id: str, pnl: float) -> None:
        genomes = self.genomes()
        for g in genomes:
            if g.id == genome_id:
                g.trades += 1
                g.pnl += pnl
                if pnl > 0:
                    g.wins += 1
                    g.win_sum += pnl
                elif pnl < 0:
                    g.losses += 1
                    g.loss_sum += -pnl
                break
        self._save(genomes)

    def evolve(self, min_trades: int = 3) -> Genome | None:
        """Replace the worst proven genome with a mutated copy of the best.
        Returns the new genome, or None if not enough evidence yet.

        Species protection (simplified niching): the mutant inherits the best
        genome's strategy, so unchecked evolution collapses the population to
        clones of one family and the learner can never rediscover a regime
        where another family works. A family's LAST member is never replaced —
        the worst genome whose family still has siblings goes instead."""
        genomes = self.genomes()
        proven = [g for g in genomes if g.trades >= min_trades]
        if len(proven) < 2:
            return None
        best = max(proven, key=lambda g: g.fitness)
        family_counts: dict[str, int] = {}
        for g in genomes:
            family_counts[g.strategy] = family_counts.get(g.strategy, 0) + 1
        replaceable = [g for g in proven
                       if g.id != best.id and family_counts[g.strategy] >= 2]
        if not replaceable:
            return None
        worst = min(replaceable, key=lambda g: g.fitness)
        mutant = Genome(
            id=uuid.uuid4().hex[:8],
            strategy=best.strategy,
            params=self._mutate_params(best.strategy, best.params),
        )
        genomes = [g for g in genomes if g.id != worst.id] + [mutant]
        self._save(genomes)
        return mutant

    def leaderboard(self) -> list[dict]:
        """Fitness-sorted view for the dashboard/status output."""
        return [
            {"id": g.id, "strategy": g.strategy, "params": g.params,
             "trades": g.trades, "pnl": round(g.pnl, 4),
             "fitness": round(g.fitness, 5)}
            for g in sorted(self.genomes(), key=lambda g: g.fitness, reverse=True)
        ]
