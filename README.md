# ⚡ LMTrade

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/269652/LMTrade/blob/claude/trading-bot-hybrid-agent-tvs90n/notebooks/LMTrade_Colab.ipynb)

A **self-sustaining trading bot** that fuses **LLMs, SLMs and classical financial
models** into one decision, and is designed to **pay for its own GPU costs**.

It runs on a rented [Vast.ai](https://vast.ai) GPU, serves a small language model
locally (via Ollama), consults a cloud LLM for hard calls, and uses **Perplexity**
for web-grounded research. A cost-accounting core continuously measures the
hourly GPU burn against realised P&L and **halts trading when the runway runs
out** — that guardrail is what turns "pays for its own GPU" from a slogan into an
enforced rule.

> **Safety first.** LMTrade runs in **paper mode** by default (simulated fills,
> no real money). Live Trade Republic execution is an explicit, guarded opt-in —
> see the honest caveats in [`docs/TRADE_REPUBLIC.md`](docs/TRADE_REPUBLIC.md).

---

## What's in the box

| Layer | Module | What it does |
|-------|--------|--------------|
| **CLI** | `lmtrade.cli` | `run`, `backtest`, `web`, `viz`, `status`, `analyze`, `deploy`, `reset`, `config` |
| **Backtest** | `backtest.walk_forward` | Walk-forward folds over historical bars; pre-trains the genome population |
| **Web dashboard** | `lmtrade.web` | Portfolio, economics, trades, options, leaderboard, benchmark, news, logs |
| **Inline viz** | `lmtrade.viz` | Notebook-native matplotlib dashboard (Colab/Jupyter) + `lmtrade viz` PNG |
| **Engine** | `core.engine` | High-cadence loop: research jobs → data → economics gate → fusion+strategy → options/equity execution |
| **Options** | `finance.options` | Black-Scholes pricing/greeks, synthetic near-ATM chain, mark-to-market |
| **Learning** | `strategies.*` | Momentum/mean-reversion/breakout genomes; evolutionary optimizer driven by realized paper P&L |
| **Research** | `research.*` | Hourly Perplexity news (cached) + daily Claude strategy review with clamped parameter updates |
| **Benchmark** | `core.engine` | Live alpha vs buy-and-hold SPY — the "retail baseline" |
| **Fusion** | `agents.fusion` | Weighted vote: financial models + SLM + LLM + news sentiment + learned strategy |
| **Models** | `models.providers` | Heuristic (financial), local SLM (Ollama), cloud LLM, Perplexity research |
| **Finance** | `finance.*` | SMA/EMA/RSI/MACD indicators, position sizing, stop-loss/take-profit |
| **Brokers** | `brokers.*` | `PaperBroker` (default) + guarded Trade Republic adapter |
| **Economics** | `economics.cost_accounting` | GPU burn accrual, runway, self-sustaining check, spend guardrail |
| **Infra** | `infra.vast` + `scripts/deploy_vast.sh` | Vast.ai GPU provisioning & pricing |

## The self-sustaining loop

```
 rent GPU (Vast.ai, $/hr) ──► serve SLM + call cloud/Perplexity ──► decisions ──► trades
        ▲                                                                            │
        │                                                                            ▼
   runway guardrail ◄──── net worth vs (GPU + inference + fees) ◄──── realised P&L
```

Every cycle the engine computes **runway** = spare net worth ÷ GPU $/hr. If it
drops below the configured floor (`min_runway_hours`), new entries are halted and
only exits are managed, so the bot can't bleed its account dry paying for its own
compute. It reports itself **self-sustaining** once cumulative P&L covers all
GPU + inference spend.

## Quick start

```bash
# 1. Install (paper mode needs no keys, no GPU, no network)
pip install -e .

# 2. Run the engine for a few cycles (synthetic data fallback works offline)
lmtrade run --cycles 5 --interval 1

# 3. See where you stand
lmtrade status

# 4. Launch the dashboard
lmtrade web            # → http://localhost:8000
```

Optional live data / models:

```bash
pip install -e '.[data]'          # yfinance market data
cp .env.example .env              # add OLLAMA_HOST, ANTHROPIC_API_KEY, PERPLEXITY_API_KEY…
```

## Run on Google Colab

Click the **Open in Colab** badge above (or open
[`notebooks/LMTrade_Colab.ipynb`](notebooks/LMTrade_Colab.ipynb)). Colab gives you
a free GPU to serve the SLM; the notebook installs everything, optionally runs
Ollama, and can persist state to Google Drive.

Because Colab can't reliably expose the web server's port, the dashboard there is
rendered **inline with matplotlib** rather than served over HTTP:

```python
import lmtrade.viz as viz
viz.show()                 # one-shot dashboard figure
viz.live(interval=5)       # auto-refreshing dashboard
viz.trades_df(); viz.activity_df(); viz.positions_df()   # feeds as DataFrames
```

The same panels render anywhere matplotlib works; `lmtrade viz` writes them to a
PNG for headless use. Note Colab is for **testing** — its runtime is ephemeral and
idles out, so an always-on self-funding bot belongs on the Vast.ai path.

## Configuration

Defaults live in [`config/default.yaml`](config/default.yaml); every value can be
overridden by environment variables (see [`.env.example`](.env.example)). Key knobs:

- `LMTRADE_MODE` — `paper` (default) or `live`
- `LMTRADE_BUDGET` — starting cash (default 10)
- `LMTRADE_MODEL_STACK` — e.g. `heuristic,slm,perplexity`
- `LMTRADE_GPU_USD_PER_HOUR` / `LMTRADE_MIN_RUNWAY_HOURS` — the economics floor

## Deploying to a Vast.ai GPU

```bash
lmtrade deploy                 # prints a plan + cheapest GPU offers (needs VAST_API_KEY)
GPU=RTX_3090 bash scripts/deploy_vast.sh
```

The onstart script installs Ollama, pulls the SLM, installs LMTrade, and launches
both the engine and the dashboard. Copy your `.env` to the box out-of-band —
**never bake secrets into the image.**

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The suite runs the whole stack end-to-end in paper mode with **no network and no
API keys**.

## Walk-forward backtesting

Pre-train the strategies on history before any live paper trading:

```bash
lmtrade backtest --bars 500 --train 150 --test 50
```

Rolling **[train | test]** folds step through the data: every genome trades the
train window (simulated with the same Black-Scholes option rules as the live
engine, P&L feeding the evolutionary optimizer), then the fittest genome is
scored on the *unseen* test window — the out-of-sample column is the honest
number. The trained population persists, so `lmtrade run` starts pre-trained.
Uses real daily bars via yfinance when available, synthetic data offline.

## The learning loop

The bot maintains a population of strategy **genomes** (momentum, mean-reversion,
breakout — each with mutable parameters). Every cycle the epsilon-greedy
optimizer picks a genome whose signal is fused with the model votes; every
closed trade's realized P&L is attributed back to its genome; every N closed
trades the worst performer is replaced by a **mutated copy of the best**. With
persistent state this runs for months of paper training, and `lmtrade status` /
the dashboard show the live leaderboard. A **daily Claude review** additionally
adjusts risk parameters inside hard safety clamps, and **hourly Perplexity news**
feeds sentiment into every decision.

Outperformance is *measured*, not promised: the benchmark tracker holds SPY from
the same starting budget (the retail baseline) and the dashboard shows live
**alpha** against it.

## GPU sizing (T4 / A100)

| GPU | SLM (`LMTRADE_SLM_MODEL`) | Typical Vast.ai rate | `LMTRADE_GPU_USD_PER_HOUR` |
|-----|---------------------------|----------------------|-----------------------------|
| Tesla T4 (16 GB) | `qwen2.5:1.5b` | ~$0.10–0.25/hr | `0.20` |
| A100 (40/80 GB) | `qwen2.5:7b` (or `14b`) | ~$0.60–1.10/hr | `0.80` |

The runway guardrail scales with the rate you configure — a bigger GPU demands
proportionally more P&L before the bot counts as self-sustaining.

## Honest limitations

- **True HFT is impossible on Trade Republic.** No official API exists; the
  unofficial mobile API has seconds-to-minutes latency and no options chains.
  LMTrade is a *high-cadence intraday* bot (seconds-scale cycles), and its
  options layer is synthetic Black-Scholes pricing for paper trading — read
  [`docs/TRADE_REPUBLIC.md`](docs/TRADE_REPUBLIC.md) before even thinking about
  live mode.
- **€10 is tiny.** A flat ~€1 equity order fee is a 10% round-trip drag; the
  paper broker models it so P&L is honest. Covering a GPU bill on top is
  genuinely hard — the economics layer is built to be honest about that, not to
  pretend otherwise.
- **No performance guarantees.** "Outperform retail" and "self-sustaining in
  3–6 months" are goals the benchmark and economics layers *measure*; nothing
  here promises returns. This is a framework and research tool, not financial
  advice. Trading risks real loss. Use paper mode until you understand exactly
  what it does.

## License

MIT
