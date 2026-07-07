# ⚡ LMTrade

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
| **CLI** | `lmtrade.cli` | `run`, `web`, `status`, `deploy`, `reset`, `config` |
| **Web dashboard** | `lmtrade.web` | Portfolio, economics, trades, activity feed, logs, equity curve |
| **Engine** | `core.engine` | Evaluation loop: data → economics gate → risk → fusion → execution |
| **Fusion** | `agents.fusion` | Weighted vote across all model providers → one `Decision` |
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

## Honest limitations

- **Trade Republic has no official API.** Live trading uses an unofficial client
  and is against TR's ToS — read [`docs/TRADE_REPUBLIC.md`](docs/TRADE_REPUBLIC.md).
- **€10 is tiny.** A flat ~€1 order fee is a 10% round-trip drag; the paper broker
  models it so P&L is honest. Covering a GPU bill on top is genuinely hard — the
  economics layer is built to be honest about that, not to pretend otherwise.
- This is a **framework and research tool**, not financial advice. Trading risks
  real loss. Use paper mode until you understand exactly what it does.

## License

MIT
