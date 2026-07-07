"""LMTrade command-line interface.

    lmtrade run            # start the trading engine (paper by default)
    lmtrade web            # launch the dashboard
    lmtrade status         # print current portfolio + economics
    lmtrade deploy         # print/inspect a Vast.ai deployment plan
    lmtrade reset          # wipe runtime state (fresh account)
    lmtrade config         # show the resolved configuration
"""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .brokers.factory import build_broker
from .config import load_settings
from .core.engine import Engine
from .core.state import Store
from .economics.cost_accounting import CostAccountant

app = typer.Typer(add_completion=False, help="Self-sustaining LLM/SLM trading bot.")
console = Console()


def _banner(settings) -> None:
    mode = settings.mode.upper()
    color = "red" if mode == "LIVE" else "cyan"
    console.print(Panel.fit(
        f"[bold]LMTrade v{__version__}[/bold]\n"
        f"mode=[{color}]{mode}[/{color}]  budget={settings.budget} {settings.currency}  "
        f"universe={','.join(settings.universe)}\n"
        f"model stack: {', '.join(settings.model.stack)}\n"
        f"GPU: ${settings.economics.gpu_usd_per_hour}/hr  "
        f"runway floor: {settings.economics.min_runway_hours}h",
        title="⚡ engine", border_style=color,
    ))


@app.command()
def run(
    cycles: int = typer.Option(0, help="Stop after N cycles (0 = run forever)."),
    interval: int = typer.Option(0, help="Override loop interval seconds (0 = config)."),
):
    """Start the trading engine."""
    settings = load_settings()
    if interval > 0:
        settings.loop.interval_seconds = interval
    _banner(settings)
    if settings.mode == "live":
        console.print("[bold red]⚠ LIVE mode — real Trade Republic execution. "
                      "This uses an unofficial API against TR's ToS.[/bold red]")
    store = Store(settings.db_path)
    broker = build_broker(settings, store)
    engine = Engine(settings, store, broker)
    try:
        engine.run_forever(max_cycles=cycles or None)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — stopping engine.[/yellow]")
    finally:
        store.close()


@app.command()
def web(
    host: str = typer.Option("", help="Bind host (default from config)."),
    port: int = typer.Option(0, help="Bind port (default from config)."),
):
    """Launch the web dashboard."""
    import uvicorn

    settings = load_settings()
    h = host or settings.web.host
    p = port or settings.web.port
    console.print(f"[cyan]Dashboard →[/cyan] http://{h}:{p}")
    uvicorn.run("lmtrade.web.app:app", host=h, port=p, log_level="info")


@app.command()
def status():
    """Print current portfolio and self-sustaining economics."""
    settings = load_settings()
    store = Store(settings.db_path)
    broker = build_broker(settings, store) if settings.mode == "paper" else None
    cash = float(store.get_meta("cash", settings.budget))
    positions = store.positions()
    pos_value = sum(p.qty * p.avg_price for p in positions)
    accountant = CostAccountant(settings, store)
    econ = accountant.snapshot(cash, pos_value)

    t = Table(title="Portfolio", show_header=True, header_style="bold")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Mode", store.get_meta("mode", settings.mode))
    t.add_row("Cash", f"{cash:.2f} {settings.currency}")
    t.add_row("Net worth", f"{econ.net_worth_eur:.2f} {settings.currency}")
    t.add_row("P&L", f"{econ.pnl_eur:+.2f} {settings.currency}")
    t.add_row("Open positions", str(len(positions)))
    t.add_row("GPU accrued", f"${econ.gpu_cost_accrued_usd:.4f}")
    t.add_row("Inference cost", f"${econ.inference_cost_usd:.4f}")
    t.add_row("Runway", f"{econ.runway_hours:.1f} h")
    t.add_row("Self-sustaining", "✅ yes" if econ.self_sustaining else "⏳ not yet")
    console.print(t)

    if positions:
        pt = Table(title="Positions")
        pt.add_column("Symbol"); pt.add_column("Qty", justify="right")
        pt.add_column("Avg", justify="right")
        for p in positions:
            pt.add_row(p.symbol, f"{p.qty:.4f}", f"{p.avg_price:.2f}")
        console.print(pt)
    store.close()


@app.command()
def deploy(
    gpu: str = typer.Option("RTX_3090", help="GPU type to price on Vast.ai."),
):
    """Inspect a Vast.ai deployment plan (cheapest GPU offers + current burn)."""
    from .infra.vast import cheapest_offers, current_hourly_burn

    settings = load_settings()
    console.print(Panel.fit(
        "Deployment plan:\n"
        "1. Provision a Vast.ai GPU instance (see scripts/deploy_vast.sh).\n"
        "2. On the box: install Ollama + pull the SLM, `pip install -e .`.\n"
        "3. Set .env (keys, GPU rate), run `lmtrade run` and `lmtrade web`.\n"
        f"4. Economics floor: halt trading below {settings.economics.min_runway_hours}h runway.",
        title="🚀 deploy", border_style="cyan",
    ))
    offers = cheapest_offers(gpu)
    if offers is None:
        console.print("[yellow]Set VAST_API_KEY to fetch live GPU offers.[/yellow]")
    else:
        t = Table(title=f"Cheapest {gpu} offers")
        t.add_column("ID"); t.add_column("GPU"); t.add_column("USD/hr", justify="right")
        t.add_column("Region")
        for o in offers:
            t.add_row(str(o["id"]), str(o["gpu"]), f"{o['usd_per_hour']}", str(o["region"]))
        console.print(t)
    burn = current_hourly_burn()
    if burn is not None:
        console.print(f"Current Vast.ai burn: [bold]${burn}/hr[/bold]")


@app.command()
def viz(
    out: str = typer.Option("data/dashboard.png", help="Output PNG path."),
):
    """Render the dashboard to a PNG (headless). In notebooks use `lmtrade.viz.show()`."""
    import matplotlib
    matplotlib.use("Agg")
    from .viz.panels import dashboard_figure

    settings = load_settings()
    fig = dashboard_figure(settings.db_path)
    out_path = settings.data_dir / "dashboard.png" if out == "data/dashboard.png" else out
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    console.print(f"[green]Dashboard written to[/green] {out_path}")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", help="Skip confirmation.")):
    """Wipe runtime state (database) and start a fresh account."""
    settings = load_settings()
    if not yes:
        typer.confirm(f"Delete {settings.db_path}? This resets all history.", abort=True)
    if settings.db_path.exists():
        settings.db_path.unlink()
    console.print("[green]State reset.[/green]")


@app.command()
def config():
    """Show the resolved configuration as JSON."""
    settings = load_settings()
    data = settings.model_dump()
    data["data_dir"] = str(data["data_dir"])
    console.print_json(json.dumps(data))


@app.command()
def version():
    """Print the version."""
    console.print(f"LMTrade v{__version__}")


if __name__ == "__main__":
    app()
