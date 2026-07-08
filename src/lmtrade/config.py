"""Configuration loading.

Precedence (highest wins): environment variables (LMTRADE_* / provider keys) >
config/default.yaml > built-in defaults.  Loading is deliberately dependency
light so the bot boots even with a bare install.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard python-dotenv dependency)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class ModelConfig(BaseModel):
    stack: list[str] = ["heuristic"]
    slm_model: str = "qwen2.5:1.5b"
    cloud_model: str = "claude-haiku-4-5-20251001"
    perplexity_model: str = "sonar"
    weights: dict[str, float] = Field(default_factory=lambda: {"heuristic": 1.0})


class RiskConfig(BaseModel):
    max_position_fraction: float = 0.5
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.08
    min_confidence: float = 0.55


class EconomicsConfig(BaseModel):
    gpu_usd_per_hour: float = 0.20
    min_runway_hours: float = 6.0
    inference_cost: dict[str, float] = Field(
        default_factory=lambda: {"slm": 0.0002, "cloud": 0.004, "perplexity": 0.005}
    )


class LoopConfig(BaseModel):
    interval_seconds: int = 60
    max_positions: int = 2


class ResearchConfig(BaseModel):
    news_interval_minutes: int = 60          # hourly Perplexity news
    daily_analysis_interval_hours: int = 24  # daily Claude strategy review


class OptionsConfig(BaseModel):
    enabled: bool = True
    expiry_days: float = 7.0
    max_option_fraction: float = 0.3   # max fraction of equity in one premium
    take_profit_pct: float = 0.5       # +50% premium -> take profit
    stop_loss_pct: float = 0.4         # -40% premium -> cut
    min_hours_to_expiry: float = 24.0  # force-close inside this window


class LearningConfig(BaseModel):
    enabled: bool = True
    population: int = 8
    epsilon: float = 0.2
    mutation_scale: float = 0.3
    evolve_every_trades: int = 10      # run evolution after N closed trades


class BenchmarkConfig(BaseModel):
    symbol: str = "SPY"                # "retail baseline": buy-and-hold SPY


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class DataConfig(BaseModel):
    provider: str = "auto"
    intraday: bool = True     # 1m bars (Yahoo, via httpx) / 10s synthetic ticks


class Settings(BaseModel):
    mode: str = "paper"
    budget: float = 100.0
    currency: str = "EUR"
    universe: list[str] = ["AAPL", "MSFT", "SPY"]
    loop: LoopConfig = LoopConfig()
    model: ModelConfig = ModelConfig()
    risk: RiskConfig = RiskConfig()
    economics: EconomicsConfig = EconomicsConfig()
    research: ResearchConfig = ResearchConfig()
    options: OptionsConfig = OptionsConfig()
    learning: LearningConfig = LearningConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    web: WebConfig = WebConfig()
    data: DataConfig = DataConfig()

    # runtime data directory (state db, logs)
    data_dir: Path = REPO_ROOT / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lmtrade.db"


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    def env(name: str) -> str | None:
        v = os.environ.get(name)
        return v if v not in (None, "") else None

    if v := env("LMTRADE_MODE"):
        cfg["mode"] = v
    if v := env("LMTRADE_BUDGET"):
        cfg["budget"] = float(v)
    if v := env("LMTRADE_UNIVERSE"):
        cfg["universe"] = [s.strip() for s in v.split(",") if s.strip()]
    if v := env("LMTRADE_MODEL_STACK"):
        cfg.setdefault("model", {})["stack"] = [s.strip() for s in v.split(",") if s.strip()]
    if v := env("LMTRADE_SLM_MODEL"):
        cfg.setdefault("model", {})["slm_model"] = v
    if v := env("LMTRADE_CLOUD_MODEL"):
        cfg.setdefault("model", {})["cloud_model"] = v
    if v := env("LMTRADE_PERPLEXITY_MODEL"):
        cfg.setdefault("model", {})["perplexity_model"] = v
    if v := env("LMTRADE_GPU_USD_PER_HOUR"):
        cfg.setdefault("economics", {})["gpu_usd_per_hour"] = float(v)
    if v := env("LMTRADE_MIN_RUNWAY_HOURS"):
        cfg.setdefault("economics", {})["min_runway_hours"] = float(v)
    if v := env("LMTRADE_WEB_HOST"):
        cfg.setdefault("web", {})["host"] = v
    if v := env("LMTRADE_WEB_PORT"):
        cfg.setdefault("web", {})["port"] = int(v)
    return cfg


def load_settings(config_path: Path | None = None, load_env: bool = True) -> Settings:
    """Build a Settings object from yaml + environment overrides."""
    if load_env:
        _load_dotenv(REPO_ROOT / ".env")

    path = config_path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    raw = _apply_env_overrides(raw)
    settings = Settings(**raw)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


# Provider secrets pulled straight from the environment (never persisted).
def secret(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None
