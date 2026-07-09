"""Configuration loading.

Precedence (highest wins): environment variables (LMTRADE_* / provider keys) >
config.toml (gitignored, per-fork user overrides) > config/default.yaml >
built-in defaults. Loading is deliberately dependency light so the bot boots
even with a bare install.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

try:
    import tomllib  # stdlib, Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - only hit on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
DEFAULT_TOML_PATH = REPO_ROOT / "config.toml"


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
    # Circuit breaker: halt trading outright if net worth ever implies this
    # many multiples of the starting budget. Catches runaway valuations from
    # any bug (e.g. a corrupted mark-to-market) long before they compound,
    # independent of the root cause. 20x is generous for legitimate options
    # leverage on a small account over any realistic short timeframe.
    sanity_max_multiple: float = 20.0
    # Fraction of realized PROFIT (never losses, never principal) on each
    # winning close that's moved into a reserve excluded from future trading
    # capital. 0.5 = 50/50 reinvest/stash, 0.7 = stash 70%. Net worth and
    # alpha still count the reserve — it's not lost, just protected from
    # being re-risked.
    profit_stash_pct: float = 0.0


class LoopConfig(BaseModel):
    interval_seconds: int = 60
    max_positions: int = 2
    # Cap new positions opened in a single cycle, independent of how many
    # slots are free — controls "how many trades per hour" alongside the
    # hourly cycle count. 0 = no extra cap (fill up to remaining slots, the
    # historical behavior).
    max_new_positions_per_cycle: int = 0


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
    min_hold_hours: float = 0.0        # block TP/SL exits before this many
                                       # hours have passed (0 = no minimum).
                                       # The expiry force-close is a hard
                                       # constraint of the option itself and
                                       # always overrides this.
    max_hold_hours: float = 0.0        # force-close after this many hours
                                       # regardless of TP/SL (0 = disabled,
                                       # relies on expiry_days instead)


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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` onto `base`, recursing into nested dicts so e.g. a
    toml [options] section overrides only the keys it specifies, not the
    whole options block."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml_overrides(path: Path) -> dict[str, Any]:
    """Load the optional, gitignored config.toml override file. Never
    raises: a malformed file is logged and ignored rather than crashing the
    bot, matching the project's degrade-gracefully convention."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("lmtrade").warning(
            "Ignoring malformed %s: %s", path, exc)
        return {}


def load_settings(
    config_path: Path | None = None, load_env: bool = True,
    toml_path: Path | None = None,
) -> Settings:
    """Build a Settings object from yaml + config.toml + environment overrides."""
    if load_env:
        _load_dotenv(REPO_ROOT / ".env")

    path = config_path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    toml_overrides = _load_toml_overrides(toml_path or DEFAULT_TOML_PATH)
    raw = _deep_merge(raw, toml_overrides)

    raw = _apply_env_overrides(raw)
    settings = Settings(**raw)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


# Provider secrets pulled straight from the environment (never persisted).
def secret(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None
