"""Tests for the config.toml override layer: a gitignored, user-specific
config file that sits between the committed config/default.yaml and
environment variables in precedence. Lets a fork be configured without
editing tracked files. Written before implementation per strict TDD."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from lmtrade.config import _load_dotenv, load_settings


@pytest.fixture()
def yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "default.yaml"
    p.write_text(
        "mode: paper\n"
        "budget: 100.0\n"
        "universe: [AAPL, MSFT]\n"
        "loop:\n  interval_seconds: 15\n  max_positions: 8\n"
        "options:\n  take_profit_pct: 0.5\n  stop_loss_pct: 0.4\n"
    )
    return p


class TestTomlOverrideLayer:
    def test_toml_overrides_yaml_defaults(self, tmp_path, yaml_path, monkeypatch):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('budget = 250.0\n[loop]\nmax_positions = 4\n')
        monkeypatch.chdir(tmp_path)
        s = load_settings(config_path=yaml_path, toml_path=toml_path, load_env=False)
        assert s.budget == 250.0
        assert s.loop.max_positions == 4
        # untouched keys keep their yaml value
        assert s.loop.interval_seconds == 15

    def test_missing_toml_is_a_noop(self, tmp_path, yaml_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = load_settings(config_path=yaml_path,
                          toml_path=tmp_path / "does_not_exist.toml",
                          load_env=False)
        assert s.budget == 100.0

    def test_env_vars_still_win_over_toml(self, tmp_path, yaml_path, monkeypatch):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('budget = 250.0\n')
        monkeypatch.setenv("LMTRADE_BUDGET", "500")
        s = load_settings(config_path=yaml_path, toml_path=toml_path, load_env=False)
        assert s.budget == 500.0

    def test_toml_can_set_nested_new_style_keys(self, tmp_path, yaml_path, monkeypatch):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[options]\n'
            'take_profit_pct = 0.3\n'
            'min_hold_hours = 2.0\n'
            '[economics]\n'
            'profit_stash_pct = 0.5\n'
        )
        s = load_settings(config_path=yaml_path, toml_path=toml_path, load_env=False)
        assert s.options.take_profit_pct == pytest.approx(0.3)
        assert s.options.min_hold_hours == pytest.approx(2.0)
        assert s.economics.profit_stash_pct == pytest.approx(0.5)
        # unspecified option field keeps its yaml default
        assert s.options.stop_loss_pct == pytest.approx(0.4)

    def test_malformed_toml_falls_back_gracefully(self, tmp_path, yaml_path, monkeypatch, caplog):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("this is not valid toml [[[")
        s = load_settings(config_path=yaml_path, toml_path=toml_path, load_env=False)
        assert s.budget == 100.0  # falls back to yaml, doesn't crash


class TestResearchProviderEnvOverrides:
    """LMTRADE_NEWS_PROVIDER / LMTRADE_ANALYSIS_PROVIDER let a fully-local
    setup switch news/analysis to the local `claude` CLI via .env alone,
    without needing a config.toml."""

    def test_news_provider_env_override(self, tmp_path, yaml_path, monkeypatch):
        monkeypatch.setenv("LMTRADE_NEWS_PROVIDER", "claude_cli")
        s = load_settings(config_path=yaml_path, load_env=False)
        assert s.research.news_provider == "claude_cli"

    def test_analysis_provider_env_override(self, tmp_path, yaml_path, monkeypatch):
        monkeypatch.setenv("LMTRADE_ANALYSIS_PROVIDER", "claude_cli")
        s = load_settings(config_path=yaml_path, load_env=False)
        assert s.research.analysis_provider == "claude_cli"

    def test_defaults_unchanged_without_env(self, tmp_path, yaml_path, monkeypatch):
        s = load_settings(config_path=yaml_path, load_env=False)
        assert s.research.news_provider == "perplexity"
        assert s.research.analysis_provider == "anthropic"


class TestDotenvLoader:
    """.env.example ships several KEY=            # comment lines (e.g.
    TR_PHONE, TR_PIN) so a user filling in the value in place — the most
    natural edit — must not have the trailing comment become part of the
    value."""

    def _load_isolated(self, monkeypatch, env_path: Path, *keys: str) -> None:
        for k in keys:
            monkeypatch.delenv(k, raising=False)
        _load_dotenv(env_path)

    def test_strips_trailing_inline_comment(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text(
            "TR_PHONE=+491511234567            # +49...\n"
            "TR_PIN=1234              # 4-digit app PIN\n"
        )
        self._load_isolated(monkeypatch, env_path, "TR_PHONE", "TR_PIN")
        assert os.environ["TR_PHONE"] == "+491511234567"
        assert os.environ["TR_PIN"] == "1234"

    def test_value_without_comment_is_unaffected(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text("LMTRADE_BUDGET=250\n")
        self._load_isolated(monkeypatch, env_path, "LMTRADE_BUDGET")
        assert os.environ["LMTRADE_BUDGET"] == "250"

    def test_full_line_comment_still_skipped(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text("# LMTRADE_BUDGET=999\nLMTRADE_BUDGET=50\n")
        self._load_isolated(monkeypatch, env_path, "LMTRADE_BUDGET")
        assert os.environ["LMTRADE_BUDGET"] == "50"
