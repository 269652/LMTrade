"""Tests for the dashboard settings editor: reads current tunables and syncs
whitelisted changes to config.toml. Written before implementation per TDD."""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from lmtrade.config import Settings
from lmtrade.web import settings_editor


@pytest.fixture()
def settings() -> Settings:
    return Settings(mode="paper", budget=100.0, universe=["AAPL", "MSFT"])


class TestSchema:
    def test_includes_profit_stash_with_current_value(self, settings):
        settings.economics.profit_stash_pct = 0.5
        fields = {f["key"]: f for f in settings_editor.schema(settings)}
        assert "economics.profit_stash_pct" in fields
        assert fields["economics.profit_stash_pct"]["value"] == 0.5

    def test_csv_fields_serialized_as_string(self, settings):
        fields = {f["key"]: f for f in settings_editor.schema(settings)}
        assert fields["universe"]["value"] == "AAPL, MSFT"
        assert fields["universe"]["kind"] == "csv"

    def test_select_field_has_options(self, settings):
        fields = {f["key"]: f for f in settings_editor.schema(settings)}
        assert fields["research.news_provider"]["options"] == ["perplexity", "claude_cli"]


class TestApplyChanges:
    def test_writes_profit_stash_to_config_toml(self, tmp_path):
        p = tmp_path / "config.toml"
        res = settings_editor.apply_changes(p, {"economics.profit_stash_pct": "0.7"})
        assert res["applied"]["economics.profit_stash_pct"] == 0.7
        loaded = tomllib.loads(p.read_text())
        assert loaded["economics"]["profit_stash_pct"] == 0.7

    def test_reloads_into_settings(self, tmp_path):
        p = tmp_path / "config.toml"
        settings_editor.apply_changes(p, {
            "economics.profit_stash_pct": "0.5",
            "loop.max_positions": "12",
            "universe": "AAPL, NVDA, SPY",
            "tr.use_derivatives": "false",
        })
        s = Settings(**tomllib.loads(p.read_text()))
        assert s.economics.profit_stash_pct == 0.5
        assert s.loop.max_positions == 12
        assert s.universe == ["AAPL", "NVDA", "SPY"]
        assert s.tr.use_derivatives is False

    def test_merges_without_clobbering_existing(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text('budget = 250.0\n[loop]\nmax_positions = 8\n')
        settings_editor.apply_changes(p, {"economics.profit_stash_pct": "0.3"})
        loaded = tomllib.loads(p.read_text())
        assert loaded["budget"] == 250.0
        assert loaded["loop"]["max_positions"] == 8            # preserved
        assert loaded["economics"]["profit_stash_pct"] == 0.3  # added

    def test_rejects_unknown_and_bad_values(self, tmp_path):
        p = tmp_path / "config.toml"
        res = settings_editor.apply_changes(p, {
            "not.a.real.key": "x",
            "research.news_provider": "not-an-option",
            "economics.profit_stash_pct": "0.4",
        })
        assert "not.a.real.key" in res["rejected"]
        assert "research.news_provider" in res["rejected"]
        assert res["applied"]["economics.profit_stash_pct"] == 0.4

    def test_int_and_bool_coercion(self, tmp_path):
        p = tmp_path / "config.toml"
        settings_editor.apply_changes(p, {
            "loop.interval_seconds": "30",
            "tr.use_derivatives": "true",
        })
        loaded = tomllib.loads(p.read_text())
        assert loaded["loop"]["interval_seconds"] == 30
        assert loaded["tr"]["use_derivatives"] is True
