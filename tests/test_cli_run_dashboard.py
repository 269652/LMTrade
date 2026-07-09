"""`lmtrade run` should launch the web dashboard alongside the engine by
default (so starting the bot locally gives you a live dashboard without a
separate `lmtrade web` invocation), with an opt-out and host/port overrides.
Offline: the dashboard launcher is always faked, no real server is bound.
Written before implementation per strict TDD."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import lmtrade.cli as cli
from lmtrade.config import Settings

runner = CliRunner()


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    monkeypatch.setattr(cli, "load_settings", lambda: s)
    return s


class TestRunWebDashboard:
    def test_run_launches_dashboard_by_default(self, settings, monkeypatch):
        calls = []
        monkeypatch.setattr(cli, "_launch_dashboard", lambda s, h, p: calls.append((h, p)))
        result = runner.invoke(cli.app, ["run", "--cycles", "1"])
        assert result.exit_code == 0, result.output
        assert calls == [(settings.web.host, settings.web.port)]

    def test_run_no_web_skips_dashboard(self, settings, monkeypatch):
        calls = []
        monkeypatch.setattr(cli, "_launch_dashboard", lambda s, h, p: calls.append((h, p)))
        result = runner.invoke(cli.app, ["run", "--cycles", "1", "--no-web"])
        assert result.exit_code == 0, result.output
        assert calls == []

    def test_run_web_host_port_override(self, settings, monkeypatch):
        calls = []
        monkeypatch.setattr(cli, "_launch_dashboard", lambda s, h, p: calls.append((h, p)))
        result = runner.invoke(
            cli.app, ["run", "--cycles", "1", "--web-host", "127.0.0.1", "--web-port", "9001"]
        )
        assert result.exit_code == 0, result.output
        assert calls == [("127.0.0.1", 9001)]
