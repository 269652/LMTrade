"""`lmtrade run` prints research-provider/TR-client diagnostics at startup so
availability problems (claude CLI not on PATH, TR session not resumable) are
visible immediately instead of requiring the user to infer them from silence
in the scrolling cycle log. Written before implementation per strict TDD."""
from __future__ import annotations

import lmtrade.cli as cli
from lmtrade.config import Settings


class FakeTRClient:
    def __init__(self, ok: bool):
        self._ok = ok

    def available(self) -> bool:
        return self._ok


class FakeEngine:
    def __init__(self, tr_derivatives=None):
        self.tr_derivatives = tr_derivatives


class TestClaudeCliDiagnostic:
    def test_missing_binary_is_flagged(self, monkeypatch, capsys):
        settings = Settings(research={"news_provider": "claude_cli"})
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda n: None)
        cli._print_diagnostics(settings, FakeEngine())
        out = capsys.readouterr().out
        assert "NOT FOUND" in out

    def test_found_binary_is_not_flagged_as_missing(self, monkeypatch, capsys):
        settings = Settings(research={"analysis_provider": "claude_cli"})
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda n: "/usr/bin/claude")
        cli._print_diagnostics(settings, FakeEngine())
        out = capsys.readouterr().out
        assert "NOT FOUND" not in out

    def test_not_printed_when_claude_cli_unused(self, monkeypatch, capsys):
        settings = Settings()   # defaults: perplexity / anthropic
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda n: None)
        cli._print_diagnostics(settings, FakeEngine())
        out = capsys.readouterr().out
        assert "claude CLI" not in out


class TestTRDiagnostic:
    def test_connected_client_reported(self, capsys):
        cli._print_diagnostics(Settings(), FakeEngine(tr_derivatives=FakeTRClient(True)))
        out = capsys.readouterr().out.lower()
        assert "connected" in out

    def test_unavailable_client_reported(self, capsys):
        cli._print_diagnostics(Settings(), FakeEngine(tr_derivatives=FakeTRClient(False)))
        out = capsys.readouterr().out.lower()
        assert "unavailable" in out

    def test_not_printed_when_no_tr_client(self, capsys):
        cli._print_diagnostics(Settings(), FakeEngine(tr_derivatives=None))
        out = capsys.readouterr().out
        assert "TR live derivatives" not in out
