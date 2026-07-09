"""Tests for the local `claude` CLI model provider (models/providers.py).

Lets the bot use a locally-installed Claude Code CLI — billed through the
user's own Claude subscription/login, no ANTHROPIC_API_KEY, everything runs
on the user's machine — for trading-decision analysis and market research,
instead of the Anthropic/Perplexity HTTP APIs.

Runs entirely offline: the CLI subprocess is always injected/faked here, so
no real `claude` process is ever spawned by the test suite. Written before
implementation per strict TDD."""
from __future__ import annotations

import pytest

from lmtrade.config import Settings
from lmtrade.models.providers import (
    CLAUDE_CLI_BIN,
    ClaudeCLIProvider,
    _default_claude_cli_runner,
    build_providers,
)


class FakeCompletedProcess:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class TestDefaultClaudeCLIRunner:
    """The runner must give the CLI live web-search access — otherwise
    "news"/"analysis" prompts only get the model's static training-cutoff
    knowledge, not real current events."""

    def test_invokes_claude_with_web_search_preauthorized(self, monkeypatch):
        captured = {}

        def fake_run(cmd, capture_output, text, timeout):
            captured["cmd"] = cmd
            return FakeCompletedProcess(stdout="some real-time result")

        monkeypatch.setattr("lmtrade.models.providers.subprocess.run", fake_run)
        out = _default_claude_cli_runner("what's the latest AAPL news?")
        assert out == "some real-time result"
        cmd = captured["cmd"]
        assert cmd[0] == CLAUDE_CLI_BIN
        assert "-p" in cmd
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert "WebSearch" in cmd[idx + 1]

    def test_raises_on_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr(
            "lmtrade.models.providers.subprocess.run",
            lambda *a, **k: FakeCompletedProcess(returncode=1, stderr="boom"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            _default_claude_cli_runner("prompt")


class TestClaudeCLIProvider:
    def test_available_when_binary_on_path(self, monkeypatch):
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        assert ClaudeCLIProvider().available() is True

    def test_unavailable_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda name: None)
        assert ClaudeCLIProvider().available() is False

    def test_analyze_parses_json_decision(self, monkeypatch):
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        calls = []

        def fake_runner(prompt, timeout):
            calls.append(prompt)
            return '{"direction":"buy","confidence":0.7,"rationale":"strong trend"}'

        provider = ClaudeCLIProvider(runner=fake_runner)
        signal = provider.analyze("AAPL", {"indicators": {"last": 100}})
        assert signal.direction == "buy"
        assert signal.confidence == pytest.approx(0.7)
        assert signal.cost_usd == 0.0
        assert calls and "AAPL" in calls[0]

    def test_analyze_degrades_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda name: None)
        provider = ClaudeCLIProvider(runner=lambda p, t: "should not be called")
        signal = provider.analyze("AAPL", {})
        assert signal.direction == "hold"

    def test_analyze_degrades_on_runner_error(self, monkeypatch):
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )

        def failing_runner(prompt, timeout):
            raise RuntimeError("claude cli crashed")

        provider = ClaudeCLIProvider(runner=failing_runner)
        signal = provider.analyze("AAPL", {})
        assert signal.direction == "hold"
        assert "crashed" in signal.rationale

    def test_research_returns_text_and_zero_cost(self, monkeypatch):
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        provider = ClaudeCLIProvider(runner=lambda p, t: "AAPL rallies. SENTIMENT: bullish")
        text, cost = provider.research("AAPL")
        assert "bullish" in text.lower()
        assert cost == 0.0

    def test_research_empty_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda name: None)
        provider = ClaudeCLIProvider(runner=lambda p, t: "unused")
        text, cost = provider.research("AAPL")
        assert text == ""
        assert cost == 0.0

    def test_registered_in_model_stack(self):
        settings = Settings(model={"stack": ["heuristic", "claude_cli"]})
        providers = build_providers(settings)
        assert "claude_cli" in providers
        assert isinstance(providers["claude_cli"], ClaudeCLIProvider)
