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
    """The runner must (a) give the CLI live web-search access — otherwise
    news/analysis prompts get only static training knowledge, not real
    current events — and (b) invoke the CLI in a way that actually works on
    Windows, where `claude` is a `.cmd` shim and the prompt contains shell
    metacharacters (the sentiment scale literally includes `|`)."""

    def _patch(self, monkeypatch, which="/usr/bin/claude", os_name="posix"):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return FakeCompletedProcess(stdout="some real-time result")

        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda n: which)
        monkeypatch.setattr("lmtrade.models.providers.os.name", os_name)
        monkeypatch.setattr("lmtrade.models.providers.subprocess.run", fake_run)
        return captured

    def test_web_search_preauthorized(self, monkeypatch):
        captured = self._patch(monkeypatch)
        out = _default_claude_cli_runner("what's the latest AAPL news?")
        assert out == "some real-time result"
        cmd = captured["cmd"]
        assert "--allowedTools" in cmd
        assert cmd[cmd.index("--allowedTools") + 1] == "WebSearch"

    def test_prompt_passed_via_stdin_not_argv(self, monkeypatch):
        # The prompt (which contains `|`, quotes, etc.) must go via stdin so
        # no shell can mangle it — passing it as an argv element is exactly
        # what broke on Windows.
        captured = self._patch(monkeypatch)
        prompt = "sentiment scale is bullish|bearish|neutral"
        _default_claude_cli_runner(prompt)
        assert captured["kwargs"]["input"] == prompt
        assert prompt not in " ".join(captured["cmd"])

    def test_uses_resolved_path_from_which(self, monkeypatch):
        captured = self._patch(monkeypatch, which="/opt/tools/claude")
        _default_claude_cli_runner("p")
        assert captured["cmd"][0] == "/opt/tools/claude"

    def test_windows_cmd_shim_wrapped_in_cmd_c(self, monkeypatch):
        captured = self._patch(
            monkeypatch, which=r"C:\Users\x\AppData\npm\claude.cmd", os_name="nt")
        _default_claude_cli_runner("p")
        assert captured["cmd"][:2] == ["cmd", "/c"]
        assert captured["cmd"][2].lower().endswith("claude.cmd")

    def test_posix_not_wrapped(self, monkeypatch):
        captured = self._patch(monkeypatch, which="/usr/bin/claude", os_name="posix")
        _default_claude_cli_runner("p")
        assert captured["cmd"][0] == "/usr/bin/claude"
        assert "cmd" not in captured["cmd"]

    def test_falls_back_to_bare_name_when_which_returns_none(self, monkeypatch):
        captured = self._patch(monkeypatch, which=None)
        _default_claude_cli_runner("p")
        assert captured["cmd"][0] == CLAUDE_CLI_BIN

    def test_raises_on_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda n: "/usr/bin/claude")
        monkeypatch.setattr(
            "lmtrade.models.providers.subprocess.run",
            lambda cmd, **k: FakeCompletedProcess(returncode=1, stderr="boom"),
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

    def test_research_without_sentiment_marker_is_skipped(self, monkeypatch):
        # CLI/shell noise (e.g. an interrupted Windows batch prompt) has no
        # SENTIMENT marker and must not be stored as neutral news.
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        provider = ClaudeCLIProvider(
            runner=lambda p, t: "Execution errorBatchvorgang abbrechen (J/N)?")
        text, cost = provider.research("AAPL")
        assert text == ""
        assert cost == 0.0

    def test_research_with_sentiment_marker_is_kept(self, monkeypatch):
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        provider = ClaudeCLIProvider(
            runner=lambda p, t: "AAPL steady into earnings. SENTIMENT: neutral")
        text, cost = provider.research("AAPL")
        assert "SENTIMENT" in text

    def test_research_failure_returns_empty_not_error_text(self, monkeypatch):
        # A crashing CLI must NOT hand back error text — NewsService would
        # store it as a news item and _parse_sentiment would call it
        # "neutral", silently poisoning the cache with fake neutral news for
        # the whole freshness window. Empty => the fetch is skipped instead.
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )

        def failing(prompt, timeout):
            raise RuntimeError("subprocess exploded")

        provider = ClaudeCLIProvider(runner=failing)
        text, cost = provider.research("AAPL")
        assert text == ""
        assert cost == 0.0

    def test_registered_in_model_stack(self):
        settings = Settings(model={"stack": ["heuristic", "claude_cli"]})
        providers = build_providers(settings)
        assert "claude_cli" in providers
        assert isinstance(providers["claude_cli"], ClaudeCLIProvider)


class TestClaudeCLITimeout:
    """A single web-search research call routinely takes longer than the old
    60s default (some symbols timed out live), so the timeout must be
    configurable and default generously."""

    def test_default_timeout_is_generous(self):
        from lmtrade.models.providers import CLAUDE_CLI_TIMEOUT

        assert CLAUDE_CLI_TIMEOUT >= 120
        assert ClaudeCLIProvider().timeout == CLAUDE_CLI_TIMEOUT

    def test_timeout_configurable_via_settings(self):
        settings = Settings(research={"claude_cli_timeout_seconds": 240})
        assert ClaudeCLIProvider(settings).timeout == 240

    def test_timeout_passed_to_runner(self, monkeypatch):
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        seen = {}
        settings = Settings(research={"claude_cli_timeout_seconds": 200})

        def runner(prompt, timeout):
            seen["timeout"] = timeout
            return "SENTIMENT: neutral"

        provider = ClaudeCLIProvider(settings, runner=runner)
        provider.research("AAPL")
        assert seen["timeout"] == 200
