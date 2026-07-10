"""Tests for scripts/diagnose_tr.py.

Most of the script's purpose is exercising a REAL Trade Republic account
over a real websocket — untestable offline (this repo's own conventions
require tests to run without network access or API keys; see CLAUDE.md).
The credentials-check path is the one deterministic, network-free path.

The instrument-diagnosis helpers (_find_candidate_fields,
_diagnose_derivative_items) are pure functions over already-fetched data,
so they ARE unit-testable offline — and worth testing directly: they're
what caught the live field-mapping bug (leverage/ask silently defaulting to
0.0 for a wrong field name, so thousands of instruments looked 'usable' but
none were ever tradeable)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_tr.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diagnose_tr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diagnose_tr():
    return _load_script()


def test_script_exists_and_is_executable():
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111, "diagnose_tr.py should be chmod +x"


def test_exits_cleanly_without_credentials(monkeypatch):
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("TR_PHONE", "TR_PIN")}
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], env=env,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 1
    assert "TR_PHONE" in result.stdout


class TestCredentialLoading:
    """Regression: the script called secret('TR_PHONE') directly, but
    .env-file loading is a side effect of load_settings() / _load_dotenv() —
    secret() only reads os.environ. A user with TR_PHONE/TR_PIN ONLY in
    .env (not their shell environment) got 'not set' even though the file
    genuinely had them, because the script never loaded the .env file at
    all."""

    def test_loads_credentials_from_dotenv_file(self, diagnose_tr, tmp_path, monkeypatch):
        import lmtrade.config as config

        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        monkeypatch.delenv("TR_PHONE", raising=False)
        monkeypatch.delenv("TR_PIN", raising=False)
        (tmp_path / ".env").write_text("TR_PHONE=+491234567\nTR_PIN=1234\n")

        phone, pin = diagnose_tr._load_credentials()
        assert phone == "+491234567"
        assert pin == "1234"

    def test_shell_env_still_works_without_a_dotenv_file(self, diagnose_tr, tmp_path, monkeypatch):
        import lmtrade.config as config

        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)   # no .env here
        monkeypatch.setenv("TR_PHONE", "+49999")
        monkeypatch.setenv("TR_PIN", "5678")

        phone, pin = diagnose_tr._load_credentials()
        assert phone == "+49999"
        assert pin == "5678"


class TestFindCandidateFields:
    def test_finds_fields_by_name_hint_regardless_of_exact_guess(self, diagnose_tr):
        item = {"isin": "DE1", "strikePrice": 100.0, "leverageFactor": 5.0,
                "askPrice": 2.0, "unrelatedJunk": "x"}
        candidates = diagnose_tr._find_candidate_fields(item)
        assert "strikePrice" in candidates
        assert "leverageFactor" in candidates
        assert "askPrice" in candidates
        assert "unrelatedJunk" not in candidates

    def test_no_hints_returns_empty(self, diagnose_tr):
        assert diagnose_tr._find_candidate_fields({"foo": 1, "bar": 2}) == {}


class TestProductCategories:
    """The authoritative answer to 'does TR offer plain options for this
    symbol' is TR's own derivativeProductCategories field on the ISIN
    search result — not a guessed list of category names to try."""

    def test_extracts_categories_from_isin_result(self, diagnose_tr):
        result = {"isin": "US0378331005",
                  "derivativeProductCategories": ["knockOutProduct", "vanillaWarrant"]}
        assert diagnose_tr._product_categories(result) == ["knockOutProduct", "vanillaWarrant"]

    def test_missing_field_returns_empty(self, diagnose_tr):
        assert diagnose_tr._product_categories({"isin": "US0378331005"}) == []

    def test_non_dict_returns_empty(self, diagnose_tr):
        assert diagnose_tr._product_categories(None) == []


class TestDiagnoseDerivativeItems:
    def test_reports_wrong_field_name_as_unparsed_not_silent_zero(self, diagnose_tr, capsys):
        # The live incident: real fields under different names than
        # 'leverage'/'optionType'. Since tr_derivatives.py requires these
        # fields (no silent 0 default — see _parse_knockout_item), a wrong
        # name correctly shows up as 0 PARSED, not '5 parsed -> 0 in band'
        # (which is what masked the bug in the first place: it looked like
        # a healthy parse with an oddly-empty leverage band).
        items = [{"isin": f"DE{i}", "strike": 100.0,
                  "leverageFactor": 5.0, "optionKind": "long"} for i in range(5)]
        diagnose_tr._diagnose_derivative_items(items, "AAPL", "_parse_knockout_item")
        out = capsys.readouterr().out
        assert "5 raw -> 0 parsed -> 0 in the" in out
        assert "leverageFactor" in out   # candidate field surfaced

    def test_reports_healthy_funnel_when_fields_correct(self, diagnose_tr, capsys):
        items = [{"isin": "DE1", "optionType": "long", "strike": 100.0,
                  "barrier": 100.0, "size": 1.0, "leverage": 5.0}]
        diagnose_tr._diagnose_derivative_items(items, "AAPL", "_parse_knockout_item")
        out = capsys.readouterr().out
        assert "1 raw -> 1 parsed -> 1 in the" in out

    def test_reports_unparseable_items_distinctly(self, diagnose_tr, capsys):
        items = [{"isin": "DE1"}]   # missing everything but isin
        diagnose_tr._diagnose_derivative_items(items, "AAPL", "_parse_knockout_item")
        out = capsys.readouterr().out
        assert "1 raw -> 0 parsed -> 0 in the" in out

    def test_does_not_crash_on_malformed_items(self, diagnose_tr):
        # Must not raise despite thoroughly broken input, including a
        # literal null entry in the results list.
        items = [None, {}, {"isin": "DE1", "strike": "not-a-number"}]
        diagnose_tr._diagnose_derivative_items(items, "AAPL", "_parse_knockout_item")
