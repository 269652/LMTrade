"""Smoke test for scripts/diagnose_tr.py.

The script's entire purpose is exercising a REAL Trade Republic account over
a real websocket — everything past the credentials check is explicitly
untestable offline (this repo's own conventions require tests to run
without network access or API keys; see CLAUDE.md). This only verifies the
one path that IS deterministic and safe to run in CI: no TR_PHONE/TR_PIN set
-> a clear, immediate, non-zero exit with no attempt at network I/O."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_tr.py"


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
