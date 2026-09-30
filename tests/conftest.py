import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def isolated_voiceover_home(tmp_path, monkeypatch):
    """Keep every test off the real user history home.

    The local ``transcribe``/``timings``/``verify-tts`` commands now persist their
    observed result into the private ``VOICEOVER_HOME`` history by default, and
    subprocess CLI runs inherit this process environment. Pointing an unset home
    at a per-test temporary directory keeps the suite offline and deterministic; a
    test that sets ``VOICEOVER_HOME`` itself still wins.
    """
    if not os.environ.get("VOICEOVER_HOME"):
        monkeypatch.setenv("VOICEOVER_HOME", str(tmp_path / "voiceover-home"))


def run_cli(*args, cwd=PROJECT_ROOT):
    proc = subprocess.run(
        [sys.executable, "-m", "voiceover_pipeline.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return proc


def cli_json(*args, cwd=PROJECT_ROOT):
    proc = run_cli(*args, cwd=cwd)
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = {
            "status": "parse_error",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return proc.returncode, data


def fixture_path(name: str) -> Path:
    return FIXTURES / name
