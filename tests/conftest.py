import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


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
