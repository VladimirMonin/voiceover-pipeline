import json
from pathlib import Path

from conftest import cli_json, fixture_path


def test_generate_skip_existing_no_network(tmp_path):
    run_dir = tmp_path / "out" / "existing-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"test": True}))

    code, data = cli_json(
        "generate",
        "--provider", "polza-chat-audio",
        "--script", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "existing-run",
        "--skip-existing",
        "--json",
    )
    assert code == 0
    assert data["status"] == "skipped"
    assert "run folder exists" in data["reason"]


def test_generate_existing_without_overwrite_fails(tmp_path):
    run_dir = tmp_path / "out" / "existing-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"test": True}))

    code, data = cli_json(
        "generate",
        "--provider", "polza-chat-audio",
        "--script", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "existing-run",
        "--json",
    )
    assert code == 30, f"expected exit 30, got {code}"
    assert data["code"] == 30


def test_timings_skip_existing_no_work(tmp_path):
    out_dir = tmp_path / "out" / "my-timings"
    out_dir.mkdir(parents=True)
    (out_dir / "my-timings.timings.json").write_text("{}")

    code, data = cli_json(
        "timings",
        "--audio", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "my-timings",
        "--skip-existing",
        "--json",
    )
    assert code == 0
    assert data["status"] == "skipped"


def test_timings_existing_without_overwrite_fails(tmp_path):
    out_dir = tmp_path / "out" / "my-timings"
    out_dir.mkdir(parents=True)
    (out_dir / "my-timings.timings.json").write_text("{}")

    code, data = cli_json(
        "timings",
        "--audio", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "my-timings",
        "--json",
    )
    assert code == 30, f"expected exit 30, got {code}"
    assert data["code"] == 30


def test_generate_skip_existing_works_without_api_key(tmp_path, monkeypatch):
    import voiceover_pipeline.config as cfg
    monkeypatch.setenv("POLZA_API_KEY", "bad-key")
    monkeypatch.setattr(cfg, "read_polza_key", lambda: "bad-key")

    run_dir = tmp_path / "out" / "skip-test"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")

    code, data = cli_json(
        "generate",
        "--provider", "polza-chat-audio",
        "--script", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "skip-test",
        "--skip-existing",
        "--json",
    )
    assert code == 0
    assert data["status"] == "skipped"
