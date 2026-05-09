import pytest
from conftest import cli_json, fixture_path


INVALID_RUN_IDS = [
    (".", "not allowed"),
    ("..", "not allowed"),
    ("a/b", "path separators"),
    ("a\\b", "path separators"),
    ("CON", "reserved name"),
    ("con", "reserved name"),
    ("COM1", "reserved name"),
    ("LPT9", "reserved name"),
    ("PRN", "reserved name"),
    ("AUX", "reserved name"),
    ("NUL", "reserved name"),
    ("prod.", "trailing dot"),
    (" prod", "leading whitespace"),
    ("prod ", "trailing whitespace"),
    (" ", "whitespace only"),
    ("C:\\test", "absolute path"),
    ("test<bad", "illegal char"),
]


@pytest.mark.parametrize("run_id,reason", INVALID_RUN_IDS)
def test_invalid_run_id(run_id, reason):
    code, data = cli_json("generate", "--run-id", run_id, "--json")
    assert code == 2, f"run_id={run_id!r} ({reason}) expected exit 2, got {code}"
    assert data["code"] == 2


VALID_RUN_IDS = [
    "prod",
    "prod-01",
    "prod_01",
    "prod.v1",
    "v0.3.0",
    "test_run",
    "my-run",
]


@pytest.mark.parametrize("run_id", VALID_RUN_IDS)
def test_valid_run_id_does_not_fail(run_id):
    from voiceover_pipeline.cli import _validate_run_id
    result = _validate_run_id(run_id)
    assert result == run_id


INVALID_OUTPUT_DIRS = [
    ("C:\\", "drive root"),
    ("C:\\Windows", "system dir — skip for now"),
]


def test_output_dir_is_drive_root_fails():
    code, data = cli_json("generate", "--output-dir", "C:\\", "--json")
    assert code == 2, f"expected exit 2, got {code}"
    assert "drive root" in data["error"].lower() or "output-dir" in data["error"].lower()


def test_output_dir_is_cwd_fails():
    from pathlib import Path
    cwd = str(Path.cwd())
    code, data = cli_json("generate", "--output-dir", cwd, "--json")
    assert code == 2, f"expected exit 2, got {code}"
    assert "current working directory" in data["error"].lower() or "output-dir" in data["error"].lower()


def test_valid_output_dir_ok(tmp_path):
    out = tmp_path / "builds"
    code, data = cli_json(
        "generate",
        "--output-dir", str(out),
        "--run-id", "valid-dir-test",
        "--script", str(fixture_path("smoke_test.md")),
        "--skip-existing",
        "--json",
    )
    assert code in (0, 30), f"expected 0 or 30, got {code}: {data}"


class TestStylePromptFlags:
    def test_no_style_prompt_flag(self):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        ns = argparse.Namespace(style_prompt=None, style_prompt_file=None, no_style_prompt=True)
        assert _resolve_style_prompt(ns) is None

    def test_style_prompt_file(self, tmp_path):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        pf = tmp_path / "prompt.txt"
        pf.write_text("File-based narration style", encoding="utf-8")
        ns = argparse.Namespace(style_prompt=None, style_prompt_file=pf, no_style_prompt=False)
        assert _resolve_style_prompt(ns) == "File-based narration style"

    def test_style_prompt_file_missing(self, tmp_path):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        ns = argparse.Namespace(style_prompt=None, style_prompt_file=tmp_path / "missing.txt", no_style_prompt=False)
        with pytest.raises(FileNotFoundError):
            _resolve_style_prompt(ns)

    def test_no_style_prompt_has_priority_over_file(self, tmp_path):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        pf = tmp_path / "prompt.txt"
        pf.write_text("Should not be read", encoding="utf-8")
        ns = argparse.Namespace(style_prompt=None, style_prompt_file=pf, no_style_prompt=True)
        assert _resolve_style_prompt(ns) is None

    def test_no_style_prompt_has_priority_over_cli(self):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        ns = argparse.Namespace(style_prompt="cli style", style_prompt_file=None, no_style_prompt=True)
        assert _resolve_style_prompt(ns) is None

    def test_default_prompt_when_no_flags(self):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt, PODCAST_NARRATION_PROMPT

        ns = argparse.Namespace(style_prompt=None, style_prompt_file=None, no_style_prompt=False)
        assert _resolve_style_prompt(ns) == PODCAST_NARRATION_PROMPT
