from pathlib import Path

from conftest import cli_json

from voiceover_pipeline.cli import build_parser
from voiceover_pipeline.config import OMNIVOICE_LOCAL_MODEL_ID
from voiceover_pipeline.voiceover_script import validate_voiceover_file


def _write_script(tmp_path: Path, body: str, *, provider: str, model: str | None = None) -> Path:
    script = tmp_path / "script.md"
    frontmatter = ["---", "format: voiceover", f"provider: {provider}"]
    if model:
        frontmatter.append(f"model: {model}")
    frontmatter.extend(["---", "", body])
    script.write_text("\n".join(frontmatter), encoding="utf-8")
    return script


def test_omnivoice_validation_uses_profile_and_reports_spoken_text_metrics(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        "Привет АлисаAlice42 мир.",
        provider="omnivoice-local",
        model=OMNIVOICE_LOCAL_MODEL_ID,
    )

    report = validate_voiceover_file(script)

    assert report["max_chunk_chars"] == 420
    assert report["spoken_text"] == {
        "raw_digit_policy": "reject",
        "latin_characters": 5,
        "latin_words": 0,
        "mixed_script_words": 1,
        "total_letters": 19,
        "latin_ratio": 5 / 19,
    }
    assert any(item["code"] == "RAW_DIGITS" for item in report["errors"])
    assert report["chunk_reports"][0]["spoken_text"] == report["spoken_text"]


def test_cloud_validation_keeps_configured_limit_and_digits_are_informational(
    tmp_path: Path,
) -> None:
    script = _write_script(tmp_path, "Version 3 is ready.", provider="polza-tts")

    report = validate_voiceover_file(script, max_chunk_chars=37)

    assert report["max_chunk_chars"] == 37
    assert report["spoken_text"]["raw_digit_policy"] == "warn"
    assert report["spoken_text"]["latin_characters"] == 14
    assert report["spoken_text"]["latin_words"] == 3
    assert report["spoken_text"]["mixed_script_words"] == 0
    assert report["spoken_text"]["total_letters"] == 14
    assert report["spoken_text"]["latin_ratio"] == 1.0
    assert report["valid"]
    assert any(item["code"] == "CONTAINS_DIGITS" for item in report["warnings"])


def test_cli_distinguishes_omitted_limit_from_explicit_2000(tmp_path: Path) -> None:
    parser = build_parser()
    for command in ("validate", "generate"):
        required = ["--script", str(tmp_path / "unused.md")] if command == "validate" else []
        omitted = parser.parse_args([command, *required])
        explicit = parser.parse_args([command, *required, "--max-chunk-chars", "2000"])
        assert omitted.max_chunk_chars is None
        assert explicit.max_chunk_chars == 2000

    script = _write_script(
        tmp_path,
        "а" * 421,
        provider="omnivoice-local",
        model=OMNIVOICE_LOCAL_MODEL_ID,
    )
    code, report = cli_json(
        "validate",
        "--script",
        str(script),
        "--max-chunk-chars",
        "2000",
        "--json",
    )
    assert code == 0
    assert report["max_chunk_chars"] == 2000
    assert report["valid"] is True
