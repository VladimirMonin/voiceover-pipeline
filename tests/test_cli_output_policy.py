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


def test_generate_markdown_skip_existing_all_providers_no_frontmatter(tmp_path):
    providers = [
        ("polza-chat-audio", None),
        ("polza-tts", "openai/gpt-4o-mini-tts"),
        ("openrouter-tts", "google/gemini-3.1-flash-tts-preview"),
        ("openrouter-tts", "openai/gpt-4o-mini-tts-2025-12-15"),
        ("qwen-local", None),
    ]

    for index, (provider, model) in enumerate(providers, start=1):
        run_id = f"markdown-skip-{index}"
        run_dir = tmp_path / "out" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

        args = [
            "generate",
            "--provider", provider,
            "--script", str(fixture_path("smoke_test.md")),
            "--output-dir", str(tmp_path / "out"),
            "--run-id", run_id,
            "--skip-existing",
            "--json",
        ]
        if model:
            args[3:3] = ["--model", model]

        code, data = cli_json(*args)
        assert code == 0, f"provider={provider} model={model} data={data}"
        assert data["status"] == "skipped"


def test_generate_plain_markdown_openrouter_gemini_does_not_require_dialogue_frontmatter(tmp_path):
    run_id = "plain-gemini-skip"
    run_dir = tmp_path / "out" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    code, data = cli_json(
        "generate",
        "--provider", "openrouter-tts",
        "--model", "google/gemini-3.1-flash-tts-preview",
        "--script", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", run_id,
        "--skip-existing",
        "--json",
    )

    assert code == 0
    assert data["status"] == "skipped"


def write_voiceover_meta_script(tmp_path, name, provider, model, voice):
    script = tmp_path / f"{name}.md"
    script.write_text(
        "\n".join([
            "---",
            "format: voiceover",
            f"provider: {provider}",
            f"model: {model}",
            f"voice: {voice}",
            "---",
            "Первый чанк обычной озвучки.",
            "******",
            "Второй чанк обычной озвучки.",
        ]),
        encoding="utf-8",
    )
    return script


def test_generate_voiceover_metadata_skip_existing_all_single_speaker_providers(tmp_path):
    cases = [
        ("polza-chat-audio", "openai/gpt-audio-mini", "ash"),
        ("polza-tts", "openai/gpt-4o-mini-tts", "ash"),
        ("polza-tts", "elevenlabs/text-to-speech-turbo-2-5", "Rachel"),
        ("openrouter-tts", "google/gemini-3.1-flash-tts-preview", "Puck"),
        ("openrouter-tts", "openai/gpt-4o-mini-tts-2025-12-15", "alloy"),
        ("qwen-local", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Aiden"),
    ]

    for index, (provider, model, voice) in enumerate(cases, start=1):
        script = write_voiceover_meta_script(tmp_path, f"voiceover-{index}", provider, model, voice)
        run_id = f"voiceover-meta-skip-{index}"
        run_dir = tmp_path / "out" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

        code, data = cli_json(
            "generate",
            "--script", str(script),
            "--output-dir", str(tmp_path / "out"),
            "--run-id", run_id,
            "--skip-existing",
            "--json",
        )

        assert code == 0, f"provider={provider} model={model} data={data}"
        assert data["status"] == "skipped"
