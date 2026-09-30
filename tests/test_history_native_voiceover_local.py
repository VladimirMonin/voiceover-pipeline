"""End-to-end contract tests for the native ``format: voiceover`` local routes.

These cover the two local families the batch-1 slice had left on the legacy
executor for a ``format: voiceover`` script even though the combination already
functioned there:

* every admitted non-dialogue ``qwen-local`` mode (``preset``, ``clone`` and
  ``design``), whose voiceover frontmatter voice the CLI's
  ``_resolve_qwen_mode_identity`` either keeps (``preset``) or replaces with the
  mode marker exactly as the legacy executor does; and
* the ``omnivoice-local`` preset bank route, whose only voiceover-validator
  permitted voice is the default bank marker the admitted catalog must resolve.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a fake
provider class installed behind the real provider factory so the real catalog
admission, provider dispatch, snapshot identity and history seams stay on the
path, and patched FFmpeg/concat seams. No real provider, network call, API key,
``.env``, model, or download is used.
"""

import hashlib
import io
import json
import sqlite3
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline.config import (
    DEFAULT_OMNIVOICE_VOICE,
    OMNIVOICE_LOCAL_MODEL_ID,
    QWEN_MODEL_BASE,
    QWEN_MODEL_CUSTOMVOICE,
    QWEN_MODEL_VOICE_DESIGN,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers import qwen_local
from voiceover_pipeline.services import native_generation, provider_factory

QWEN_VOICEOVER_VOICE = "Aiden"
OMNI_REFERENCE_TEXT = "reference for the marker profile"


def _mono_wav(seed: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes([seed]) * 160)
    return buffer.getvalue()


class FakeQwenVoiceover:
    """Offline stand-in for the local Qwen provider, installed behind the factory.

    Replacing the provider class (not the factory) keeps the real
    ``build_tts_provider`` dispatch and the real voiceover/identity plumbing on the
    path, so the mode/model/voice the factory actually receives stays observable.
    """

    def __init__(self, *, mode: str, voice: str | None, instruct: str, **kwargs: Any) -> None:
        self.mode = mode
        self.voice = voice
        self.instruct = instruct
        self.state = _QWEN_STATE
        self.state["constructed"].append(
            {"mode": mode, "voice": voice, "instruct": instruct, "kwargs": kwargs}
        )

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self.state["calls"].append(
            {"id": chunk_id, "text": text, "mode": self.mode, "voice": self.voice}
        )
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="wav",
            transcript=text,
            client_path="qwen-local",
        )


class FakeOmniVoice:
    """Offline stand-in for the local OmniVoice provider, behind the factory."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "FakeOmniVoice":
        reference = kwargs.get("voice_bank")
        if reference is not None:
            profile, _path = reference
            _OMNI_STATE["bank"] = {
                "id": profile.id,
                "reference_text": profile.reference_text,
                "reference_sha256": profile.reference_sha256,
            }
        return cls(_OMNI_STATE)

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self._state["calls"].append((chunk_id, text))
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="wav",
            transcript=text,
            client_path="omnivoice-local",
        )


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("this path must not build a provider or read an API key")


_QWEN_STATE: dict[str, Any] = {}
_OMNI_STATE: dict[str, Any] = {}


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    # Force the default python runtime so the committed identity is deterministic
    # regardless of the developer's environment.
    monkeypatch.delenv("VOICEOVER_QWEN_TTS_RUNTIME", raising=False)
    # The probe is a real dependency check; stub it so no qwen runtime or cached
    # model is required and no download is attempted.
    monkeypatch.setattr(
        qwen_local,
        "qwen_local_tts_availability",
        lambda *_args, **_kwargs: qwen_local.QwenLocalTTSAvailability(available=True),
    )
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda _f, data, _fmt, path: path.write_bytes(data)
    )
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(
        cli,
        "concat_audio_files",
        lambda _f, paths, output: output.write_bytes(b"".join(p.read_bytes() for p in paths)),
    )
    _QWEN_STATE.clear()
    _QWEN_STATE.update({"constructed": [], "calls": []})
    _OMNI_STATE.clear()
    _OMNI_STATE.update({"calls": []})
    monkeypatch.setattr(provider_factory, "QwenLocalTTSProvider", FakeQwenVoiceover)
    monkeypatch.setattr(provider_factory, "OmniVoiceLocalTTSProvider", FakeOmniVoice)
    return home


def _voiceover_script(
    tmp_path: Path,
    provider: str,
    model: str,
    voice: str,
    parts: list[str],
    name: str = "voiceover.md",
) -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(
            [
                "---",
                "format: voiceover",
                f"provider: {provider}",
                f"model: {model}",
                f"voice: {voice}",
                "---",
                "\n******\n".join(parts),
            ]
        ),
        encoding="utf-8",
    )
    return path


def _generate_argv(tmp_path: Path, script: Path, run_id: str, *, extra=()):
    return [
        "voiceover-pipeline",
        "generate",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        *extra,
        "--json",
    ]


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    code = excinfo.value.code
    assert not isinstance(code, str) and code is not None
    return code, json.loads(capsys.readouterr().out)


def _history_argv(run_uuid: str, verb: str) -> list[str]:
    return ["voiceover-pipeline", "history", verb, run_uuid, "--json"]


def _run_uuid(prefix: str) -> str:
    from voiceover_pipeline.commands.history import list_history

    runs = list_history()["runs"]
    matches = [run for run in runs if run["user_label"] == prefix]
    assert len(matches) == 1, matches
    return matches[0]["run_uuid"]


def _rows(home: Path, sql: str, params=()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def _snapshot_config(home: Path, run_uuid: str) -> dict:
    row = _rows(home, "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run_uuid,))[0]
    return json.loads(row["config_snapshot"])


def _attempts(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT call_type, provider, model, status, cost, cost_currency FROM attempts "
        "WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )


def _write_marker_bank(root: Path) -> Path:
    """Write a one-profile voice bank whose id is the default OmniVoice marker."""
    voices_dir = root / "voices"
    voices_dir.mkdir(parents=True)
    data = _mono_wav(7)
    (voices_dir / "marker.wav").write_bytes(data)
    entry = {
        "id": DEFAULT_OMNIVOICE_VOICE,
        "display_name": DEFAULT_OMNIVOICE_VOICE,
        "description": "synthetic",
        "language": "ru",
        "reference_audio": "voices/marker.wav",
        "reference_text": OMNI_REFERENCE_TEXT,
        "reference_sha256": hashlib.sha256(data).hexdigest(),
        "origin": {"mode": "owner-reference", "instruction": None, "seed": None},
    }
    catalog = root / "catalog.json"
    catalog.write_text(
        json.dumps(
            {"schema_version": 1, "default_voice": DEFAULT_OMNIVOICE_VOICE, "voices": [entry]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return catalog


# ── qwen-local voiceover modes ────────────────────────────────────────────────


def test_native_voiceover_qwen_preset_keeps_the_validated_voice(
    tmp_path, monkeypatch, capsys, local_env
):
    script = _voiceover_script(
        tmp_path, "qwen-local", QWEN_MODEL_CUSTOMVOICE, QWEN_VOICEOVER_VOICE, ["Первый.", "Второй."]
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "vo-qwen-preset", extra=("--mode", "preset")),
    )

    assert code == 0, payload
    # The voiceover frontmatter voice is the run's effective voice for preset: the
    # real factory receives exactly it, once per part.
    assert [(call["id"], call["mode"], call["voice"]) for call in _QWEN_STATE["calls"]] == [
        ("chunk_01", "preset", QWEN_VOICEOVER_VOICE),
        ("chunk_02", "preset", QWEN_VOICEOVER_VOICE),
    ]
    run_root = tmp_path / "out" / "vo-qwen-preset"
    assert (run_root / ".voiceover-native-history.json").exists()
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["script_format"] == "voiceover"

    run_uuid = _run_uuid("vo-qwen-preset")
    assert _snapshot_config(local_env, run_uuid)["script_format"] == "voiceover"
    config = _snapshot_config(local_env, run_uuid)
    assert config["voice"] == QWEN_VOICEOVER_VOICE
    assert config["qwen_mode"] == {
        "mode": "preset",
        "model": QWEN_MODEL_CUSTOMVOICE,
        "voice": QWEN_VOICEOVER_VOICE,
        "instruct": "Use a calm, warm, clear narration style. Speak naturally and steadily.",
        "runtime": "python",
        "language": "Russian",
    }
    parts = _rows(
        local_env,
        "SELECT voice, prepared_text FROM parts WHERE run_uuid = ? ORDER BY position",
        (run_uuid,),
    )
    assert [tuple(row) for row in parts] == [
        (QWEN_VOICEOVER_VOICE, "Первый."),
        (QWEN_VOICEOVER_VOICE, "Второй."),
    ]
    attempts = _attempts(local_env, run_uuid)
    assert [(row["call_type"], row["provider"], row["status"]) for row in attempts] == [
        ("local_tts_chunk", "qwen-local", "local_completed"),
        ("local_tts_chunk", "qwen-local", "local_completed"),
    ]
    # Exact NULL cost, never a fabricated zero.
    assert all(row["cost"] is None and row["cost_currency"] is None for row in attempts)
    # The private run-level source is the exact prepared chunk text.
    script_source = _rows(
        local_env,
        "SELECT content FROM text_sources WHERE run_uuid = ? AND kind = 'tts_script' "
        "AND part_uuid IS NULL",
        (run_uuid,),
    )
    assert [row["content"] for row in script_source] == ["Первый.\nВторой."]
    # The committed raw bytes are linked before conversion, with no paid receipt.
    assert (run_root / "raw" / "chunk_01.wav").read_bytes() == b"chunk_01-audio"
    assert not (run_root / "raw" / "chunk_01.wav.receipt.json").exists()
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "voiceover"
    assert chunks["voice"] == QWEN_VOICEOVER_VOICE
    assert chunks["model"] == QWEN_MODEL_CUSTOMVOICE


def test_native_voiceover_qwen_clone_preserves_the_legacy_mode_marker_voice(
    tmp_path, monkeypatch, capsys, local_env
):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(_mono_wav(3))
    script = _voiceover_script(
        tmp_path, "qwen-local", QWEN_MODEL_BASE, QWEN_VOICEOVER_VOICE, ["Первый."]
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "vo-qwen-clone",
            extra=("--mode", "clone", "--sample", str(sample), "--sample-text", "ref"),
        ),
    )

    assert code == 0, payload
    # Exactly as the legacy executor does, clone replaces the validated voice with
    # its mode marker; the frontmatter voice never reaches the provider.
    assert [(call["mode"], call["voice"]) for call in _QWEN_STATE["calls"]] == [("clone", "clone")]
    run_uuid = _run_uuid("vo-qwen-clone")
    config = _snapshot_config(local_env, run_uuid)
    assert config["voice"] == "clone"
    assert config["qwen_clone"] == {
        "mode": "clone",
        "model": QWEN_MODEL_BASE,
        "sample_path": str(sample.resolve()),
        "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "sample_size": sample.stat().st_size,
        "sample_text": "ref",
        "runtime": "python",
        "language": "Russian",
    }
    assert config.get("qwen_mode") is None


def test_native_voiceover_qwen_design_preserves_the_legacy_mode_marker_voice(
    tmp_path, monkeypatch, capsys, local_env
):
    script = _voiceover_script(
        tmp_path, "qwen-local", QWEN_MODEL_VOICE_DESIGN, QWEN_VOICEOVER_VOICE, ["Первый."]
    )
    instruction = "Тихий, доверительный шёпот."

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "vo-qwen-design",
            extra=("--mode", "design", "--qwen-instruct", instruction),
        ),
    )

    assert code == 0, payload
    # Design receives no voice at all; the mode marker is the run's effective voice.
    assert [(call["mode"], call["voice"]) for call in _QWEN_STATE["calls"]] == [("design", None)]
    run_uuid = _run_uuid("vo-qwen-design")
    config = _snapshot_config(local_env, run_uuid)
    assert config["voice"] == "design"
    assert config["qwen_mode"] == {
        "mode": "design",
        "model": QWEN_MODEL_VOICE_DESIGN,
        "voice": "design",
        "instruct": instruction,
        "runtime": "python",
        "language": "Russian",
    }
    assert config.get("qwen_clone") is None


def test_native_voiceover_qwen_history_resume_and_sync_without_the_script(
    tmp_path, monkeypatch, capsys, local_env
):
    script = _voiceover_script(
        tmp_path, "qwen-local", QWEN_MODEL_CUSTOMVOICE, QWEN_VOICEOVER_VOICE, ["Первый."]
    )
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "vo-qwen-resume", extra=("--mode", "preset")),
    )
    assert code == 0
    assert len(_QWEN_STATE["calls"]) == 1
    run_uuid = _run_uuid("vo-qwen-resume")
    script.unlink()

    # A reconstructed run must not build a provider or read the deleted script.
    monkeypatch.setattr(provider_factory, "QwenLocalTTSProvider", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["mode"] == "sync"
    assert (tmp_path / "out" / "vo-qwen-resume" / "chunks" / "chunk_01.mp3").is_file()

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert len(_QWEN_STATE["calls"]) == 1


def test_native_voiceover_qwen_changed_voice_fails_resume_before_the_model(
    tmp_path, monkeypatch, capsys, local_env
):
    script = _voiceover_script(
        tmp_path, "qwen-local", QWEN_MODEL_CUSTOMVOICE, QWEN_VOICEOVER_VOICE, ["Первый."]
    )
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "vo-qwen-changed", extra=("--mode", "preset")),
    )
    assert code == 0
    assert len(_QWEN_STATE["calls"]) == 1

    # A changed voiceover voice is a different synthesis identity; it is refused
    # before any local model runs.
    script.write_text(
        script.read_text(encoding="utf-8").replace("voice: Aiden", "voice: Serena"),
        encoding="utf-8",
    )
    monkeypatch.setattr(provider_factory, "QwenLocalTTSProvider", _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "vo-qwen-changed", extra=("--mode", "preset", "--resume")),
    )

    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert len(_QWEN_STATE["calls"]) == 1


def test_native_voiceover_qwen_legacy_root_stays_legacy(tmp_path, monkeypatch, local_env):
    run_root = tmp_path / "out" / "vo-qwen-legacy"
    (run_root / "chunks").mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "completed", "run_id": "vo-qwen-legacy"}), encoding="utf-8"
    )
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "legacy"
    assert not (run_root / ".voiceover-native-history.json").exists()


# ── omnivoice-local marker bank voiceover ─────────────────────────────────────


def test_native_voiceover_omnivoice_marker_bank_commits_its_bank_identity(
    tmp_path, monkeypatch, capsys, local_env
):
    catalog = _write_marker_bank(tmp_path / "bank")
    script = _voiceover_script(
        tmp_path,
        "omnivoice-local",
        OMNIVOICE_LOCAL_MODEL_ID,
        DEFAULT_OMNIVOICE_VOICE,
        ["Первый фрагмент.", "Второй фрагмент."],
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "vo-omni-marker",
            extra=("--mode", "preset", "--voice-bank", str(catalog)),
        ),
    )

    assert code == 0, payload
    # The whole script merges into one OmniVoice session call cloning the profile.
    assert _OMNI_STATE["calls"] == [
        ("chunk_01_omnivoice_session", "Первый фрагмент. Второй фрагмент.")
    ]
    assert _OMNI_STATE["bank"]["id"] == DEFAULT_OMNIVOICE_VOICE
    assert _OMNI_STATE["bank"]["reference_text"] == OMNI_REFERENCE_TEXT

    run_uuid = _run_uuid("vo-omni-marker")
    config = _snapshot_config(local_env, run_uuid)
    assert config["script_format"] == "voiceover"
    assert config["voice"] == DEFAULT_OMNIVOICE_VOICE
    digest = hashlib.sha256(_mono_wav(7)).hexdigest()
    assert config["voice_bank"] == {
        "catalog_path": str(catalog.resolve()),
        "mode": "preset",
        "profiles": [
            {
                "profile_id": DEFAULT_OMNIVOICE_VOICE,
                "reference_audio": "voices/marker.wav",
                "reference_sha256": digest,
                "reference_text": OMNI_REFERENCE_TEXT,
                "language": "ru",
            }
        ],
    }
    attempts = _attempts(local_env, run_uuid)
    assert [(row["call_type"], row["status"]) for row in attempts] == [
        ("local_tts_chunk", "local_completed")
    ]
    assert attempts[0]["cost"] is None and attempts[0]["cost_currency"] is None
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "voiceover"
    assert chunks["voice"] == DEFAULT_OMNIVOICE_VOICE


def test_native_voiceover_omnivoice_ordinary_bank_profile_stays_a_usage_error(
    tmp_path, monkeypatch, capsys, local_env
):
    # An ordinary bank whose profile id is not the permitted marker cannot satisfy
    # the voiceover validator's single voice, so the combination stays a usage error.
    bank_root = tmp_path / "bank"
    voices_dir = bank_root / "voices"
    voices_dir.mkdir(parents=True)
    data = _mono_wav(9)
    (voices_dir / "voice.wav").write_bytes(data)
    catalog = bank_root / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_voice": "voice_a",
                "voices": [
                    {
                        "id": "voice_a",
                        "display_name": "voice_a",
                        "description": "synthetic",
                        "language": "ru",
                        "reference_audio": "voices/voice.wav",
                        "reference_text": "reference",
                        "reference_sha256": hashlib.sha256(data).hexdigest(),
                        "origin": {"mode": "owner-reference", "instruction": None, "seed": None},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script = _voiceover_script(
        tmp_path,
        "omnivoice-local",
        OMNIVOICE_LOCAL_MODEL_ID,
        DEFAULT_OMNIVOICE_VOICE,
        ["Первый фрагмент."],
    )
    monkeypatch.setattr(provider_factory, "OmniVoiceLocalTTSProvider", _explode)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "vo-omni-ordinary",
            extra=("--mode", "preset", "--voice-bank", str(catalog)),
        ),
    )

    assert code == 2, payload
    assert "not found in the voice bank" in payload["error"]
    assert not (tmp_path / "out" / "vo-omni-ordinary").exists()
