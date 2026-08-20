from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Final

from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError
from voiceover_pipeline.local_runtime.transports.audio_cpp_package import (
    NativeAudioCppInstall,
    admit_audio_cpp_native_package,
)

NATIVE_AUDIO_CPP_EXECUTABLE_ENV: Final = "VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE"
_PRIVATE_TMP_PREFIX: Final = "voiceover-audio-cpp-"
_STAGED_AUDIO_RATE_HZ: Final = 16_000
_MODEL_ARTIFACT_GLOB: Final = "*.gguf"


@dataclass(frozen=True)
class _FamilySpec:
    operation: str
    provider_id: str
    model_id: str
    cli_family: str


_FAMILY_SPECS: Final = {
    "qwen3-asr": _FamilySpec(
        operation="asr",
        provider_id="qwen-local",
        model_id="Qwen/Qwen3-ASR-0.6B",
        cli_family="qwen3_asr",
    ),
    "nemotron-3.5-asr": _FamilySpec(
        operation="asr",
        provider_id="nemotron-local",
        model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        cli_family="nemotron_asr",
    ),
    "qwen3-tts": _FamilySpec(
        operation="tts",
        provider_id="qwen-local",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        cli_family="qwen3_tts",
    ),
    "omnivoice": _FamilySpec(
        operation="tts",
        provider_id="omnivoice-local",
        model_id="audio-cpp/omnivoice-q8_0",
        cli_family="omnivoice",
    ),
}

_QWEN_TTS_MODEL_IDS: Final = frozenset(
    {
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }
)
_QWEN_TTS_TASKS: Final = {
    "custom-voice": "tts",
    "voice-clone": "clon",
    "voice-design": "vdes",
}
_QWEN_TTS_MODELS_BY_MODE: Final = {
    "custom-voice": frozenset({"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
    "voice-clone": frozenset(
        {
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        }
    ),
    "voice-design": frozenset({"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"}),
}


def discover_native_audio_cpp_install(
    executable_path: Path, *, required_model_paths: tuple[Path, ...] = ()
) -> NativeAudioCppInstall:
    """Verify the native executable and every manifest-listed colocated dependency.

    Legacy entry point retained for provider compatibility. It delegates to the
    single structured admission so that the receipt, DLL closure and model
    coverage checks never diverge between routes. The manifest is intentionally
    required: a bare ``audiocpp_cli.exe`` does not prove that a Windows package
    has the DLL closure needed by the selected build.
    """

    try:
        return admit_audio_cpp_native_package(
            executable_path, required_model_paths=required_model_paths
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def decode_audio_cpp_cli_request(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode the runtime-neutral VOP envelope without delivery-adapter details."""

    if payload.get("schema_version") != 1:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported schema version")
    family = payload.get("family")
    if not isinstance(family, str) or family not in _FAMILY_SPECS:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported model family")
    spec = _FAMILY_SPECS[family]
    if payload.get("operation") != spec.operation:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported operation")
    if payload.get("provider_id") != spec.provider_id:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported provider")
    request_payload = payload.get("payload")
    if not isinstance(request_payload, Mapping):
        raise RuntimeProtocolError("audio.cpp CLI request payload must be an object")
    request = dict(request_payload)
    if family == "qwen3-tts":
        _validate_qwen_tts_request(request)
    elif request.get("model_id") != spec.model_id:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported model")
    if spec.operation == "asr":
        _validate_asr_request(family, request)
    elif family != "qwen3-tts":
        _validate_tts_request(family, request)
    return {"family": family, **request}


def build_audio_cpp_family_arguments(
    *,
    family: str,
    payload: Mapping[str, object],
    model_argument: str,
    output_directory: Path,
    audio_argument: str | None = None,
    forced_aligner_argument: str | None = None,
    reference_audio_argument: str | None = None,
    wav_filename: str = "speech.wav",
) -> tuple[str, ...]:
    """Map a decoded family request to CLI argv without choosing a launcher.

    Sensitive text (prompt/context, transcript, reference transcript, design
    instruction) is passed to the native process via argv flags because the
    pinned upstream revision provides no file/stdin transport. It must never
    appear in exceptions, logs, receipts, or metadata.
    """

    spec = _FAMILY_SPECS.get(family)
    if spec is None:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported model family")
    command: list[str] = [
        "--task",
        spec.operation,
        "--family",
        spec.cli_family,
        "--model",
        model_argument,
        "--backend",
        "cuda",
    ]
    if spec.operation == "asr":
        command.extend(
            (
                "--audio",
                audio_argument or _required_string(payload, "audio_path", "audio path"),
                "--text-out",
                str(output_directory / "transcript.txt"),
                "--segments-out",
                str(output_directory / "segments.json"),
            )
        )
        language = payload.get("language")
        if isinstance(language, str) and language:
            command.extend(("--language", language))
        context_text = payload.get("context_text")
        if family == "qwen3-asr" and isinstance(context_text, str) and context_text:
            command.extend(("--text", context_text))
        if payload.get("timestamp_mode") == "word":
            command.extend(("--words-out", str(output_directory / "words.json")))
        if family == "qwen3-asr" and payload.get("timestamp_mode") == "word":
            if forced_aligner_argument is None:
                raise RuntimeTransportError(
                    "audio.cpp native forced aligner artifact is unavailable"
                )
            command.extend(
                (
                    "--session-option",
                    f"qwen3_asr.forced_aligner_model_path={forced_aligner_argument}",
                )
            )
        return tuple(command)
    if family == "qwen3-tts" and payload.get("mode") is not None:
        return _build_qwen_tts_arguments(
            payload=payload,
            model_argument=model_argument,
            output_directory=output_directory,
            reference_audio_argument=reference_audio_argument,
            wav_filename=wav_filename,
        )
    command.extend(
        (
            "--text",
            _required_string(payload, "text", "text"),
            "--out",
            str(output_directory / wav_filename),
        )
    )
    _append_optional(command, "--voice", payload.get("voice"))
    _append_optional(command, "--language", payload.get("language"))
    if family == "omnivoice":
        mode = payload.get("omnivoice_mode")
        if mode == "clone":
            if reference_audio_argument is None:
                raise RuntimeTransportError("audio.cpp clone reference audio is unavailable")
            command.extend(
                (
                    "--voice-ref",
                    reference_audio_argument,
                    "--reference-text",
                    _required_string(payload, "reference_text", "reference text"),
                    "--text-chunk-size",
                    str(payload["text_chunk_size"]),
                    "--mode",
                    "offline",
                    "--seed",
                    str(payload["seed"]),
                    "--num-inference-steps",
                    str(payload["num_inference_steps"]),
                    "--guidance-scale",
                    str(payload["guidance_scale"]),
                )
            )
        elif mode == "design":
            command.extend(
                (
                    "--instruct",
                    _required_string(payload, "design_instruction", "design instruction"),
                    "--text-chunk-size",
                    str(payload["text_chunk_size"]),
                    "--mode",
                    "offline",
                    "--seed",
                    str(payload["seed"]),
                    "--num-inference-steps",
                    str(payload["num_inference_steps"]),
                    "--guidance-scale",
                    str(payload["guidance_scale"]),
                )
            )
        else:
            command.extend(
                (
                    "--instruct",
                    _required_string(payload, "style_condition", "OmniVoice style condition"),
                    "--text-chunk-size",
                    str(payload["text_chunk_size"]),
                    "--mode",
                    "offline",
                    "--seed",
                    str(payload["seed"]),
                    "--num-inference-steps",
                    str(payload["num_inference_steps"]),
                    "--guidance-scale",
                    str(payload["guidance_scale"]),
                )
            )
    return tuple(command)


def _build_qwen_tts_arguments(
    *,
    payload: Mapping[str, object],
    model_argument: str,
    output_directory: Path,
    reference_audio_argument: str | None,
    wav_filename: str,
) -> tuple[str, ...]:
    mode = payload.get("mode")
    assert isinstance(mode, str)
    task = _QWEN_TTS_TASKS[mode]
    command = [
        "--task",
        task,
        "--family",
        "qwen3_tts",
        "--model",
        model_argument,
        "--backend",
        "cuda",
        "--text",
        _required_string(payload, "text", "text"),
        "--out",
        str(output_directory / wav_filename),
    ]
    _append_optional(command, "--language", payload.get("language"))
    if mode == "custom-voice":
        command.extend(("--speaker", _required_string(payload, "voice", "speaker")))
        command.extend(("--instruct", _required_string(payload, "instruction", "instruction")))
    elif mode == "voice-clone":
        command.extend(
            (
                "--voice-ref",
                reference_audio_argument
                or _required_string(payload, "reference_audio_path", "reference audio path"),
                "--reference-text",
                _required_string(payload, "reference_text", "reference text"),
            )
        )
    else:
        command.extend(("--instruct", _required_string(payload, "instruction", "instruction")))
    return tuple(command)


def _gguf_artifacts_in_directory(model_directory: Path) -> tuple[Path, ...]:
    artifacts = tuple(
        candidate
        for candidate in model_directory.glob(_MODEL_ARTIFACT_GLOB)
        if candidate.is_file() and not candidate.is_symlink()
    )
    if not artifacts:
        return ()
    return artifacts


def _stage_asr_audio(
    *,
    source_path: Path,
    staged_path: Path,
    ffmpeg_command: Sequence[str],
    timeout_seconds: float,
    child_environment: Mapping[str, str],
) -> None:
    """Convert arbitrary input audio into a private 16 kHz mono 16-bit PCM WAV."""
    if not source_path.is_file():
        raise RuntimeTransportError("audio.cpp input audio is unavailable")
    try:
        completed = subprocess.run(
            [
                *ffmpeg_command,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-ar",
                str(_STAGED_AUDIO_RATE_HZ),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(staged_path),
            ],
            check=False,
            cwd=staged_path.parent,
            env=dict(child_environment),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeTransportError("audio.cpp input preparation failed") from exc
    if completed.returncode != 0:
        raise RuntimeTransportError("audio.cpp input preparation failed")


def _stage_reference_audio(request: Mapping[str, object], workspace: Path) -> Path | None:
    """Copy clone reference audio under a neutral name into the private workspace.

    Clone reference audio must be a mono WAV file. Sensitive text (reference
    transcript, instruction) is passed to the native process via argv flags:
    the pinned upstream revision exposes no file/stdin transport. Never include
    such text in exceptions, logs, receipts, or metadata.
    """
    raw_reference = request.get("reference_audio_path")
    if raw_reference is None:
        return None
    if not isinstance(raw_reference, str) or not raw_reference:
        raise RuntimeProtocolError("audio.cpp clone reference audio is unavailable")
    try:
        source = Path(raw_reference).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeProtocolError("audio.cpp clone reference audio is unavailable") from exc
    if not source.is_file():
        raise RuntimeProtocolError("audio.cpp clone reference audio must be a regular file")
    staged = workspace / "reference.wav"
    try:
        shutil.copyfile(source, staged)
        os.chmod(staged, 0o600)
        with wave.open(str(staged), "rb") as audio:
            if audio.getnframes() <= 0:
                raise RuntimeProtocolError(
                    "audio.cpp clone reference audio contains no audio frames"
                )
            if audio.getnchannels() != 1:
                raise RuntimeProtocolError("audio.cpp clone reference audio must be mono")
    except RuntimeProtocolError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise RuntimeProtocolError(
            "audio.cpp clone reference audio is not a readable WAV file"
        ) from exc
    return staged


def _staged_audio_duration_s(staged_audio_path: Path) -> float:
    try:
        with wave.open(str(staged_audio_path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getframerate() != _STAGED_AUDIO_RATE_HZ:
                raise RuntimeProtocolError("audio.cpp input staging produced an invalid WAV format")
            duration_s = audio.getnframes() / audio.getframerate()
    except (OSError, wave.Error) as exc:
        raise RuntimeProtocolError(
            "audio.cpp input staging did not produce a readable WAV"
        ) from exc
    if duration_s <= 0:
        raise RuntimeProtocolError("audio.cpp staged input audio contains no frames")
    return duration_s


def build_audio_cpp_cli_arguments(
    *,
    family: str,
    payload: Mapping[str, object],
    model_paths: Mapping[str, Path],
    output_directory: Path,
    audio_argument: str | None = None,
    reference_audio_argument: str | None = None,
) -> tuple[str, ...]:
    """Map a decoded request to native ``audiocpp_cli`` argv arguments."""

    model_path = model_paths.get(family)
    if model_path is None or not model_path.is_file() and not model_path.is_dir():
        raise RuntimeTransportError("audio.cpp native model artifact is unavailable")
    model_argument = _model_argument_for_cli(model_path, family)
    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    forced_aligner: str | None = None
    if family == "qwen3-asr" and payload.get("timestamp_mode") == "word":
        aligner = model_paths.get("qwen3-forced-aligner")
        if aligner is None or not aligner.is_file() and not aligner.is_dir():
            raise RuntimeTransportError("audio.cpp native forced aligner artifact is unavailable")
        forced_aligner = _model_argument_for_cli(aligner, "qwen3-forced-aligner")
    return build_audio_cpp_family_arguments(
        family=family,
        payload=payload,
        model_argument=model_argument,
        output_directory=output_directory,
        audio_argument=audio_argument,
        forced_aligner_argument=forced_aligner,
        reference_audio_argument=reference_audio_argument,
    )


def _model_argument_for_cli(model_path: Path, family: str) -> str:
    if model_path.is_file():
        return str(model_path.resolve())
    artifacts = _gguf_artifacts_in_directory(model_path)
    if not artifacts:
        raise RuntimeTransportError(
            f"audio.cpp native {family} model directory contains no GGUF artifact"
        )
    if len(artifacts) > 1:
        raise RuntimeTransportError(
            f"audio.cpp native {family} model directory must contain exactly one GGUF artifact"
        )
    return str(artifacts[0].resolve())


def decode_audio_cpp_cli_response(
    *,
    request_id: str,
    family: str,
    payload: Mapping[str, object],
    output_directory: Path,
    wav_filename: str = "speech.wav",
    expected_sample_rate_hz: int | None = None,
) -> Mapping[str, object]:
    """Decode native text, timing and WAV outputs back into the VOP response envelope."""

    spec = _FAMILY_SPECS[family]
    if spec.operation == "tts":
        return _encode_wav_response(
            request_id=request_id,
            output_path=output_directory / wav_filename,
            expected_sample_rate_hz=(
                24_000
                if family == "omnivoice" and expected_sample_rate_hz is None
                else expected_sample_rate_hz
            ),
        )
    response: dict[str, object] = {
        "transcript": _read_text_output(output_directory / "transcript.txt"),
    }
    if payload.get("timestamp_mode") == "word":
        if family == "nemotron-3.5-asr":
            response["word_timestamps"] = decode_audio_cpp_words(
                output_directory / "words.json", missing_ok=True, allow_blank_text=True
            )
            response["words_emitted"] = (output_directory / "words.json").is_file()
        else:
            response["forced_aligner_available"] = True
            response["words"] = decode_audio_cpp_words(output_directory / "words.json")
    segments = _decode_json_output(output_directory / "segments.json", "segments")
    if segments is not None:
        response["segments"] = segments
    return {
        "schema_version": 1,
        "request_id": request_id,
        "ok": True,
        "response": response,
    }


def decode_audio_cpp_words(
    words_path: Path, *, missing_ok: bool = False, allow_blank_text: bool = False
) -> list[dict[str, object]]:
    raw = _decode_json_output(words_path, "words", missing_ok=missing_ok)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        raw = raw.get("words", raw.get("word_timestamps"))
    if not isinstance(raw, list):
        raise RuntimeProtocolError("audio.cpp CLI words output must be a list")
    words: list[dict[str, object]] = []
    for index, raw_word in enumerate(raw):
        if not isinstance(raw_word, Mapping):
            raise RuntimeProtocolError(f"audio.cpp CLI word {index} must be an object")
        text = raw_word.get("word", raw_word.get("text"))
        start_sample = raw_word.get("start_sample")
        end_sample = raw_word.get("end_sample")
        if not isinstance(text, str) or (not text and not allow_blank_text):
            raise RuntimeProtocolError(f"audio.cpp CLI word {index} has no text")
        start_s = _sample_seconds(start_sample, index, "start")
        end_s = _sample_seconds(end_sample, index, "end")
        if end_s < start_s:
            raise RuntimeProtocolError(f"audio.cpp CLI word {index} has reversed boundaries")
        word = dict(raw_word)
        word["text"] = text
        word["start_s"] = start_s
        word["end_s"] = end_s
        if "confidence" in raw_word:
            word["confidence"] = raw_word["confidence"]
        words.append(word)
    return words


class AudioCppNativeCLITransport:
    """Host-native Windows launcher for the platform-neutral audio.cpp CLI codec."""

    def __init__(
        self,
        *,
        executable_path: Path,
        model_paths: Mapping[str, Path],
        timeout_seconds: float = 300.0,
        host_platform: str | None = None,
        ffmpeg_command: Sequence[str] = ("ffmpeg",),
    ) -> None:
        self._host_platform = host_platform or sys.platform
        if not self._host_platform.startswith("win"):
            raise ValueError("audio.cpp native CLI transport is Windows-only")
        if not ffmpeg_command or not all(ffmpeg_command):
            raise ValueError("audio.cpp ffmpeg command must not be empty")
        executable = executable_path.expanduser().resolve()
        if executable.suffix.casefold() != ".exe" or not executable.is_file():
            raise ValueError("audio.cpp native executable is unavailable")
        if timeout_seconds <= 0:
            raise ValueError("audio.cpp native timeout must be positive")
        resolved_models = {name: path.expanduser().resolve() for name, path in model_paths.items()}
        if not resolved_models or any(
            not path.is_file() and not path.is_dir() for path in resolved_models.values()
        ):
            raise ValueError("audio.cpp native model artifacts are unavailable")
        for model_path in resolved_models.values():
            if model_path.is_dir() and not _gguf_artifacts_in_directory(model_path):
                raise ValueError("audio.cpp native model directory is unavailable")
        self._executable_path = executable
        self._model_paths = resolved_models
        self._timeout_seconds = timeout_seconds
        self._ffmpeg_command = tuple(ffmpeg_command)
        self._lock = RLock()
        self._processes: dict[str, subprocess.Popen[str] | None] = {}
        self._cancelled: set[str] = set()

    def invoke(self, request_id: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        request = decode_audio_cpp_cli_request(payload)
        family = request["family"]
        assert isinstance(family, str)
        spec = _FAMILY_SPECS[family]
        with self._lock:
            self._processes[request_id] = None
        try:
            with tempfile.TemporaryDirectory(prefix=_PRIVATE_TMP_PREFIX) as temporary_directory:
                workspace = Path(temporary_directory)
                output_directory = workspace / "output"
                with self._lock:
                    if request_id in self._cancelled:
                        raise RuntimeTransportError("audio.cpp native invocation cancelled")
                staged_audio_path: Path | None = None
                staged_duration_s: float | None = None
                if spec.operation == "asr":
                    raw_audio = request["audio_path"]
                    assert isinstance(raw_audio, str)
                    staged_audio_path = workspace / "input.wav"
                    _stage_asr_audio(
                        source_path=Path(raw_audio),
                        staged_path=staged_audio_path,
                        ffmpeg_command=self._ffmpeg_command,
                        timeout_seconds=self._timeout_seconds,
                        child_environment=_windows_child_environment(),
                    )
                    staged_duration_s = _staged_audio_duration_s(staged_audio_path)
                else:
                    staged_reference_path = _stage_reference_audio(request, workspace)
                    if staged_reference_path is not None:
                        staged_audio_path = staged_reference_path
                with self._lock:
                    if request_id in self._cancelled:
                        raise RuntimeTransportError("audio.cpp native invocation cancelled")
                command = (
                    str(self._executable_path),
                    *build_audio_cpp_cli_arguments(
                        family=family,
                        payload=request,
                        model_paths=self._model_paths,
                        output_directory=output_directory,
                        audio_argument=(
                            str(staged_audio_path)
                            if spec.operation == "asr" and staged_audio_path is not None
                            else None
                        ),
                        reference_audio_argument=(
                            str(staged_audio_path)
                            if spec.operation != "asr" and staged_audio_path is not None
                            else None
                        ),
                    ),
                )
                with self._lock:
                    if request_id in self._cancelled:
                        raise RuntimeTransportError("audio.cpp native invocation cancelled")
                    process = self._start_process(command, workspace)
                    self._processes[request_id] = process
                try:
                    process.communicate(timeout=self._timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    self._terminate_process(process)
                    raise RuntimeTransportError("audio.cpp native invocation timed out") from exc
                with self._lock:
                    cancelled = request_id in self._cancelled
                if cancelled:
                    raise RuntimeTransportError("audio.cpp native invocation cancelled")
                if process.returncode != 0:
                    raise RuntimeTransportError(
                        f"audio.cpp native process exited with code {process.returncode}"
                    )
                return self._decode_response(
                    request_id=request_id,
                    family=family,
                    request=request,
                    output_directory=output_directory,
                    staged_duration_s=staged_duration_s,
                )
        finally:
            with self._lock:
                self._processes.pop(request_id, None)
                self._cancelled.discard(request_id)

    def cancel(self, request_id: str) -> None:
        with self._lock:
            self._cancelled.add(request_id)
            process = self._processes.get(request_id)
        if process is not None:
            self._terminate_process(process)

    def close(self) -> None:
        with self._lock:
            request_ids = tuple(self._processes)
        for request_id in request_ids:
            self.cancel(request_id)

    def _start_process(self, command: tuple[str, ...], workspace: Path) -> subprocess.Popen[str]:
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=workspace,
                env=_windows_child_environment(),
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                creationflags=_windows_creation_flags(),
            )
        except OSError as exc:
            raise RuntimeTransportError("audio.cpp native process could not be started") from exc

    @staticmethod
    def _decode_response(
        *,
        request_id: str,
        family: str,
        request: Mapping[str, object],
        output_directory: Path,
        staged_duration_s: float | None,
    ) -> Mapping[str, object]:
        encoded = decode_audio_cpp_cli_response(
            request_id=request_id,
            family=family,
            payload=request,
            output_directory=output_directory,
        )
        if staged_duration_s is None:
            return encoded
        response = encoded["response"]
        assert isinstance(response, Mapping)
        return {
            "schema_version": 1,
            "request_id": request_id,
            "ok": True,
            "response": {**response, "duration_s": staged_duration_s},
        }

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        control_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if control_break is not None:
            try:
                process.send_signal(control_break)
            except (OSError, ValueError):
                pass
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                return
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                return


def _validate_asr_request(family: str, payload: Mapping[str, object]) -> None:
    allowed = {"audio_path", "model_id", "language", "timestamp_mode", "context_text"}
    _reject_unexpected_fields(payload, allowed)
    _required_string(payload, "audio_path", "audio path")
    language = payload.get("language")
    if language is not None and not isinstance(language, str):
        raise RuntimeProtocolError("audio.cpp CLI request language must be a string or null")
    if payload.get("timestamp_mode") not in {"none", "word"}:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported timestamp mode")
    context_text = payload.get("context_text")
    if context_text is not None and not isinstance(context_text, str):
        raise RuntimeProtocolError("audio.cpp CLI request context must be a string or null")


def _validate_tts_request(family: str, payload: Mapping[str, object]) -> None:
    allowed = {"text", "model_id", "voice", "language"}
    if family == "omnivoice":
        allowed.update(
            {
                "seed",
                "num_inference_steps",
                "guidance_scale",
                "instruction",
                "text_chunk_size",
                "reference_audio_path",
                "reference_text",
                "omnivoice_mode",
                "style_condition",
                "design_instruction",
            }
        )
    _reject_unexpected_fields(payload, allowed)
    _required_string(payload, "text", "text")
    for field in ("voice", "language"):
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise RuntimeProtocolError(f"audio.cpp CLI request {field} must be a string or null")
    reference_audio_path = payload.get("reference_audio_path")
    if reference_audio_path is not None and (
        not isinstance(reference_audio_path, str) or not reference_audio_path.strip()
    ):
        raise RuntimeProtocolError(
            "audio.cpp CLI request reference audio path must be a string or null"
        )
    reference_text = payload.get("reference_text")
    if reference_text is not None and not isinstance(reference_text, str):
        raise RuntimeProtocolError("audio.cpp CLI request reference text must be a string or null")
    if (reference_audio_path is None) != (reference_text is None):
        raise RuntimeProtocolError(
            "audio.cpp CLI request clone requires both reference audio and reference text"
        )
    if family == "omnivoice":
        if not isinstance(payload.get("language"), str) or not str(payload["language"]).strip():
            raise RuntimeProtocolError("OmniVoice request language must not be blank")
        seed = payload.get("seed")
        steps = payload.get("num_inference_steps")
        guidance = payload.get("guidance_scale")
        text_chunk_size = payload.get("text_chunk_size")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise RuntimeProtocolError("OmniVoice request seed must be a non-negative integer")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise RuntimeProtocolError(
                "OmniVoice request inference steps must be a positive integer"
            )
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, (int, float))
            or not isfinite(float(guidance))
            or float(guidance) < 0
        ):
            raise RuntimeProtocolError(
                "OmniVoice request guidance scale must be a non-negative number"
            )
        if (
            isinstance(text_chunk_size, bool)
            or not isinstance(text_chunk_size, int)
            or text_chunk_size < 1
        ):
            raise RuntimeProtocolError(
                "OmniVoice request text chunk size must be a positive integer"
            )
        mode = payload.get("omnivoice_mode")
        if payload.get("instruction") is not None:
            raise RuntimeProtocolError(
                "OmniVoice request does not accept the Qwen instruction field"
            )
        if mode is None:
            _required_string(payload, "style_condition", "OmniVoice style condition")
            if (
                payload.get("reference_audio_path") is not None
                or payload.get("reference_text") is not None
            ):
                raise RuntimeProtocolError(
                    "OmniVoice fixed-style request does not accept clone fields"
                )
            if payload.get("design_instruction") is not None:
                raise RuntimeProtocolError(
                    "OmniVoice fixed-style request does not accept design fields"
                )
            return
        if mode not in ("fixed-style", "clone", "design"):
            raise RuntimeProtocolError("OmniVoice request has an unsupported mode")
        if mode == "fixed-style":
            _required_string(payload, "style_condition", "OmniVoice style condition")
            if (
                payload.get("reference_audio_path") is not None
                or payload.get("reference_text") is not None
            ):
                raise RuntimeProtocolError(
                    "OmniVoice fixed-style request does not accept clone fields"
                )
            if payload.get("design_instruction") is not None:
                raise RuntimeProtocolError(
                    "OmniVoice fixed-style request does not accept design fields"
                )
            return
        if mode == "clone":
            _required_string(payload, "reference_audio_path", "reference audio path")
            _required_string(payload, "reference_text", "reference text")
            if (
                payload.get("style_condition") is not None
                or payload.get("design_instruction") is not None
            ):
                raise RuntimeProtocolError(
                    "OmniVoice clone request does not accept style or design fields"
                )
            return
        _required_string(payload, "design_instruction", "design instruction")
        if payload.get("style_condition") is not None:
            raise RuntimeProtocolError(
                "OmniVoice design request does not accept fixed-style fields"
            )
        if (
            payload.get("reference_audio_path") is not None
            or payload.get("reference_text") is not None
        ):
            raise RuntimeProtocolError("OmniVoice design request does not accept clone fields")


def _validate_qwen_tts_request(payload: Mapping[str, object]) -> None:
    allowed = {
        "text",
        "model_id",
        "model_artifact_path",
        "voice",
        "language",
        "mode",
        "instruction",
        "reference_audio_path",
        "reference_text",
    }
    _reject_unexpected_fields(payload, allowed)
    _required_string(payload, "text", "text")
    if payload.get("model_id") not in _QWEN_TTS_MODEL_IDS:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported model")
    model_artifact_path = payload.get("model_artifact_path")
    if model_artifact_path is not None and (
        not isinstance(model_artifact_path, str) or not model_artifact_path.strip()
    ):
        raise RuntimeProtocolError(
            "audio.cpp CLI request model artifact path must be a string or null"
        )
    language = payload.get("language")
    if language is not None and not isinstance(language, str):
        raise RuntimeProtocolError("audio.cpp CLI request language must be a string or null")
    mode = payload.get("mode")
    if mode is None:
        voice = payload.get("voice")
        if voice is not None and not isinstance(voice, str):
            raise RuntimeProtocolError("audio.cpp CLI request voice must be a string or null")
        return
    if not isinstance(mode, str) or mode not in _QWEN_TTS_TASKS:
        raise RuntimeProtocolError("audio.cpp CLI request has an unsupported Qwen3-TTS mode")
    if payload.get("model_id") not in _QWEN_TTS_MODELS_BY_MODE[mode]:
        raise RuntimeProtocolError("Qwen3-TTS request model does not support the requested mode")
    if mode == "custom-voice":
        _required_string(payload, "voice", "speaker")
        _required_string(payload, "instruction", "instruction")
        return
    if mode == "voice-clone":
        if payload.get("voice") is not None:
            raise RuntimeProtocolError("Qwen3-TTS clone request must not declare a speaker")
        _required_string(payload, "reference_audio_path", "reference audio path")
        _required_string(payload, "reference_text", "reference text")
        return
    if payload.get("voice") is not None:
        raise RuntimeProtocolError("Qwen3-TTS VoiceDesign request must not declare a speaker")
    _required_string(payload, "instruction", "instruction")


def _reject_unexpected_fields(payload: Mapping[str, object], allowed: set[str]) -> None:
    if set(payload) - allowed:
        raise RuntimeProtocolError("audio.cpp CLI request contains unsupported runtime fields")


def _required_string(payload: Mapping[str, object], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeProtocolError(f"audio.cpp CLI request {label} must not be blank")
    return value


def _append_optional(command: list[str], flag: str, value: object) -> None:
    if isinstance(value, str) and value:
        command.extend((flag, value))


def _read_text_output(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").removesuffix("\n")
    except OSError as exc:
        raise RuntimeProtocolError(
            "audio.cpp CLI response did not contain transcript output"
        ) from exc


def _decode_json_output(path: Path, label: str, *, missing_ok: bool = False) -> object | None:
    if not path.is_file() and (label == "segments" or missing_ok):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeProtocolError(
            f"audio.cpp CLI response did not contain valid {label} JSON"
        ) from exc


def _sample_seconds(value: object, index: int, boundary: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise RuntimeProtocolError(
            f"audio.cpp CLI word {index} {boundary}_sample must be a finite number"
        )
    seconds = float(value) / _STAGED_AUDIO_RATE_HZ
    if seconds < 0:
        raise RuntimeProtocolError(
            f"audio.cpp CLI word {index} {boundary}_sample must be non-negative"
        )
    return seconds


def _encode_wav_response(
    *, request_id: str, output_path: Path, expected_sample_rate_hz: int | None
) -> Mapping[str, object]:
    try:
        audio_bytes = output_path.read_bytes()
    except OSError as exc:
        raise RuntimeProtocolError("audio.cpp CLI did not produce audio output") from exc
    if len(audio_bytes) <= 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise RuntimeProtocolError("audio.cpp CLI output is not a nonempty RIFF/WAVE file")
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate_hz = audio.getframerate()
            frames = audio.getnframes()
    except wave.Error as exc:
        raise RuntimeProtocolError("audio.cpp CLI output is not a readable WAV file") from exc
    if channels != 1:
        raise RuntimeProtocolError("audio.cpp CLI output must be mono")
    if expected_sample_rate_hz is not None and sample_rate_hz != expected_sample_rate_hz:
        raise RuntimeProtocolError("audio.cpp CLI output has an unexpected sample rate")
    if frames <= 0:
        raise RuntimeProtocolError("audio.cpp CLI output contains no audio frames")
    return {
        "schema_version": 1,
        "request_id": request_id,
        "ok": True,
        "response": {
            "audio_bytes": audio_bytes,
            "audio_format": "wav",
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "duration_s": frames / sample_rate_hz,
        },
    }


def _windows_creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def _windows_child_environment() -> dict[str, str]:
    keys = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LANG", "LC_ALL")
    return {name: value for name in keys if (value := os.environ.get(name)) is not None}
