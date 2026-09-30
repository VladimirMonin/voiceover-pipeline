"""Contract tests for the bounded paid raw-audio receipt (``history.raw_receipt``).

Every fixture is synthetic and lives under ``tmp_path``: no real ``VOICEOVER_HOME``,
no ``.env``, no provider, model, database, or network path is touched. The
helpers are the offline emergency evidence writer/reader that runs before a later
SQLite artifact commit, so the tests exercise the crash-like stages explicitly: a
raw file without a receipt blocks verification, a valid receipt with its raw file
verifies, a repeat write is an idempotent no-overwrite no-op, and any conflicting
existing evidence is rejected instead of adopting another paid attempt.

Privacy, path safety, and fail-closed behavior are asserted directly: the receipt
carries exactly the bounded fields and no raw bytes or absolute run path, private
modes are applied to new files, a foreign, symlinked, or group-/world-accessible
``raw`` directory or evidence file is rejected, an oversized receipt is read only
in bounded blocks, filesystem failures report a fixed message without the run
path, unsupported format/chunk/id and malformed identity are rejected before
anything is written, and a tampered digest or identity fails verification.
"""

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from voiceover_pipeline.history.raw_receipt import (
    RAW_DIRECTORY_MODE,
    RAW_FILE_MODE,
    PaidRawReceiptConflictError,
    PaidRawReceiptVerificationError,
    verify_paid_raw_receipt,
    write_paid_raw_receipt,
)

ATTEMPT_UUID = "11111111-1111-4111-8111-111111111111"
PART_UUID = "22222222-2222-4222-8222-222222222222"
FINGERPRINT = "a" * 64
# An ASCII marker plus binary bytes, so a leak of raw content into the receipt
# text is detectable as well as a length or digest mismatch.
AUDIO = b"PAID-RAW-MARKER" + bytes(range(16))

_RECEIPT_KEYS = frozenset(
    {
        "artifact_type",
        "receipt_version",
        "attempt_uuid",
        "part_uuid",
        "synthesis_fingerprint",
        "chunk_id",
        "number",
        "format",
        "path",
        "size",
        "sha256",
        "remote_task_id",
        "generation_id",
    }
)

_POSIX = os.name == "posix"

# A run-root segment that must never appear in a public error message: the
# underlying OSError text normally embeds the absolute path.
_SENSITIVE_SEGMENT = "run-signed-token-secret"


class _RecordingReader:
    """Record every binary ``read`` size a wrapped file handle is asked for."""

    def __init__(self, handle, sizes):
        self._handle = handle
        self._sizes = sizes

    def read(self, size=-1):
        self._sizes.append(size)
        return self._handle.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._handle.close()
        return False


def _sensitive_run_root(tmp_path):
    """Return an existing absolute run root whose name must not leak."""
    root = tmp_path / _SENSITIVE_SEGMENT
    root.mkdir()
    return root


def _write(run_root, **overrides):
    kwargs = {
        "run_root": run_root,
        "attempt_uuid": ATTEMPT_UUID,
        "part_uuid": PART_UUID,
        "synthesis_fingerprint": FINGERPRINT,
        "chunk_id": "chunk_01",
        "number": 1,
        "audio_format": "mp3",
        "audio_bytes": AUDIO,
    }
    kwargs.update(overrides)
    return write_paid_raw_receipt(**kwargs)


def _verify(run_root, **overrides):
    kwargs = {
        "run_root": run_root,
        "attempt_uuid": ATTEMPT_UUID,
        "part_uuid": PART_UUID,
        "synthesis_fingerprint": FINGERPRINT,
        "chunk_id": "chunk_01",
        "number": 1,
        "audio_format": "mp3",
    }
    kwargs.update(overrides)
    return verify_paid_raw_receipt(**kwargs)


def _receipt_path(tmp_path, file_name="chunk_01.mp3"):
    return tmp_path / "raw" / f"{file_name}.receipt.json"


def _tamper_receipt(tmp_path, **fields):
    receipt = _receipt_path(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload.update(fields)
    receipt.write_text(json.dumps(payload), encoding="utf-8")


def test_write_then_verify_round_trip_returns_bounded_evidence(tmp_path):
    written = _write(tmp_path, remote_task_id="task-abc", generation_id="gen-123")

    assert written.relative_path == "raw/chunk_01.mp3"
    assert written.raw_path == tmp_path / "raw" / "chunk_01.mp3"
    assert written.sha256 == hashlib.sha256(AUDIO).hexdigest()
    assert written.size == len(AUDIO)

    verified = _verify(tmp_path)
    assert verified == written
    assert verified.raw_path.read_bytes() == AUDIO


def test_write_preserves_original_audio_bytes_and_digest(tmp_path):
    _write(tmp_path)

    assert (tmp_path / "raw" / "chunk_01.mp3").read_bytes() == AUDIO


@pytest.mark.skipif(not _POSIX, reason="POSIX mode bits")
def test_new_raw_directory_and_files_are_private(tmp_path):
    _write(tmp_path)

    raw_dir = tmp_path / "raw"
    assert stat.S_IMODE(raw_dir.stat().st_mode) == RAW_DIRECTORY_MODE
    assert stat.S_IMODE((raw_dir / "chunk_01.mp3").stat().st_mode) == RAW_FILE_MODE
    assert stat.S_IMODE((raw_dir / "chunk_01.mp3.receipt.json").stat().st_mode) == RAW_FILE_MODE


@pytest.mark.skipif(not _POSIX, reason="POSIX mode bits")
@pytest.mark.parametrize("file_name", ["chunk_01.mp3", "chunk_01.mp3.receipt.json"])
@pytest.mark.parametrize("operation", ["write", "verify"])
def test_insecure_existing_evidence_is_rejected_without_chmod(tmp_path, file_name, operation):
    written = _write(tmp_path)
    evidence_path = written.raw_path.parent / file_name
    evidence_path.chmod(0o644)

    with pytest.raises(
        PaidRawReceiptConflictError if operation == "write" else PaidRawReceiptVerificationError
    ):
        if operation == "write":
            _write(tmp_path)
        else:
            _verify(tmp_path)

    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o644
    assert written.raw_path.read_bytes() == AUDIO


def test_repeat_write_is_idempotent_and_does_not_overwrite(tmp_path):
    first = _write(tmp_path)
    before = first.raw_path.stat()

    second = _write(tmp_path)
    after = second.raw_path.stat()

    assert second == first
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert first.raw_path.read_bytes() == AUDIO


def test_conflicting_raw_bytes_are_rejected_and_preserved(tmp_path):
    first = _write(tmp_path)

    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path, audio_bytes=b"PAID-RAW-MARKER-different")

    assert first.raw_path.read_bytes() == AUDIO
    stored = json.loads(_receipt_path(tmp_path).read_text(encoding="utf-8"))
    assert stored["sha256"] == hashlib.sha256(AUDIO).hexdigest()


def test_conflicting_receipt_identity_is_rejected(tmp_path):
    first = _write(tmp_path)

    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path, attempt_uuid="33333333-3333-4333-8333-333333333333")

    assert first.raw_path.read_bytes() == AUDIO


def test_raw_without_receipt_blocks_verification_and_write(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(mode=RAW_DIRECTORY_MODE)
    raw_dir.chmod(RAW_DIRECTORY_MODE)
    (raw_dir / "chunk_01.mp3").write_bytes(AUDIO)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)
    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path)


def test_receipt_without_raw_is_rejected(tmp_path):
    written = _write(tmp_path)
    written.raw_path.unlink()

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)
    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path)


def test_verify_rejects_tampered_raw_digest(tmp_path):
    written = _write(tmp_path)
    written.raw_path.write_bytes(b"PAID-RAW-MARKER-tampered")

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_receipt_digest_that_does_not_match_file(tmp_path):
    _write(tmp_path)
    _tamper_receipt(tmp_path, sha256="c" * 64)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_receipt_size_that_does_not_match_file(tmp_path):
    _write(tmp_path)
    _tamper_receipt(tmp_path, size=len(AUDIO) + 1)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"attempt_uuid": "33333333-3333-4333-8333-333333333333"},
        {"part_uuid": "44444444-4444-4444-8444-444444444444"},
        {"synthesis_fingerprint": "b" * 64},
    ],
)
def test_verify_rejects_mismatched_identity(tmp_path, override):
    _write(tmp_path)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path, **override)


def test_verify_rejects_mismatched_remote_identity(tmp_path):
    _write(tmp_path, remote_task_id="task-abc")

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path, remote_task_id="task-xyz")
    # Omitting the remote id accepts whatever bounded id the receipt already holds.
    assert _verify(tmp_path).remote_task_id == "task-abc"
    assert _verify(tmp_path, remote_task_id="task-abc").remote_task_id == "task-abc"


def test_verify_rejects_receipt_naming_a_different_chunk(tmp_path):
    _write(tmp_path)
    _tamper_receipt(tmp_path, chunk_id="chunk_02", number=2, path="raw/chunk_02.mp3")

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


@pytest.mark.parametrize("bad_path", ["../outside.mp3", "/etc/passwd", "raw/../evil.mp3"])
def test_verify_rejects_traversal_shaped_receipt_path(tmp_path, bad_path):
    _write(tmp_path)
    _tamper_receipt(tmp_path, path=bad_path)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_missing_raw_directory(tmp_path):
    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_missing_raw_audio(tmp_path):
    written = _write(tmp_path)
    written.raw_path.unlink()

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_receipt_with_extra_fields(tmp_path):
    _write(tmp_path)
    _tamper_receipt(tmp_path, secret="https://signed.example/leak?token=abc")

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_non_json_receipt(tmp_path):
    _write(tmp_path)
    _receipt_path(tmp_path).write_bytes(b"not json")

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_oversized_receipt(tmp_path):
    _write(tmp_path)
    _receipt_path(tmp_path).write_bytes(b"{" + b" " * (64 * 1024) + b"}")

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_verify_rejects_deeply_nested_receipt_as_fixed_error(tmp_path):
    _write(tmp_path)
    # 20001 bytes, below the 64 KiB bound, but deep enough that json.loads hits
    # the recursion limit: the reader must report its fixed verification error
    # instead of leaking a raw RecursionError.
    _receipt_path(tmp_path).write_bytes(b"[" * 10_000 + b"0" + b"]" * 10_000)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


@pytest.mark.skipif(not _POSIX, reason="sparse oversized file via truncate")
def test_oversized_receipt_is_read_in_bounded_blocks(tmp_path, monkeypatch):
    _write(tmp_path)
    receipt = _receipt_path(tmp_path)
    # A sparse oversized file proves the reader stays bounded without the test
    # ever allocating the multi-megabyte contents.
    with receipt.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024)

    sizes: list[int] = []
    real_open = Path.open

    def recording_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == receipt:
            return _RecordingReader(handle, sizes)
        return handle

    monkeypatch.setattr(Path, "open", recording_open)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)

    assert sizes, "the receipt read was not instrumented"
    assert all(0 <= size <= 64 * 1024 + 1 for size in sizes)


@pytest.mark.skipif(not _POSIX, reason="POSIX mode bits")
def test_write_rejects_group_or_world_accessible_raw_directory(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.chmod(0o777)

    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path)

    assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o777
    assert list(raw_dir.iterdir()) == []


@pytest.mark.skipif(not _POSIX, reason="POSIX mode bits")
def test_verify_rejects_widened_raw_directory(tmp_path):
    _write(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.chmod(0o777)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)

    assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o777


@pytest.mark.skipif(not _POSIX, reason="symlink support")
def test_write_rejects_symlinked_raw_directory(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, tmp_path / "raw")

    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path)
    assert list(elsewhere.iterdir()) == []


@pytest.mark.skipif(not _POSIX, reason="symlink support")
def test_write_rejects_symlinked_raw_file(tmp_path):
    written = _write(tmp_path)
    written.raw_path.unlink()
    target = tmp_path / "target.bin"
    target.write_bytes(AUDIO)
    os.symlink(target, written.raw_path)

    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path)


@pytest.mark.skipif(not _POSIX, reason="symlink support")
def test_verify_rejects_symlinked_evidence(tmp_path):
    written = _write(tmp_path)
    written.raw_path.unlink()
    target = tmp_path / "target.bin"
    target.write_bytes(AUDIO)
    os.symlink(target, written.raw_path)

    with pytest.raises(PaidRawReceiptVerificationError):
        _verify(tmp_path)


def test_write_rejects_raw_path_that_is_a_file(tmp_path):
    (tmp_path / "raw").write_bytes(b"not a directory")

    with pytest.raises(PaidRawReceiptConflictError):
        _write(tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempt_uuid": "not-a-uuid"},
        {"part_uuid": "not-a-uuid"},
        {"synthesis_fingerprint": "not-a-digest"},
        {"chunk_id": "chunk_99", "number": 1},
        {"number": 0},
        {"number": True},
        {"number": 1_000_001},
        {"audio_format": "ogg"},
        {"audio_bytes": b""},
        {"audio_bytes": "not bytes"},
        {"remote_task_id": "has/slash"},
        {"remote_task_id": "https://signed.example/x?token=secret"},
    ],
)
def test_invalid_inputs_are_rejected_before_any_write(tmp_path, overrides):
    with pytest.raises(ValueError):
        _write(tmp_path, **overrides)

    assert list(tmp_path.iterdir()) == []


def test_relative_run_root_is_rejected_before_any_write(tmp_path):
    with pytest.raises(ValueError):
        _write("relative/run")

    assert list(tmp_path.iterdir()) == []


def test_missing_run_root_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _write(tmp_path / "does-not-exist")


def test_receipt_records_only_bounded_anonymous_fields(tmp_path):
    _write(tmp_path, remote_task_id="task-abc", generation_id="gen-123")

    text = _receipt_path(tmp_path).read_text(encoding="utf-8")
    payload = json.loads(text)
    assert set(payload) == _RECEIPT_KEYS
    assert payload["path"] == "raw/chunk_01.mp3"
    assert payload["format"] == "mp3"
    assert payload["size"] == len(AUDIO)
    assert payload["sha256"] == hashlib.sha256(AUDIO).hexdigest()
    assert payload["remote_task_id"] == "task-abc"
    # No raw bytes and no absolute run path leak into the anonymous receipt.
    assert "PAID-RAW-MARKER" not in text
    assert str(tmp_path) not in text


def test_atomic_write_leaves_no_temp_files(tmp_path):
    _write(tmp_path)

    leftovers = [
        entry.name for entry in (tmp_path / "raw").iterdir() if entry.name.endswith(".tmp")
    ]
    assert leftovers == []


def test_uuid_case_is_normalized_for_matching(tmp_path):
    _write(tmp_path, attempt_uuid=ATTEMPT_UUID.upper())

    assert _verify(tmp_path, attempt_uuid=ATTEMPT_UUID).attempt_uuid == ATTEMPT_UUID


def test_write_and_verify_support_dialogue_turn_ids(tmp_path):
    written = _write(tmp_path, chunk_id="turn_0003", number=3, audio_format="wav")

    assert written.relative_path == "raw/turn_0003.wav"
    assert _verify(tmp_path, chunk_id="turn_0003", number=3, audio_format="wav") == written


def test_pcm16_format_uses_pcm_extension(tmp_path):
    written = _write(tmp_path, audio_format="pcm16")

    assert written.relative_path == "raw/chunk_01.pcm"
    assert written.raw_path.name == "chunk_01.pcm"


def test_mkdir_failure_hides_run_path_and_writes_nothing(tmp_path, monkeypatch):
    root = _sensitive_run_root(tmp_path)

    def failing_mkdir(path, *args, **kwargs):
        raise OSError(13, "Permission denied", str(path))

    monkeypatch.setattr(os, "mkdir", failing_mkdir)

    with pytest.raises(PaidRawReceiptConflictError) as excinfo:
        _write(root)

    message = str(excinfo.value)
    assert str(root) not in message
    assert _SENSITIVE_SEGMENT not in message
    assert list(root.iterdir()) == []


def test_receipt_open_failure_hides_run_path(tmp_path, monkeypatch):
    root = _sensitive_run_root(tmp_path)
    _write(root)
    raw_path = root / "raw" / "chunk_01.mp3"
    receipt = _receipt_path(root)
    real_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == receipt:
            raise OSError(13, "Permission denied", str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(PaidRawReceiptVerificationError) as excinfo:
        _verify(root)

    message = str(excinfo.value)
    assert str(root) not in message
    assert _SENSITIVE_SEGMENT not in message
    assert raw_path.read_bytes() == AUDIO


def test_stat_failure_hides_run_path(tmp_path, monkeypatch):
    root = _sensitive_run_root(tmp_path)
    _write(root)
    raw_path = root / "raw" / "chunk_01.mp3"
    receipt = _receipt_path(root)
    real_stat = Path.stat

    def failing_stat(self, *args, **kwargs):
        if self == receipt:
            raise OSError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)

    with pytest.raises(PaidRawReceiptVerificationError) as excinfo:
        _verify(root)

    message = str(excinfo.value)
    assert str(root) not in message
    assert _SENSITIVE_SEGMENT not in message
    assert raw_path.read_bytes() == AUDIO


def test_replace_failure_hides_run_path_and_leaves_no_evidence(tmp_path, monkeypatch):
    root = _sensitive_run_root(tmp_path)

    def failing_replace(src, dst, *args, **kwargs):
        raise OSError(13, "Permission denied", str(dst))

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(PaidRawReceiptConflictError) as excinfo:
        _write(root)

    message = str(excinfo.value)
    assert str(root) not in message
    assert _SENSITIVE_SEGMENT not in message
    raw_dir = root / "raw"
    assert list(raw_dir.iterdir()) == []
    assert not (raw_dir / "chunk_01.mp3").exists()
    assert not _receipt_path(root).exists()
