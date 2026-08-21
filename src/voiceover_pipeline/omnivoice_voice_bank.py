"""Local OmniVoice voice bank: catalog loader, profile admission, and reference verification.

A voice bank lives outside the repository in a user-provided directory and
consists of ``catalog.json`` plus a ``voices/`` subdirectory. The catalog is
the only machine-visible metadata surface: profile reference paths are
relative to the bank root, and the loader rejects any absolute or escaping
path before it is ever used.
"""

from __future__ import annotations

import hashlib
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path

_VOICE_BANK_SCHEMA_VERSION = 1
_ORIGIN_MODES = ("auto", "design", "owner-reference")
_SHA256_HEX_LENGTH = 64
_DRIVE_LETTER_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_READ_CHUNK_BYTES = 1 << 20


class VoiceBankError(ValueError):
    """A voice bank catalog or profile is invalid, unsafe, or unusable."""


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    display_name: str
    description: str
    language: str
    reference_audio: str
    reference_text: str
    reference_sha256: str
    origin: dict[str, str | int | None]


@dataclass(frozen=True)
class VoiceBankCatalog:
    root: Path
    default_voice: str
    profiles: tuple[VoiceProfile, ...]


def load_voice_bank(catalog_path: Path) -> VoiceBankCatalog:
    """Load and validate a voice bank catalog.

    Raises :class:`VoiceBankError` for any structural, containment, or
    reference violation. Error messages reference only the catalog filename,
    never the absolute bank root.
    """
    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        raise VoiceBankError(f"voice bank catalog '{catalog_path.name}' is not a readable file")
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' is not a valid JSON file"
        ) from exc
    if not isinstance(raw, dict):
        raise VoiceBankError(f"voice bank catalog '{catalog_path.name}' must be a JSON object")
    if raw.get("schema_version") != _VOICE_BANK_SCHEMA_VERSION:
        raise VoiceBankError(f"voice bank catalog '{catalog_path.name}' requires schema_version 1")

    raw_voices = raw.get("voices")
    if not isinstance(raw_voices, list):
        raise VoiceBankError(f"voice bank catalog '{catalog_path.name}' voices must be a list")

    default_voice = raw.get("default_voice")
    if not isinstance(default_voice, str) or not default_voice:
        raise VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' default_voice must be a non-empty string"
        )

    root = catalog_path.parent
    root_resolved = root.resolve()
    seen_ids: set[str] = set()
    profiles: list[VoiceProfile] = []
    for raw_profile in raw_voices:
        profile = _voice_profile(catalog_path, raw_profile)
        if profile.id in seen_ids:
            raise VoiceBankError(
                f"voice bank catalog '{catalog_path.name}' contains duplicate profile id "
                f"'{profile.id}'"
            )
        seen_ids.add(profile.id)
        _validate_reference_path(catalog_path, profile, root, root_resolved)
        profiles.append(profile)

    if default_voice not in seen_ids:
        raise VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' default_voice '{default_voice}' "
            "does not match any profile id"
        )
    return VoiceBankCatalog(
        root=root,
        default_voice=default_voice,
        profiles=tuple(profiles),
    )


def _voice_profile_error(catalog_path: Path, voice_id: str | None, detail: str) -> VoiceBankError:
    if voice_id is None:
        return VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' has an invalid profile: {detail}"
        )
    return VoiceBankError(f"voice bank catalog '{catalog_path.name}' profile '{voice_id}' {detail}")


def _voice_profile(catalog_path: Path, raw_profile: object) -> VoiceProfile:
    if not isinstance(raw_profile, dict):
        raise _voice_profile_error(catalog_path, None, "must be a JSON object")

    voice_id = raw_profile.get("id")
    if not isinstance(voice_id, str) or not voice_id:
        raise _voice_profile_error(catalog_path, None, "id must be a non-empty string")

    display_name = raw_profile.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        raise _voice_profile_error(
            catalog_path, voice_id, "display_name must be a non-empty string"
        )

    description = raw_profile.get("description")
    if not isinstance(description, str):
        raise _voice_profile_error(catalog_path, voice_id, "description must be a string")

    language = raw_profile.get("language")
    if not isinstance(language, str) or not language:
        raise _voice_profile_error(catalog_path, voice_id, "language must be a non-empty string")

    reference_audio = raw_profile.get("reference_audio")
    if not isinstance(reference_audio, str) or not reference_audio:
        raise _voice_profile_error(
            catalog_path, voice_id, "reference_audio must be a non-empty string"
        )

    reference_text = raw_profile.get("reference_text")
    if not isinstance(reference_text, str) or not reference_text:
        raise _voice_profile_error(
            catalog_path, voice_id, "reference_text must be a non-empty string"
        )

    reference_sha256 = raw_profile.get("reference_sha256")
    if not isinstance(reference_sha256, str):
        raise _voice_profile_error(catalog_path, voice_id, "reference_sha256 must be a string")
    reference_sha256 = reference_sha256.strip().lower()
    if len(reference_sha256) != _SHA256_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in reference_sha256
    ):
        raise _voice_profile_error(
            catalog_path, voice_id, "reference_sha256 must be a 64-character hex digest"
        )

    origin = _parse_origin(catalog_path, voice_id, raw_profile.get("origin"))

    return VoiceProfile(
        id=voice_id,
        display_name=display_name,
        description=description,
        language=language,
        reference_audio=reference_audio,
        reference_text=reference_text,
        reference_sha256=reference_sha256,
        origin=origin,
    )


def _parse_origin(
    catalog_path: Path, voice_id: str, raw_origin: object
) -> dict[str, str | int | None]:
    if not isinstance(raw_origin, dict):
        raise _voice_profile_error(catalog_path, voice_id, "origin must be an object")
    origin_mode = raw_origin.get("mode")
    if origin_mode not in _ORIGIN_MODES:
        raise _voice_profile_error(
            catalog_path,
            voice_id,
            f"origin mode must be one of {', '.join(_ORIGIN_MODES)}",
        )
    instruction = raw_origin.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise _voice_profile_error(catalog_path, voice_id, "origin instruction must be a string")
    seed = raw_origin.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise _voice_profile_error(catalog_path, voice_id, "origin seed must be an integer")
    return {"mode": origin_mode, "instruction": instruction, "seed": seed}


def _validate_reference_path(
    catalog_path: Path, profile: VoiceProfile, root: Path, root_resolved: Path
) -> None:
    reference = profile.reference_audio
    if Path(reference).is_absolute() or _DRIVE_LETTER_PATTERN.match(reference):
        raise VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' profile '{profile.id}' "
            "reference_audio is an absolute path outside the bank"
        )
    parts = [part for part in re.split(r"[\\/]", reference) if part not in ("", ".")]
    if ".." in parts:
        raise VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' profile '{profile.id}' "
            "reference_audio contains a parent-directory escape"
        )
    try:
        resolved = (root / reference).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' profile '{profile.id}' "
            "reference_audio does not resolve inside the bank"
        ) from exc
    if not resolved.is_relative_to(root_resolved):
        raise VoiceBankError(
            f"voice bank catalog '{catalog_path.name}' profile '{profile.id}' "
            "reference_audio escapes the voice bank root"
        )


def resolve_bank_profile(catalog: VoiceBankCatalog, voice_id: str) -> tuple[VoiceProfile, Path]:
    """Resolve one profile and verify its reference file on disk.

    Verifies the file exists, is a regular file, its SHA-256 matches the
    catalog digest, and it is a readable mono WAV file. Bank references may
    use any sample rate; the native transport normalizes them during staging.
    """
    profile = next((item for item in catalog.profiles if item.id == voice_id), None)
    if profile is None:
        raise VoiceBankError(f"voice '{voice_id}' not found in the voice bank")
    reference_path = catalog.root / profile.reference_audio
    if not reference_path.is_file():
        raise VoiceBankError(
            f"voice bank profile '{voice_id}' reference audio is not a readable file"
        )
    if _sha256_file(reference_path) != profile.reference_sha256:
        raise VoiceBankError("SHA-256 mismatch")
    try:
        with wave.open(str(reference_path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getnframes() <= 0:
                raise VoiceBankError("must be a readable mono WAV file")
    except (OSError, EOFError, wave.Error) as exc:
        if isinstance(exc, VoiceBankError):
            raise
        raise VoiceBankError("must be a readable mono WAV file") from exc
    return profile, reference_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
