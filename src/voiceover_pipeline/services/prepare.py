"""Bounded script preparation and prepared speech parts for one CLI generation run."""

import argparse
import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..config import (
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_OPENROUTER_TTS_VOICE,
    DEFAULT_POLZA_TTS_VOICE,
    DEFAULT_QWEN_VOICE,
    DEFAULT_VOICE,
    OMNIVOICE_LOCAL_MODEL_ID,
)
from ..gemini_dialogue import DIALOGUE_FORMAT, is_dialogue_format
from ..models import ScriptChunk
from ..omnivoice_voice_bank import VoiceBankCatalog
from ..tts_prompting import resolve_prompt_mode
from ..voiceover_script import VOICEOVER_FORMAT, detect_frontmatter_format


class PreparationError(ValueError):
    """Script or run-preparation argument failure the CLI reports as a usage error."""


LocalChunkPreparation = Callable[[Iterable[ScriptChunk], str, str | None], list[ScriptChunk]]


class SessionFragmentMerge(Protocol):
    """The OmniVoice session-merge seam the CLI forwards for patchability."""

    def __call__(
        self,
        fragments: Iterable[ScriptChunk],
        *,
        mode: str = ...,
        reference_audio_path: Path | str | None = ...,
        reference_text: str | None = ...,
        design_instruction: str | None = ...,
    ) -> list[ScriptChunk]: ...


@dataclass(frozen=True)
class PreparedScriptFragments:
    """Limited script fragments plus the fragment counts a dry run reports.

    ``chunks`` holds the exact objects the run hashes and stores. A provider/model
    spoken-text profile may replace them, but ``--limit-chunks`` only slices, so no
    fragment is renumbered, reordered, or rewritten while limiting.
    """

    chunks: list[ScriptChunk]
    original_count: int
    requested_count: int


@dataclass(frozen=True)
class PreparedPart:
    """One original script chunk plus the per-part voice identity for its call.

    The wrapped chunk is the exact object the run hashes and stores, so part
    identity (number, id, text, speaker, pause) never changes between
    preparation and synthesis. ``voice`` is the chunk's own voice from the
    dialogue cast; ``None`` means the run-level voice applies.
    """

    chunk: ScriptChunk
    voice: str | None


@dataclass(frozen=True)
class OmniVoiceBankProfileIdentity:
    """One referenced voice-bank profile's immutable, nonsecret settings.

    This is the profile's ``reference_text`` and reference locator/digest taken
    from the admitted catalog at preparation time, so a later resume can prove
    the exact clone reference it would submit instead of re-deriving it from a
    catalog that may have changed.
    """

    profile_id: str
    reference_audio: str
    reference_sha256: str
    reference_text: str
    language: str

    def to_payload(self) -> dict[str, str]:
        return {
            "profile_id": self.profile_id,
            "reference_audio": self.reference_audio,
            "reference_sha256": self.reference_sha256,
            "reference_text": self.reference_text,
            "language": self.language,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "OmniVoiceBankProfileIdentity":
        if not isinstance(payload, Mapping):
            raise ValueError("voice-bank profile identity must be a mapping")
        fields = {
            key: payload.get(key)
            for key in (
                "profile_id",
                "reference_audio",
                "reference_sha256",
                "reference_text",
                "language",
            )
        }
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise ValueError("voice-bank profile identity fields must be non-empty strings")
        return cls(**fields)  # type: ignore[arg-type]


@dataclass(frozen=True)
class OmniVoiceVoiceBankIdentity:
    """The admitted preset voice bank one OmniVoice dialogue run commits to.

    ``catalog_path`` locates the user's catalog for a later resume; ``profiles``
    holds only the cast profiles the run actually references, in stable profile
    id order, so an unrelated catalog edit does not change this identity.
    """

    catalog_path: str
    mode: str
    profiles: tuple[OmniVoiceBankProfileIdentity, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "catalog_path": self.catalog_path,
            "mode": self.mode,
            "profiles": [profile.to_payload() for profile in self.profiles],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "OmniVoiceVoiceBankIdentity":
        if not isinstance(payload, Mapping):
            raise ValueError("voice-bank identity must be a mapping")
        catalog_path = payload.get("catalog_path")
        mode = payload.get("mode")
        raw_profiles = payload.get("profiles")
        if not isinstance(catalog_path, str) or not catalog_path:
            raise ValueError("voice-bank identity catalog_path must be a non-empty string")
        if not isinstance(mode, str) or not mode:
            raise ValueError("voice-bank identity mode must be a non-empty string")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError("voice-bank identity profiles must be a non-empty list")
        profiles = tuple(OmniVoiceBankProfileIdentity.from_payload(item) for item in raw_profiles)
        seen: set[str] = set()
        for profile in profiles:
            if profile.profile_id in seen:
                raise ValueError("voice-bank identity profiles must be unique")
            seen.add(profile.profile_id)
        return cls(catalog_path=catalog_path, mode=mode, profiles=profiles)


@dataclass(frozen=True)
class QwenCloneVoiceIdentity:
    """The immutable clone inputs one local Qwen clone run commits to.

    The standard run identity already covers the run's ``provider``, ``voice``
    (``"clone"``), and ``model``; this block adds the clone-specific nonsecret
    inputs that also change the synthesized bytes: the canonical absolute sample
    locator and its byte digest/size (never the sample bytes themselves), the exact
    reference text (empty means the runtime's x-vector-only clone), and the runtime
    and language knobs. Storing them lets a later resume prove it would clone the
    exact same reference with the same runtime instead of guessing.
    """

    mode: str
    model: str
    sample_path: str
    sample_sha256: str
    sample_size: int
    sample_text: str
    runtime: str
    language: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "sample_path": self.sample_path,
            "sample_sha256": self.sample_sha256,
            "sample_size": self.sample_size,
            "sample_text": self.sample_text,
            "runtime": self.runtime,
            "language": self.language,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "QwenCloneVoiceIdentity":
        if not isinstance(payload, Mapping):
            raise ValueError("qwen clone identity must be a mapping")
        fields = {
            key: payload.get(key)
            for key in ("mode", "model", "sample_path", "sample_sha256", "runtime", "language")
        }
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise ValueError("qwen clone identity fields must be non-empty strings")
        sample_size = payload.get("sample_size")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
            raise ValueError("qwen clone identity sample_size must be a non-negative integer")
        sample_text = payload.get("sample_text")
        if not isinstance(sample_text, str):
            raise ValueError("qwen clone identity sample_text must be a string")
        return cls(**fields, sample_size=sample_size, sample_text=sample_text)  # type: ignore[arg-type]


QWEN_CLONE_MODE = "clone"


QWEN_PRESET_MODE = "preset"
QWEN_DESIGN_MODE = "design"
# The two instructed ``qwen-local`` modes whose identity is the run's model, its
# preset speaker (or the ``design`` marker), and the per-run instruction plus the
# selected runtime/language. ``auto`` is a CLI choice that resolves no model, so it
# never reaches a provider and is not part of this family.
QWEN_INSTRUCT_MODES = frozenset({QWEN_PRESET_MODE, QWEN_DESIGN_MODE})


@dataclass(frozen=True)
class QwenModeVoiceIdentity:
    """The immutable inputs one instructed local Qwen preset/design run commits to.

    The standard run identity already covers the run's ``provider``, ``voice`` (the
    preset speaker, or the ``design`` marker), ``model``, and ``style_prompt`` (the
    verbatim instruction). This block adds the remaining nonsecret knobs that also
    change the synthesized bytes -- the mode, the exact instruction, and the runtime
    and language -- and restates the effective mode/model/voice so a later resume
    can prove the whole identity from one committed block instead of re-deriving the
    mode from the model. A changed instruction is therefore refused before any local
    model call, not just after the run identity already differs.
    """

    mode: str
    model: str
    voice: str
    instruct: str
    runtime: str
    language: str

    def to_payload(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "model": self.model,
            "voice": self.voice,
            "instruct": self.instruct,
            "runtime": self.runtime,
            "language": self.language,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "QwenModeVoiceIdentity":
        if not isinstance(payload, Mapping):
            raise ValueError("qwen mode identity must be a mapping")
        fields = {
            key: payload.get(key)
            for key in ("mode", "model", "voice", "instruct", "runtime", "language")
        }
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise ValueError("qwen mode identity fields must be non-empty strings")
        return cls(**fields)  # type: ignore[arg-type]


def build_qwen_mode_identity(
    *, mode: str, model: str, voice: str, instruct: str | None, runtime: str, language: str
) -> QwenModeVoiceIdentity:
    """Capture one instructed ``qwen-local`` preset/design run's immutable settings.

    The instruction is the exact text the runtime speaks with, so it must be
    non-empty; an empty or whitespace instruction is refused here as a usage error
    before any provider or snapshot exists, mirroring the runtime's own
    ``VoiceDesign`` requirement and the snapshot's style-prompt rule.
    """
    if mode not in QWEN_INSTRUCT_MODES:
        raise PreparationError("qwen-local instructed mode must be 'preset' or 'design'.")
    if not isinstance(instruct, str) or not instruct.strip():
        raise PreparationError("qwen-local preset/design mode requires a non-empty instruction.")
    return QwenModeVoiceIdentity(
        mode=mode,
        model=model,
        voice=voice,
        instruct=instruct,
        runtime=runtime,
        language=language,
    )


def build_qwen_clone_identity(
    *, model: str, sample_path: str, sample_text: str, runtime: str, language: str
) -> QwenCloneVoiceIdentity:
    """Capture one ``qwen-local`` clone run's immutable reference identity.

    The sample file is read once, here, so its bytes are hashed without ever being
    copied into the snapshot. A missing or unreadable reference raises
    :class:`PreparationError` before any provider or model exists, so a native run
    can never snapshot an identity whose bytes it does not have.
    """
    path = Path(sample_path).expanduser()
    try:
        resolved = path.resolve()
        data = resolved.read_bytes()
    except OSError as exc:
        raise PreparationError(
            "qwen-local clone mode requires a readable --sample reference audio file."
        ) from exc
    return QwenCloneVoiceIdentity(
        mode=QWEN_CLONE_MODE,
        model=model,
        sample_path=str(resolved),
        sample_sha256=hashlib.sha256(data).hexdigest(),
        sample_size=len(data),
        sample_text=sample_text,
        runtime=runtime,
        language=language,
    )


@dataclass(frozen=True)
class PreparedRun:
    """Run-scoped provider identity plus the ordered parts to synthesize.

    ``voice`` is the run-level voice the CLI already resolved; a part without
    its own cast voice falls back to it. ``voice_bank_identity`` is present only
    for the admitted ``omnivoice-local`` preset dialogue route, and
    ``qwen_clone_identity``/``qwen_mode_identity`` only for the admitted
    ``qwen-local`` local routes (clone, or the instructed preset/design modes).
    """

    provider: str
    model: str
    voice: str
    style_prompt: str | None
    prompt_mode: str
    parts: tuple[PreparedPart, ...]
    voice_bank_identity: OmniVoiceVoiceBankIdentity | None = None
    qwen_clone_identity: QwenCloneVoiceIdentity | None = None
    qwen_mode_identity: QwenModeVoiceIdentity | None = None


def default_voice(args: argparse.Namespace) -> str | None:
    """Return the provider/model default voice when the run specifies none.

    ``omnivoice-local`` has no single default voice: its preset catalog or an
    explicit mode selects one, so this returns ``None``.
    """
    if args.provider == "polza-tts":
        if args.model and args.model.startswith("elevenlabs/"):
            return DEFAULT_ELEVENLABS_VOICE
        return DEFAULT_POLZA_TTS_VOICE
    if args.provider == "openrouter-tts":
        if args.model and args.model.startswith("openai/"):
            return DEFAULT_OPENAI_TTS_VOICE
        return DEFAULT_OPENROUTER_TTS_VOICE
    if args.provider == "qwen-local":
        return DEFAULT_QWEN_VOICE
    if args.provider == "omnivoice-local":
        return None
    return DEFAULT_VOICE


def resolve_script_format(script_path: Path, requested_format: str) -> str:
    """Resolve the run's script format from frontmatter and the requested flag.

    A ``markdown`` request is upgraded by voiceover/dialogue frontmatter, an
    explicit dialogue alias collapses to ``dialogue``, and an unset format
    falls back to ``markdown``.
    """
    detected_format = detect_frontmatter_format(script_path)
    script_format = requested_format
    if (
        detected_format is not None
        and script_format == "markdown"
        and (detected_format == VOICEOVER_FORMAT or is_dialogue_format(detected_format))
    ):
        script_format = detected_format
    if script_format is None:
        return "markdown"
    return DIALOGUE_FORMAT if is_dialogue_format(script_format) else script_format


def bind_omnivoice_dialogue_fingerprints(
    chunks: list[ScriptChunk], catalog: VoiceBankCatalog
) -> list[ScriptChunk]:
    """Bind each dialogue turn to its voice-bank profile's reference fingerprint.

    A turn without a cast voice or with a profile id absent from the admitted
    catalog raises ``PreparationError`` before provider work in CLI generation,
    preventing submission of a turn with an unknown clone identity.
    """
    profiles = {profile.id: profile for profile in catalog.profiles}
    bound: list[ScriptChunk] = []
    for chunk in chunks:
        if chunk.voice is None:
            raise PreparationError("OmniVoice dialogue turn is missing a voice-bank profile")
        profile = profiles.get(chunk.voice)
        if profile is None:
            raise PreparationError(f"voice '{chunk.voice}' not found in the voice bank")
        bound.append(replace(chunk, voice_fingerprint=profile.reference_sha256))
    return bound


def build_omnivoice_voice_bank_identity(
    catalog: VoiceBankCatalog, catalog_path: Path, profile_ids: Iterable[str]
) -> OmniVoiceVoiceBankIdentity:
    """Bind the cast profile ids of one dialogue run to their bank settings.

    The referenced profiles are resolved from the admitted catalog and stored in
    stable id order, so the run identity never depends on a catalog's own order
    or on profiles the run does not use. An unknown cast voice raises
    ``PreparationError`` before any provider work.
    """
    by_id = {profile.id: profile for profile in catalog.profiles}
    selected: list[OmniVoiceBankProfileIdentity] = []
    for voice_id in sorted(set(profile_ids)):
        profile = by_id.get(voice_id)
        if profile is None:
            raise PreparationError(f"voice '{voice_id}' not found in the voice bank")
        selected.append(
            OmniVoiceBankProfileIdentity(
                profile_id=profile.id,
                reference_audio=profile.reference_audio,
                reference_sha256=profile.reference_sha256,
                reference_text=profile.reference_text,
                language=profile.language,
            )
        )
    if not selected:
        raise PreparationError("OmniVoice dialogue has no cast voice-bank profile")
    return OmniVoiceVoiceBankIdentity(
        catalog_path=str(catalog_path), mode="preset", profiles=tuple(selected)
    )


@dataclass(frozen=True)
class PreparedGenerationIdentity:
    """Run-scoped cast identity plus the resolved style prompt and prompt mode.

    ``chunks`` are the exact objects the run hashes and stores: the CLI's
    dialogue cast may bind each chunk's OmniVoice fingerprint here, but no
    number, id, text, speaker, or pause changes.
    """

    chunks: list[ScriptChunk]
    style_prompt: str | None
    prompt_mode: str


def prepare_generation_identity(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    gemini_report: dict[str, Any] | None,
    *,
    resolve_style_prompt: Callable[[argparse.Namespace], str | None],
) -> PreparedGenerationIdentity:
    """Resolve the run's cast voice, style prompt, and prompt mode for one generation.

    The steps keep their exact order from the CLI: the cast voice and per-chunk
    OmniVoice fingerprints bind first, then ``resolve_style_prompt`` runs, then a
    dialogue report may supply the style prompt, then the prompt mode is resolved.
    ``resolve_style_prompt`` is injected so the service keeps no ``cli`` import and
    the CLI still owns reading/validating the provider's style input.
    """
    requested_voice = args.voice
    if gemini_report:
        args.speaker_voice_map = gemini_report["speaker_voice_map"]
        args.voice = requested_voice or next(iter(args.speaker_voice_map.values()))
        if args.provider == "omnivoice-local":
            chunks = bind_omnivoice_dialogue_fingerprints(chunks, args.voice_bank_catalog)
    else:
        args.speaker_voice_map = {}
        args.voice = requested_voice or default_voice(args)

    style_prompt = resolve_style_prompt(args)
    if (
        gemini_report
        and args.provider != "openrouter-tts"
        and not args.no_style_prompt
        and args.style_prompt is None
        and args.style_prompt_file is None
    ):
        style_prompt = gemini_report["style_prompt"]
    prompt_mode = resolve_prompt_mode(args.provider, args.model)
    return PreparedGenerationIdentity(
        chunks=chunks,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
    )


def prepare_run(
    args: argparse.Namespace,
    chunks: Sequence[ScriptChunk],
    style_prompt: str | None,
    prompt_mode: str,
    *,
    qwen_clone_identity: QwenCloneVoiceIdentity | None = None,
    qwen_mode_identity: QwenModeVoiceIdentity | None = None,
) -> PreparedRun:
    """Wrap the already-resolved run identity and chunks without changing them.

    ``qwen_clone_identity`` is supplied only by the admitted local Qwen clone
    route, which has already read and hashed its reference sample; the instructed
    local Qwen preset/design routes supply ``qwen_mode_identity`` instead. Every
    other caller leaves both ``None``.
    """
    voice_bank_identity: OmniVoiceVoiceBankIdentity | None = None
    if args.provider == "omnivoice-local" and is_dialogue_format(
        getattr(args, "format", "markdown")
    ):
        catalog = getattr(args, "voice_bank_catalog", None)
        bank_arg = getattr(args, "voice_bank", None)
        if not isinstance(catalog, VoiceBankCatalog) or bank_arg is None:
            raise PreparationError(
                "omnivoice-local dialogue requires an admitted --voice-bank catalog"
            )
        voice_bank_identity = build_omnivoice_voice_bank_identity(
            catalog,
            Path(bank_arg).expanduser().resolve(),
            [chunk.voice for chunk in chunks if chunk.voice],
        )
    return PreparedRun(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
        parts=tuple(PreparedPart(chunk=chunk, voice=chunk.voice) for chunk in chunks),
        voice_bank_identity=voice_bank_identity,
        qwen_clone_identity=qwen_clone_identity,
        qwen_mode_identity=qwen_mode_identity,
    )


def prepare_script_fragments(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    script_format: str,
    *,
    local_prepare: LocalChunkPreparation,
) -> PreparedScriptFragments:
    """Normalize and limit the script fragments a run will synthesize.

    Non-dialogue scripts first pass through the caller's provider/model spoken-text
    profile, so ``--limit-chunks`` always slices already-prepared fragments. Dialogue
    chunks are one turn each and bypass that step. Empty scripts and non-positive
    limits raise ``PreparationError`` before any provider, key, or pricing work.
    """
    prepared = chunks
    if not is_dialogue_format(script_format):
        prepared = local_prepare(prepared, args.provider, args.model)
    if not prepared:
        raise PreparationError("Script produced no chunks. Check delimiter and content.")
    original_count = len(prepared)
    if args.limit_chunks is not None:
        if args.limit_chunks <= 0:
            raise PreparationError("--limit-chunks must be greater than zero")
        prepared = prepared[: args.limit_chunks]
    return PreparedScriptFragments(
        chunks=prepared,
        original_count=original_count,
        requested_count=len(prepared),
    )


def prepare_runtime_chunks(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    script_format: str,
    *,
    merge_session_fragments: SessionFragmentMerge,
) -> list[ScriptChunk]:
    """Fold prepared non-dialogue fragments into one OmniVoice request when eligible.

    A preset run with an admitted voice-bank profile always clones that profile's
    resolved reference; otherwise the explicit mode/reference/design arguments apply
    unchanged. Dialogue turns and every other provider/model keep their fragments.
    """
    if (
        args.provider == "omnivoice-local"
        and args.model == OMNIVOICE_LOCAL_MODEL_ID
        and not is_dialogue_format(script_format)
    ):
        bank_profile = getattr(args, "voice_bank_profile", None)
        bank_catalog = getattr(args, "voice_bank_catalog", None)
        if getattr(args, "mode", "preset") == "preset" and bank_profile is not None:
            reference_audio_path = (
                str(bank_catalog.root / bank_profile.reference_audio)
                if bank_catalog is not None
                else str(Path(bank_profile.reference_audio))
            )
            return merge_session_fragments(
                chunks,
                mode="clone",
                reference_audio_path=reference_audio_path,
                reference_text=bank_profile.reference_text,
            )
        return merge_session_fragments(
            chunks,
            mode=getattr(args, "mode", "preset"),
            reference_audio_path=getattr(args, "reference_audio", None),
            reference_text=getattr(args, "reference_text", None),
            design_instruction=getattr(args, "design_instruction", None),
        )
    return chunks
