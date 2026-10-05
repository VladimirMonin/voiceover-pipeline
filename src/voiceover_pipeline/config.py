import contextvars
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

POLZA_BASE_URL = "https://polza.ai/api/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
XAI_BASE_URL = "https://api.x.ai/v1"

DEFAULT_SCRIPT_DIR = Path.cwd() / "in"
DEFAULT_OUTPUT_DIR = Path.cwd() / "out"
DEFAULT_TEMP_DIR = Path.cwd() / "temp"
DEFAULT_LOG_FILE = Path.cwd() / "podcast_generation.log"

DEFAULT_MODEL = "openai/gpt-audio-mini"
DEFAULT_PROVIDER = "polza-chat-audio"
PROVIDER_DEFAULT_MODELS = {
    "polza-chat-audio": "openai/gpt-audio-mini",
    "polza-tts": "openai/gpt-4o-mini-tts",
    "openrouter-tts": "google/gemini-3.1-flash-tts-preview",
    "qwen-local": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "omnivoice-local": "audio-cpp/omnivoice-q8_0",
}
DEFAULT_VOICE = "ash"
DEFAULT_OPENROUTER_TTS_VOICE = "Puck"
DEFAULT_FALLBACK_VOICE = "onyx"
DEFAULT_QWEN_VOICE = "Aiden"
DEFAULT_OMNIVOICE_VOICE = "built-in-female-style-condition"
DEFAULT_POLZA_TTS_VOICE = "alloy"
DEFAULT_OPENAI_TTS_VOICE = "alloy"
DEFAULT_ELEVENLABS_VOICE = "Rachel"

DEFAULT_POLZA_TTS_MODEL = "openai/gpt-4o-mini-tts"
DEFAULT_POLZA_TTS_RESPONSE_FORMAT = "mp3"
DEFAULT_OPENROUTER_TTS_RESPONSE_FORMAT = "pcm"

OPENAI_TTS_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
]
ELEVENLABS_TTS_VOICES = [
    "Rachel",
    "Aria",
    "Roger",
    "Sarah",
    "Laura",
    "Charlie",
    "George",
    "Callum",
    "River",
    "Liam",
    "Charlotte",
    "Alice",
    "Matilda",
    "Will",
    "Jessica",
    "Eric",
    "Chris",
    "Brian",
    "Daniel",
    "Lily",
    "Bill",
]

# The ordinary Gemini 3.8 speech models. Both are listed for both cloud speech
# providers, and each request carries exactly one scalar ``voice`` with the part's
# effective direction in a separate ``instructions`` field, so no experimental
# opt-in is required and no request ever implies more than one speaker.
GEMINI_38_TTS_MODELS = [
    "google/gemini-3.8-flash-tts",
    "google/gemini-3.8-flash-lite-tts",
]

POLZA_TTS_MODELS = [
    "openai/gpt-4o-mini-tts",
    "elevenlabs/text-to-speech-turbo-2-5",
    "elevenlabs/text-to-speech-multilingual-v2",
    *GEMINI_38_TTS_MODELS,
]

TTS_PROMPT_MODE_NONE = "none"
TTS_PROMPT_MODE_PREFIX = "prefix"
TTS_PROMPT_MODE_NATIVE = "native"

PROMPTABLE_TTS_MODELS: dict[str, str] = {
    "google/gemini-3.1-flash-tts-preview": "none",
}

POLZA_PROMPTABLE_TTS_MODELS: dict[str, str] = {}

OPENROUTER_TTS_MODELS = [
    "google/gemini-3.1-flash-tts-preview",
    *GEMINI_38_TTS_MODELS,
]

OPENROUTER_WHISPER_MODELS = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-1",
]

GROQ_WHISPER_MODELS = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]
DEFAULT_TIMING_PROVIDER = "faster-whisper"

GEMINI_TTS_VOICES = [
    "Puck",
    "Charon",
    "Fenrir",
    "Orus",
    "Aoede",
    "Kore",
    "Zephyr",
    "Leda",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]

DEFAULT_TIMING_MODEL = "small"
DEFAULT_TIMING_DEVICE = "cpu"
DEFAULT_TIMING_COMPUTE = "int8"
DEFAULT_TIMING_LANGUAGE = "ru"
DEFAULT_ASR_DEVICE = "cpu"
DEFAULT_ASR_COMPUTE = "auto"

WHISPER_HF_REPOS: dict[str, str] = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
}

SAMPLE_RATE = 24000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
MP3_BITRATE = "128k"
OUTPUT_MP3_BITRATE_QWEN = "64k"


PODCAST_NARRATION_PROMPT = (
    "Голос технического подкаста: спокойный, вдумчивый, живой и уверенный. "
    "Тёплый мужской тембр, средний темп, ясная артикуляция, без театральности."
)

PODCAST_NARRATION_FALLBACK_PROMPT = (
    "Спокойный живой голос подкаста. Тёплый мужской тембр, средний темп, вдумчивая подача."
)


POLZA_CHAT_NARRATION_SYSTEM_PROMPT = (
    "You are a professional text-to-speech narrator, not a chat assistant. "
    "Read the user's Russian script verbatim. Do not answer, explain, summarize, "
    "continue the conversation, or add any extra words. Use a calm, warm, low male "
    "voice with clear pronunciation."
)


QWEN_PRESET_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

QWEN_MODEL_CUSTOMVOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_MODEL_BASE = os.environ.get("VOICEOVER_QWEN_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
QWEN_MODEL_VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_LANGUAGE = "Russian"
QWEN_INSTRUCT = "Use a calm, warm, clear narration style. Speak naturally and steadily."
QWEN_DEVICE = "cuda:0"
QWEN_ATTN_IMPL = "eager"

OMNIVOICE_LOCAL_MODEL_ID = "audio-cpp/omnivoice-q8_0"
OMNIVOICE_DEFAULT_LANGUAGE = "ru"
OMNIVOICE_DEFAULT_SEED = 1234
OMNIVOICE_DEFAULT_STEPS = 32
OMNIVOICE_DEFAULT_GUIDANCE_SCALE = 2.0
OMNIVOICE_STYLE_CONDITION = "female"
OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE = 420


def model_slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-").replace(".", "-")


# Runtime ``.env`` resolution. The process environment always wins; then an explicit
# ``--env-file`` scoped to the current CLI call; then the optional path-only provider
# source configured for that secret; then ``<call-time CWD>/.env`` for compatibility.
# No hidden parent-directory search runs, nothing is copied or merged, and no
# import-time path is captured for a runtime lookup.
#
# ``VOICEOVER_POLZA_ENV_FILE``/``VOICEOVER_OPENROUTER_ENV_FILE`` are optional
# *path-only* defaults: each holds a path to an external file the operator already
# maintains, so a provider source outside the working directory can be selected
# without copying a key into this project.
PROVIDER_ENV_FILE_VARIABLES: dict[str, str] = {
    "POLZA_API_KEY": "VOICEOVER_POLZA_ENV_FILE",
    "OPENROUTER_API_KEY": "VOICEOVER_OPENROUTER_ENV_FILE",
}

# The two required sources are named in their fixed messages by label only, so a
# diagnostic says which source failed without echoing a path or a value.
_EXPLICIT_ENV_FILE_LABEL = "Explicit --env-file"
_CONFIGURED_ENV_FILE_LABEL = "The configured provider env file"
_ACTIVE_ENV_FILE: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "voiceover_active_env_file", default=None
)


class EnvFileError(RuntimeError):
    """A fixed, redacted failure for an unusable runtime env file.

    The message never carries the file path, its contents, or a secret value.
    """


def resolved_env_file_path(secret_name: str | None = None) -> Path:
    """The file a runtime secret lookup would read.

    Returns the explicit ``--env-file`` when one is scoped to this call. Otherwise a
    ``secret_name`` whose provider configured a path-only source
    (:data:`PROVIDER_ENV_FILE_VARIABLES`) resolves to that configured path, which
    replaces the working-directory file instead of merging with it; without one,
    ``<call-time CWD>/.env`` remains for compatibility. It never reads the file, so a
    read-only command such as ``doctor`` or ``help`` can report or ignore the path
    without touching a secret.
    """
    active = _ACTIVE_ENV_FILE.get()
    if active is not None:
        return active
    if secret_name is not None:
        configured = _configured_provider_env_file(secret_name)
        if configured is not None:
            return configured
    return Path.cwd() / ".env"


def _configured_provider_env_file(secret_name: str) -> Path | None:
    """The path-only source configured for this secret, or ``None``.

    The variable holds a path and nothing else. A blank value means "not
    configured", so an accidental empty override keeps the call-time
    working-directory compatibility instead of failing every paid run. The path is
    used exactly as given -- no parent search, no expansion, and no merge with any
    other file.
    """
    variable = PROVIDER_ENV_FILE_VARIABLES.get(secret_name)
    if variable is None:
        return None
    value = os.environ.get(variable)
    if value is None or not value.strip():
        return None
    return Path(value.strip())


@contextmanager
def use_env_file(env_path: Path | None) -> Iterator[None]:
    """Scope an explicit ``--env-file`` to the current call.

    Setting the override never reads the file, so ``help`` and other read-only
    commands keep working even when the flag points at a missing file. The token is
    reset on every exit -- including ``SystemExit`` and raised errors -- so a later
    in-process call falls back to ``<call-time CWD>/.env``.
    """
    token = _ACTIVE_ENV_FILE.set(env_path)
    try:
        yield
    finally:
        _ACTIVE_ENV_FILE.reset(token)


def _require_regular_env_file(env_path: Path, *, label: str) -> None:
    """Reject an unusable required source without opening or parsing its contents.

    ``label`` names the caller-supplied or configured source in the fixed, path-free
    message, so a diagnostic says which source failed without echoing a path.
    """
    try:
        regular_file = env_path.is_file()
    except OSError as exc:
        raise EnvFileError(f"{label} could not be read.") from exc
    if not regular_file:
        raise EnvFileError(f"{label} is missing or not a regular file.")


def _require_regular_explicit_env_file(env_path: Path) -> None:
    """Reject an unusable explicit ``--env-file`` without opening its contents."""
    _require_regular_env_file(env_path, label=_EXPLICIT_ENV_FILE_LABEL)


def _read_env_values(env_path: Path, *, required: bool) -> dict[str, str]:
    if required:
        _require_regular_explicit_env_file(env_path)
    try:
        if not env_path.exists():
            return {}
        text = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        message = (
            f"{_EXPLICIT_ENV_FILE_LABEL} could not be read."
            if required
            else "Env file could not be read."
        )
        raise EnvFileError(message) from exc

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value

    return values


def _read_configured_env_values(env_path: Path) -> dict[str, str]:
    """Read a required provider-configured path-only source.

    The file is validated by metadata first and every failure is reported under its
    own path-free label, so a missing or unreadable provider file fails closed
    instead of silently falling back to another file. A present file is then read with
    the same parser as every other source.
    """
    _require_regular_env_file(env_path, label=_CONFIGURED_ENV_FILE_LABEL)
    try:
        return _read_env_values(env_path, required=False)
    except EnvFileError:
        raise EnvFileError(f"{_CONFIGURED_ENV_FILE_LABEL} could not be read.") from None


def read_env_file(env_path: Path | None = None) -> dict[str, str]:
    """Read a ``.env`` file under the runtime resolution rules.

    An explicit ``env_path`` (direct-call compatibility) reads exactly that file and
    returns an empty mapping when it does not exist. With no argument, an explicit
    ``--env-file`` scoped through :func:`use_env_file` is read next and fails closed
    when it is missing or unreadable; otherwise ``<call-time CWD>/.env`` is read for
    compatibility. Parents are never searched. This reader has no secret identity, so
    it never substitutes a provider-configured source; :func:`get_secret` does that
    for a named secret.
    """
    if env_path is not None:
        return _read_env_values(env_path, required=False)

    active = _ACTIVE_ENV_FILE.get()
    if active is not None:
        return _read_env_values(active, required=True)

    return _read_env_values(Path.cwd() / ".env", required=False)


def get_secret(name: str, env_path: Path | None = None) -> str | None:
    """Resolve a secret under the one documented credential precedence.

    1. A non-empty process value wins, and the file is not read at all.
    2. A supplied ``--env-file`` must exist and be regular even when the process has
       a usable key, and an unusable explicit path fails closed. A direct-call
       ``env_path`` keeps its historical lenient behavior and wins over a configured
       provider source.
    3. Without either, a provider-configured path-only source
       (:data:`PROVIDER_ENV_FILE_VARIABLES`) is used instead of the call-time
       working-directory file; a missing or unreadable configured source fails
       closed with one path-free message.
    4. Without a configured source, ``<call-time CWD>/.env`` is read for
       compatibility.

    Validation checks metadata only, and no parent directory is ever searched or
    merged.
    """
    if env_path is None:
        active = _ACTIVE_ENV_FILE.get()
        if active is not None:
            _require_regular_explicit_env_file(active)
    value = os.environ.get(name)
    if value:
        return value

    if env_path is not None:
        return _read_env_values(env_path, required=False).get(name)

    active = _ACTIVE_ENV_FILE.get()
    if active is not None:
        return _read_env_values(active, required=True).get(name)

    configured = _configured_provider_env_file(name)
    if configured is not None:
        return _read_configured_env_values(configured).get(name)

    return _read_env_values(Path.cwd() / ".env", required=False).get(name)


def read_polza_key() -> str:
    env_key = get_secret("POLZA_API_KEY")
    if not env_key:
        raise RuntimeError("POLZA_API_KEY not found. Set it in .env: POLZA_API_KEY=...")
    return env_key.removeprefix("Bearer ").strip()


def read_openrouter_key() -> str:
    env_key = get_secret("OPENROUTER_API_KEY")
    if not env_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for provider=openrouter-tts. "
            "Set it in the environment or an env file: OPENROUTER_API_KEY=sk-or-..."
        )
    return env_key.removeprefix("Bearer ").strip()


def read_groq_key() -> str:
    env_key = get_secret("GROQ_API_KEY")
    if not env_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for timing-provider=groq-whisper. "
            "Set it in the environment or an env file: GROQ_API_KEY=gsk_..."
        )
    return env_key.removeprefix("Bearer ").strip()


def read_xai_key() -> str:
    env_key = get_secret("X_AI_API_KEY")
    if not env_key:
        raise RuntimeError(
            "X_AI_API_KEY is required for timing-provider=xai-stt. "
            "Set it in the environment or an env file: X_AI_API_KEY=xai-..."
        )
    return env_key.removeprefix("Bearer ").strip()
