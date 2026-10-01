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

POLZA_TTS_MODELS = [
    "openai/gpt-4o-mini-tts",
    "elevenlabs/text-to-speech-turbo-2-5",
    "elevenlabs/text-to-speech-multilingual-v2",
]

# The one experimental Polza Gemini TTS model whose scalar per-part request was
# observed but never documented. It is deliberately absent from
# ``POLZA_TTS_MODELS`` and from ``list`` so nothing advertises it as a stable
# route; only the ordinary CLI's explicit opt-in flag admits it, and only for the
# ``speech-parts`` format. ``instructions`` is an undocumented field for this
# model, so the provider sends it only behind that same opt-in.
POLZA_EXPERIMENTAL_GEMINI_SPEECH_PARTS_MODEL = "google/gemini-3.8-flash-tts"

TTS_PROMPT_MODE_NONE = "none"
TTS_PROMPT_MODE_PREFIX = "prefix"
TTS_PROMPT_MODE_NATIVE = "native"

PROMPTABLE_TTS_MODELS: dict[str, str] = {
    "google/gemini-3.1-flash-tts-preview": "none",
}

POLZA_PROMPTABLE_TTS_MODELS: dict[str, str] = {}

OPENROUTER_TTS_MODELS = [
    "google/gemini-3.1-flash-tts-preview",
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
# ``--env-file`` scoped to the current CLI call; then ``<call-time CWD>/.env`` for
# compatibility. No hidden parent-directory search runs and no import-time path is
# captured for a runtime lookup.
_ACTIVE_ENV_FILE: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "voiceover_active_env_file", default=None
)


class EnvFileError(RuntimeError):
    """A fixed, redacted failure for an unusable runtime env file.

    The message never carries the file path, its contents, or a secret value.
    """


def resolved_env_file_path() -> Path:
    """The file a runtime secret lookup would read.

    Returns the explicit ``--env-file`` when one is scoped to this call, otherwise
    ``<call-time CWD>/.env``. It never reads the file, so a read-only command can
    report the path without touching a secret.
    """
    active = _ACTIVE_ENV_FILE.get()
    if active is not None:
        return active
    return Path.cwd() / ".env"


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


def _require_regular_explicit_env_file(env_path: Path) -> None:
    """Reject an unusable explicit path without opening or parsing its contents."""
    try:
        regular_file = env_path.is_file()
    except OSError as exc:
        raise EnvFileError("Explicit --env-file could not be read.") from exc
    if not regular_file:
        raise EnvFileError("Explicit --env-file is missing or not a regular file.")


def _read_env_values(env_path: Path, *, required: bool) -> dict[str, str]:
    if required:
        _require_regular_explicit_env_file(env_path)
    try:
        if not env_path.exists():
            return {}
        text = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        message = (
            "Explicit --env-file could not be read." if required else "Env file could not be read."
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


def read_env_file(env_path: Path | None = None) -> dict[str, str]:
    """Read a ``.env`` file under the runtime resolution rules.

    An explicit ``env_path`` (direct-call compatibility) reads exactly that file and
    returns an empty mapping when it does not exist. With no argument, an explicit
    ``--env-file`` scoped through :func:`use_env_file` is read next and fails closed
    when it is missing or unreadable; otherwise ``<call-time CWD>/.env`` is read for
    compatibility. Parents are never searched.
    """
    if env_path is not None:
        return _read_env_values(env_path, required=False)

    active = _ACTIVE_ENV_FILE.get()
    if active is not None:
        return _read_env_values(active, required=True)

    return _read_env_values(Path.cwd() / ".env", required=False)


def get_secret(name: str, env_path: Path | None = None) -> str | None:
    """Resolve a secret: validate an explicit CLI path, then prefer the process value.

    A supplied --env-file must exist and be regular even when the process has a
    usable key. Validation checks metadata only; contents are not read when the
    process value wins. Direct-call env_path retains its historical behavior.
    """
    if env_path is None:
        active = _ACTIVE_ENV_FILE.get()
        if active is not None:
            _require_regular_explicit_env_file(active)
    value = os.environ.get(name)
    if value:
        return value

    return read_env_file(env_path).get(name)


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
