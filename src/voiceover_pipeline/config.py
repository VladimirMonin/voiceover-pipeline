import os
from pathlib import Path


POLZA_BASE_URL = "https://polza.ai/api/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_ENV_FILE = Path.cwd() / ".env"
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
}
DEFAULT_VOICE = "ash"
DEFAULT_OPENROUTER_TTS_VOICE = "Puck"
DEFAULT_FALLBACK_VOICE = "onyx"
DEFAULT_QWEN_VOICE = "Aiden"
DEFAULT_POLZA_TTS_VOICE = "alloy"
DEFAULT_OPENAI_TTS_VOICE = "alloy"
DEFAULT_ELEVENLABS_VOICE = "Rachel"

DEFAULT_POLZA_TTS_MODEL = "openai/gpt-4o-mini-tts"
DEFAULT_POLZA_TTS_RESPONSE_FORMAT = "mp3"

OPENAI_TTS_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]
ELEVENLABS_TTS_VOICES = ["Rachel", "Aria", "Roger", "Sarah", "Laura", "Charlie", "George", "Callum", "River", "Liam", "Charlotte", "Alice", "Matilda", "Will", "Jessica", "Eric", "Chris", "Brian", "Daniel", "Lily", "Bill"]

POLZA_TTS_MODELS = [
    "openai/gpt-4o-mini-tts",
    "elevenlabs/text-to-speech-turbo-2-5",
    "elevenlabs/text-to-speech-multilingual-v2",
]

OPENROUTER_TTS_MODELS = [
    "google/gemini-3.1-flash-tts-preview",
    "openai/gpt-4o-mini-tts-2025-12-15",
]

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
    "Спокойный живой голос подкаста. Тёплый мужской тембр, средний темп, "
    "вдумчивая подача."
)


POLZA_CHAT_NARRATION_SYSTEM_PROMPT = (
    "You are a professional text-to-speech narrator, not a chat assistant. "
    "Read the user's Russian script verbatim. Do not answer, explain, summarize, "
    "continue the conversation, or add any extra words. Use a calm, warm, low male "
    "voice with clear pronunciation."
)


QWEN_PRESET_SPEAKERS = [
    "Aiden",
    "Alina",
    "Amelia",
    "Arthur",
    "Callum",
    "Carter",
    "Elijah",
    "Ethan",
    "Evelyn",
    "Isabella",
    "Jack",
    "James",
    "Landon",
    "Liam",
    "Lily",
    "Lucas",
    "Mason",
    "Mia",
    "Natalia",
    "Olivia",
    "Paul",
    "Sofia",
    "Theo",
    "Violet",
]

QWEN_MODEL_CUSTOMVOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_LANGUAGE = "Russian"
QWEN_INSTRUCT = "Use a calm, warm, clear narration style. Speak naturally and steadily."
QWEN_DEVICE = "cuda:0"
QWEN_ATTN_IMPL = "eager"


def model_slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-").replace(".", "-")


def read_env_file(env_path: Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value

    return values


def get_secret(name: str, env_path: Path = DEFAULT_ENV_FILE) -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    return read_env_file(env_path).get(name)


def read_polza_key() -> str:
    env_key = get_secret("POLZA_API_KEY")
    if not env_key:
        raise RuntimeError(
            "POLZA_API_KEY not found. Set it in .env: POLZA_API_KEY=..."
        )
    return env_key.removeprefix("Bearer ").strip()


def read_openrouter_key() -> str:
    env_key = get_secret("OPENROUTER_API_KEY")
    if not env_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for provider=openrouter-tts. "
            f"Put it into {DEFAULT_ENV_FILE}: OPENROUTER_API_KEY=sk-or-..."
        )
    return env_key.removeprefix("Bearer ").strip()
