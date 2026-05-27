# Cloud Transcription Providers — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Refactor the monolithic local-only Whisper timing system into an extensible provider architecture (like TTS providers), then add OpenRouter Whisper API as the first cloud provider.

**Architecture:** Abstract base class `TranscriptionProvider` with a `transcribe(audio_path) → TranscriptionResult` interface. Existing `whisper_timing.py` becomes `FasterWhisperProvider` (local). New `OpenRouterWhisperProvider` calls OpenRouter API using the existing `OPENROUTER_API_KEY`. CLI gets `--timing-provider` flag alongside `--timing-model`. `list timing-providers` shows available providers with their models.

**Tech Stack:** Python 3.11+, `requests` (already a dep), OpenRouter API (`/api/v1/audio/transcriptions`), `faster-whisper` (existing optional extra).

---

## Current State

```
whisper_timing.py          → monolithic, only faster-whisper
models.py / TimingResult   → has "backend" (hardcoded), "device", "compute_type" — local-only fields
cli.py / timings           → --model, --device, --compute — local-specific flags
cli.py / list timing-models → local models only
cli.py / doctor            → checks faster_whisper import only
config.py                  → DEFAULT_TIMING_MODEL = "small", device/compute/constants
providers/                 → TTS only, no transcription providers
```

## Target State

```
providers/
├── base.py                    # TTSProvider (unchanged) + NEW TranscriptionProvider ABC
├── faster_whisper.py          # NEW — extracted from whisper_timing.py
├── openrouter_whisper.py      # NEW — OpenRouter Whisper API
├── openrouter_tts.py          # unchanged
├── polza_*.py                 # unchanged
├── qwen_local.py              # unchanged
└── __init__.py                # + transcription exports

whisper_timing.py              # DELETED (moved to providers/faster_whisper.py)
models.py                      # TimingResult gains optional provider field
cli.py                         # --timing-provider, --timing-model (per-provider)
config.py                      # Cloud model constants, DEFAULT_TIMING_PROVIDER
```

---

### Task 1: Define TranscriptionProvider ABC and TranscriptionResult model

**Objective:** Create the provider interface that both local and cloud backends implement.

**Files:**
- Modify: `src/voiceover_pipeline/models.py` — add optional `provider: str | None` to `TimingResult`
- Modify: `src/voiceover_pipeline/providers/base.py` — add `TranscriptionProvider` ABC

**Step 1: Update TimingResult model**

In `src/voiceover_pipeline/models.py`, add `provider` field after `backend`:

```python
@dataclass(frozen=True)
class TimingResult:
    segments: list[TimingSegment]
    model: str
    backend: str
    provider: str | None = None        # NEW — "faster-whisper", "openrouter-whisper", etc.
    device: str = ""                    # changed from required to optional default
    compute_type: str = ""             # changed from required to optional default
    language: str = ""
    source_audio: str = ""
```

**Step 2: Add TranscriptionProvider ABC**

In `src/voiceover_pipeline/providers/base.py`, add after `TTSProvider`:

```python
class TranscriptionProvider(ABC):
    provider_id: str

    @abstractmethod
    def transcribe(
        self, audio_path: Path | str, language: str = "ru",
        word_timestamps: bool = False, quiet: bool = False,
    ) -> TimingResult:
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """Return [{"id": "model-id", ...metadata...}]."""
        raise NotImplementedError
```

**Step 3: Run existing tests to verify no regression**

```bash
cd /home/v/projects/voiceover-pipeline && uv run pytest --tb=short -q
```

Expected: all existing tests pass (model changes are backward-compatible with defaults).

**Step 4: Commit**

```bash
git add src/voiceover_pipeline/models.py src/voiceover_pipeline/providers/base.py
git commit -m "feat: add TranscriptionProvider ABC and provider field to TimingResult"
```

---

### Task 2: Extract FasterWhisperProvider from whisper_timing.py

**Objective:** Move the local faster-whisper logic into a proper provider class.

**Files:**
- Create: `src/voiceover_pipeline/providers/faster_whisper.py`
- Delete: `src/voiceover_pipeline/whisper_timing.py`
- Modify: `src/voiceover_pipeline/config.py` — move `_WHISPER_HF_REPOS` from `whisper_timing.py`

**Step 1: Move model registry to config**

In `src/voiceover_pipeline/config.py`, add:

```python
WHISPER_HF_REPOS: dict[str, str] = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
}
```

**Step 2: Create FasterWhisperProvider**

Create `src/voiceover_pipeline/providers/faster_whisper.py`:

```python
import shutil
import sys
from pathlib import Path

from voiceover_pipeline.config import WHISPER_HF_REPOS, DEFAULT_TIMING_MODEL, DEFAULT_TIMING_DEVICE, DEFAULT_TIMING_COMPUTE
from voiceover_pipeline.models import TimingResult, TimingSegment
from voiceover_pipeline.providers.base import TranscriptionProvider


def _detect_device(requested: str) -> str:  # moved from whisper_timing.py
    if requested == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return requested


def _detect_compute_type(requested: str, device: str) -> str:  # moved
    if requested != "auto":
        return requested
    if device == "cpu":
        return "int8"
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 12:
                return "float16"
            if cap[0] >= 7:
                return "int8_float16"
            return "int8"
    except Exception:
        pass
    return "int8"


class FasterWhisperProvider(TranscriptionProvider):
    provider_id = "faster-whisper"

    def __init__(
        self,
        model_size: str = DEFAULT_TIMING_MODEL,
        device: str = DEFAULT_TIMING_DEVICE,
        compute_type: str = DEFAULT_TIMING_COMPUTE,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"id": k, "parameters_m": ..., "disk_mb": ..., "speed": ...}
            for k in WHISPER_HF_REPOS
        ]

    def transcribe(
        self, audio_path: Path | str, language: str = "ru",
        word_timestamps: bool = False, quiet: bool = False,
    ) -> TimingResult:
        # ... same logic as original transcribe_audio() ...
        # Return TimingResult with provider=self.provider_id
```

**Step 3: Update `__init__.py`**

```python
from .faster_whisper import FasterWhisperProvider
# add to __all__
```

**Step 4: Update CLI references**

In `cli.py`, change `from .whisper_timing import transcribe_audio` to:
```python
from .providers.faster_whisper import FasterWhisperProvider
```

And in `_extract_timings`, replace `transcribe_audio(...)` call with:
```python
provider = FasterWhisperProvider(model_size=model, device=device, compute_type=compute_type)
timing = provider.transcribe(audio_path=audio_path, language=language, word_timestamps=word_timestamps, quiet=quiet)
```

**Step 5: Delete whisper_timing.py**

```bash
rm src/voiceover_pipeline/whisper_timing.py
```

**Step 6: Run tests**

```bash
cd /home/v/projects/voiceover-pipeline && uv run pytest --tb=short -q
```

Expected: all tests pass.

**Step 7: Commit**

```bash
git add src/voiceover_pipeline/providers/faster_whisper.py src/voiceover_pipeline/providers/__init__.py src/voiceover_pipeline/config.py src/voiceover_pipeline/cli.py
git rm src/voiceover_pipeline/whisper_timing.py
git commit -m "refactor: extract FasterWhisperProvider from whisper_timing.py"
```

---

### Task 3: Add OpenRouter Whisper provider

**Objective:** Implement `OpenRouterWhisperProvider` calling OpenRouter's `/api/v1/audio/transcriptions`.

**Files:**
- Create: `src/voiceover_pipeline/providers/openrouter_whisper.py`
- Modify: `src/voiceover_pipeline/config.py` — add constants
- Modify: `src/voiceover_pipeline/providers/__init__.py` — export

**Step 1: Add config constants**

In `src/voiceover_pipeline/config.py`:

```python
OPENROUTER_WHISPER_MODELS = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-1",
]
DEFAULT_TIMING_PROVIDER = "faster-whisper"
```

**Step 2: Create OpenRouterWhisperProvider**

Create `src/voiceover_pipeline/providers/openrouter_whisper.py`:

```python
import json
from pathlib import Path

import requests

from voiceover_pipeline.config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_WHISPER_MODELS,
    read_openrouter_key,
)
from voiceover_pipeline.models import TimingResult, TimingSegment
from voiceover_pipeline.providers.base import TranscriptionProvider


class OpenRouterWhisperProvider(TranscriptionProvider):
    provider_id = "openrouter-whisper"

    def __init__(self, model: str = "openai/whisper-large-v3-turbo", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or read_openrouter_key()
        self.base_url = OPENROUTER_BASE_URL.rstrip("/")

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"id": "openai/whisper-large-v3-turbo", "description": "Optimized Whisper Large V3 — fast, 99+ languages"},
            {"id": "openai/whisper-large-v3", "description": "Whisper Large V3 — highest accuracy"},
            {"id": "openai/whisper-1", "description": "Whisper v1 — legacy, cheapest"},
        ]

    def transcribe(
        self, audio_path: Path | str, language: str = "ru",
        word_timestamps: bool = False, quiet: bool = False,
    ) -> TimingResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, self._mime_type(audio_path))}
            data = {
                "model": self.model,
                "language": language,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word" if word_timestamps else "segment",
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            resp = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers, files=files, data=data, timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()

        segments = []
        for idx, seg in enumerate(result.get("segments", [])):
            start_ms = round(seg["start"] * 1000)
            end_ms = round(seg["end"] * 1000)
            words_list = None
            if word_timestamps and "words" in seg and seg["words"]:
                words_list = [
                    {"word": w["word"].strip(), "start_ms": round(w["start"] * 1000), "end_ms": round(w["end"] * 1000)}
                    for w in seg["words"]
                ]
            segments.append(TimingSegment(
                id=idx,
                start_sec=round(seg["start"], 3),
                end_sec=round(seg["end"], 3),
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                text=seg["text"].strip(),
                words=words_list,
            ))

        return TimingResult(
            segments=segments,
            model=self.model,
            backend="whisper",
            provider=self.provider_id,
            language=language,
            source_audio=str(audio_path.resolve()),
        )

    @staticmethod
    def _mime_type(path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".mp3": "audio/mpeg", ".wav": "audio/wav",
            ".ogg": "audio/ogg", ".flac": "audio/flac",
            ".m4a": "audio/mp4", ".webm": "audio/webm",
        }.get(ext, "audio/mpeg")
```

**Step 3: Update `__init__.py`**

```python
from .openrouter_whisper import OpenRouterWhisperProvider
# add to __all__
```

**Step 4: Write unit test**

Create `tests/test_openrouter_whisper.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch, Mock

from voiceover_pipeline.providers.openrouter_whisper import OpenRouterWhisperProvider


def test_list_models():
    provider = OpenRouterWhisperProvider(api_key="sk-test")
    models = provider.list_models()
    assert len(models) == 3
    assert models[0]["id"] == "openai/whisper-large-v3-turbo"


def test_transcribe_parses_segments():
    """Verify segments are correctly parsed from OpenRouter API response."""
    mock_response = {
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world", "words": [
                {"word": "Hello", "start": 0.0, "end": 1.0},
                {"word": "world", "start": 1.2, "end": 2.5},
            ]},
        ]
    }
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = Mock()
        
        provider = OpenRouterWhisperProvider(api_key="sk-test")
        # Create a tiny valid MP3 for the file check
        result = provider.transcribe(
            audio_path=Path(__file__).parent / "fixtures" / "silence.mp3",
            word_timestamps=True,
        )
    
    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello world"
    assert result.segments[0].start_ms == 0
    assert result.segments[0].end_ms == 2500
    assert result.segments[0].words is not None
    assert len(result.segments[0].words) == 2
    assert result.provider == "openrouter-whisper"
    assert result.backend == "whisper"
```

**Step 5: Run tests**

```bash
cd /home/v/projects/voiceover-pipeline && uv run pytest tests/test_openrouter_whisper.py -v
```

Expected: 2 passed.

**Step 6: Commit**

```bash
git add src/voiceover_pipeline/providers/openrouter_whisper.py src/voiceover_pipeline/providers/__init__.py src/voiceover_pipeline/config.py tests/test_openrouter_whisper.py
git commit -m "feat: add OpenRouterWhisperProvider for cloud transcription"
```

---

### Task 4: Add `--timing-provider` to CLI (timings + generate)

**Objective:** Let users choose between `faster-whisper` (local) and `openrouter-whisper` (cloud).

**Files:**
- Modify: `src/voiceover_pipeline/cli.py`

**Step 1: Add `--timing-provider` to `timings` subcommand**

In `build_parser()`, in the `timings` subparser (`timp`):

```python
timp.add_argument("--timing-provider", default="faster-whisper", choices=["faster-whisper", "openrouter-whisper"])
```

Also add `--timing-model` that accepts provider-specific model IDs:

```python
timp.add_argument("--timing-model", default=None, help="Provider-specific model ID (e.g. openai/whisper-large-v3-turbo)")
```

**Step 2: Add to `generate` subcommand's timing group**

```python
tim.add_argument("--timing-provider", default="faster-whisper", choices=["faster-whisper", "openrouter-whisper"])
```

**Step 3: Implement provider dispatch in `_extract_timings`**

```python
def _extract_timings(audio_path, output_dir, prefix, timing_provider, model, device, compute_type, language, word_timestamps=False, quiet=False):
    if timing_provider == "openrouter-whisper":
        from .providers.openrouter_whisper import OpenRouterWhisperProvider
        effective_model = model or "openai/whisper-large-v3-turbo"
        provider = OpenRouterWhisperProvider(model=effective_model)
    else:
        from .providers.faster_whisper import FasterWhisperProvider
        effective_model = model or DEFAULT_TIMING_MODEL
        provider = FasterWhisperProvider(model_size=effective_model, device=device, compute_type=compute_type)

    timing = provider.transcribe(
        audio_path=audio_path, language=language,
        word_timestamps=word_timestamps, quiet=quiet,
    )
    # ... rest is same (write json, srt)
```

**Step 4: Update `run_timings` to pass through `--timing-provider`**

Pass `args.timing_provider` to `_extract_timings`.

**Step 5: Update `list timing-models` → `timing-providers`**

Add `"timing-providers"` to the `list` target choices, and in `list_cmd`:

```python
elif args.target == "timing-providers":
    data = {
        "timing_providers": [
            {
                "id": "faster-whisper",
                "type": "local",
                "models": [
                    {"id": "base", "parameters_m": 74, "disk_mb": 148, "speed": "fastest"},
                    {"id": "small", "parameters_m": 244, "disk_mb": 486, "speed": "fast", "default": True},
                    ...
                ]
            },
            {
                "id": "openrouter-whisper",
                "type": "cloud",
                "currency": "USD",
                "models": [
                    {"id": "openai/whisper-large-v3-turbo", "description": "..."},
                    ...
                ]
            },
        ]
    }
```

Keep `"timing-models"` for backward compat (maps to faster-whisper models).

**Step 6: Update doctor for cloud provider**

In `doctor_cmd`, when `--with-timings` and provider is cloud, check API key instead of imports:

```python
if args.with_timings:
    if timing_provider == "openrouter-whisper":
        try:
            read_openrouter_key()
        except RuntimeError:
            fail("OPENROUTER_API_KEY not set", _EXIT_NO_KEY)
    else:
        _preflight_timing_dependency()
```

**Step 7: Run all tests**

```bash
cd /home/v/projects/voiceover-pipeline && uv run pytest --tb=short -q
```

**Step 8: Commit**

```bash
git add src/voiceover_pipeline/cli.py
git commit -m "feat: add --timing-provider flag to CLI (faster-whisper | openrouter-whisper)"
```

---

### Task 5: Integration smoke test

**Objective:** Verify the full pipeline works with the new cloud provider.

**Step 1: Run list command**

```bash
uv run voiceover list timing-providers --json
```

Expected: JSON with both providers and their models.

**Step 2: Test timings with cloud provider (real API call)**

```bash
uv run voiceover timings --audio out/some-test/full.mp3 --timing-provider openrouter-whisper --timing-model openai/whisper-large-v3-turbo --json
```

Expected: timings.json + .srt output.

**Step 3: Test generate with cloud timings**

```bash
uv run voiceover generate --provider openrouter-tts --with-timings --timing-provider openrouter-whisper --script in/script.md --run-id test-cloud --json
```

**Step 4: Final test suite**

```bash
cd /home/v/projects/voiceover-pipeline && uv run pytest --tb=short -q
```

Expected: all green.

**Step 5: Commit if any fixes**

```bash
git add -u && git commit -m "test: integration smoke test for cloud transcription"
```

---

## Summary

| # | Task | Files | Risk |
|---|------|-------|------|
| 1 | ABC + model update | `base.py`, `models.py` | Low — backward compat |
| 2 | Extract FasterWhisperProvider | `faster_whisper.py` (new), `whisper_timing.py` (delete) | Medium — moving code |
| 3 | OpenRouterWhisperProvider | `openrouter_whisper.py` (new) | Low — new file |
| 4 | CLI provider dispatch | `cli.py` | Medium — flag plumbing |
| 5 | Integration test | — | Low — smoke only |

**Future extensibility:** Adding another cloud provider (e.g., Polza Whisper when available) means:
1. Create `providers/polza_whisper.py` implementing `TranscriptionProvider`
2. Add to `__init__.py`
3. Add `"polza-whisper"` to CLI choices
4. That's it — no other files touched.

## Verification Checklist

- [ ] `voiceover list timing-providers --json` shows both providers
- [ ] `voiceover timings --audio X.mp3 --timing-provider faster-whisper` works (backward compat)
- [ ] `voiceover timings --audio X.mp3 --timing-provider openrouter-whisper` works (new)
- [ ] `voiceover generate --with-timings --timing-provider openrouter-whisper` works
- [ ] `voiceover doctor --with-timings` checks faster-whisper deps
- [ ] `voiceover doctor --with-timings --timing-provider openrouter-whisper` checks API key
- [ ] All existing tests pass
- [ ] No import errors on `voiceover --help`
