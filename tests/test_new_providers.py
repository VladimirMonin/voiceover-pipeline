import pytest
from unittest.mock import patch, MagicMock

from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider
from voiceover_pipeline.config import (
    PODCAST_NARRATION_PROMPT,
    TTS_PROMPT_MODE_NATIVE,
    TTS_PROMPT_MODE_PREFIX,
    TTS_PROMPT_MODE_NONE,
)
from voiceover_pipeline.tts_prompting import (
    resolve_prompt_mode,
    build_request_body,
    build_prompted_input,
    read_style_prompt_from_file,
)


class TestPolzaTTSProvider:
    def test_construction(self):
        p = PolzaTTSProvider(
            api_key="sk-test",
            model="elevenlabs/text-to-speech-turbo-2-5",
            voice="Rachel",
        )
        assert p.provider_id == "polza-tts"
        assert p.model == "elevenlabs/text-to-speech-turbo-2-5"
        assert p.voice == "Rachel"
        assert p.response_format == "mp3"
        assert p._is_elevenlabs is True

    def test_is_elevenlabs_openai_false(self):
        p = PolzaTTSProvider(api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash")
        assert p._is_elevenlabs is False

    def test_synthesize_media_elevenlabs(self):
        import json
        from voiceover_pipeline.config import POLZA_BASE_URL

        mock_submit = MagicMock()
        mock_submit.status_code = 200
        mock_submit.json.return_value = {"id": "gen-123", "status": "pending"}
        mock_submit.content = json.dumps({"id": "gen-123", "status": "pending"}).encode()

        mock_poll = MagicMock()
        mock_poll.status_code = 200
        mock_poll.json.return_value = {
            "id": "gen-123",
            "status": "completed",
            "data": [{"url": "https://s3.polza.ai/fake.mp3"}],
            "usage": {"cost_rub": 0.1575, "cost": 0.1575},
        }
        mock_poll.content = json.dumps(mock_poll.json.return_value).encode()

        # Second GET: download the audio
        mock_dl = MagicMock()
        mock_dl.status_code = 200
        mock_dl.content = b"fake-elevenlabs-audio"

        with patch("voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_submit) as mock_post:
            with patch("voiceover_pipeline.providers.polza_tts.requests.get", side_effect=[mock_poll, mock_dl]) as mock_get:
                with patch("voiceover_pipeline.providers.polza_tts.time.sleep", return_value=None):
                    p = PolzaTTSProvider(
                        api_key="sk-test",
                        model="elevenlabs/text-to-speech-turbo-2-5",
                        voice="Rachel",
                    )
                    result = p.synthesize_chunk("Hello elevenlabs", "chunk_01")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == f"{POLZA_BASE_URL}/media"
        json_body = call_args[1]["json"]
        assert json_body["model"] == "elevenlabs/text-to-speech-turbo-2-5"
        assert json_body["input"]["prompt"] == "Hello elevenlabs"
        assert json_body["input"]["voice"] == "Rachel"
        assert json_body["input"]["language_code"] == "ru"

        assert mock_get.call_count == 2
        assert result.audio_bytes == b"fake-elevenlabs-audio"
        import base64, json
        from voiceover_pipeline.config import POLZA_BASE_URL

        audio_b64 = base64.b64encode(b"fake-audio-bytes").decode()
        resp_json = {"audio": audio_b64, "contentType": "audio/mpeg", "model": "gpt-4o-mini-tts"}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps(resp_json).encode()
        mock_response.json.return_value = resp_json
        mock_response.headers = {"X-Generation-Id": "gen-123"}

        with patch("voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response) as mock_post:
            p = PolzaTTSProvider(api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash")
            result = p.synthesize_chunk("Hello world", "chunk_01")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == f"{POLZA_BASE_URL}/audio/speech"
        assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test"
        json_body = call_args[1]["json"]
        assert json_body["model"] == "openai/gpt-4o-mini-tts"
        assert json_body["input"] == "Hello world"
        assert json_body["voice"] == "ash"
        assert json_body["response_format"] == "mp3"

        assert isinstance(result, SynthesisResult)
        assert result.audio_bytes == b"fake-audio-bytes"
        assert result.audio_format == "mp3"
        assert result.generation_id == "gen-123"

    def test_synthesize_chunk_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.content = b"{}"
        mock_response.json.return_value = {}

        with patch("voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response):
            p = PolzaTTSProvider(api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash")
            with pytest.raises(RuntimeError, match="HTTP 500"):
                p.synthesize_chunk("Hello", "chunk_01")

    def test_synthesize_chunk_empty_body(self):
        import json
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({"audio": ""}).encode()
        mock_response.json.return_value = {"audio": ""}
        mock_response.headers = {}

        with patch("voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response):
            p = PolzaTTSProvider(api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash")
            with pytest.raises(RuntimeError, match="missing"):
                p.synthesize_chunk("Hello", "chunk_01")


class TestOpenRouterTTSProviderOpenAI:
    def test_is_openai_model_true(self):
        p = OpenRouterTTSProvider(
            api_key="sk-or",
            model="openai/gpt-4o-mini-tts-2025-12-15",
            voice="alloy",
        )
        assert p._is_openai_model is True

    def test_is_openai_model_false_for_gemini(self):
        p = OpenRouterTTSProvider(
            api_key="sk-or",
            model="google/gemini-3.1-flash-tts-preview",
            voice="Puck",
        )
        assert p._is_openai_model is False

    def test_openai_model_skips_style_prompt(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-or-1"}

        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="openai/gpt-4o-mini-tts-2025-12-15",
                voice="alloy",
                style_prompt="should be ignored",
            )
            result = p.synthesize_chunk("Hello openai", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "Hello openai"
        assert json_body["voice"] == "alloy"
        assert result.audio_bytes == b"fake-audio"
        assert "prompt" not in json_body
        assert "style_prompt" not in json_body.get("input", "").lower()

    def test_gemini_model_uses_native_prompt(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio-gemini"
        mock_response.headers = {"X-Generation-Id": "gen-gemini-1"}

        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="podcast narration style",
            )
            result = p.synthesize_chunk("Hello gemini", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "Hello gemini"
        assert json_body["prompt"] == "podcast narration style"
        assert json_body["voice"] == "Puck"
        assert result.audio_bytes == b"fake-audio-gemini"

    def test_gemini_no_style_prompt_sends_none_prompt(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio-gemini"
        mock_response.headers = {"X-Generation-Id": "gen-gemini-2"}

        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt=None,
            )
            result = p.synthesize_chunk("Hello gemini", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "Hello gemini"
        assert "prompt" not in json_body
        assert result.audio_bytes == b"fake-audio-gemini"


class TestGeminiExplicitPromptModes:
    def test_prefix_mode_falls_back_to_old_concatenation(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-1"}

        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="podcast style",
                prompt_mode="prefix",
            )
            p.synthesize_chunk("Hello", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "podcast style\n\nHello"
        assert "prompt" not in json_body

    def test_native_explicit_mode_sends_separate_prompt(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-2"}

        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="podcast style",
                prompt_mode="native",
            )
            p.synthesize_chunk("Hello", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "Hello"
        assert json_body["prompt"] == "podcast style"

    def test_none_mode_sends_no_prompt_field(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-3"}

        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="should be ignored",
                prompt_mode="none",
            )
            p.synthesize_chunk("Hello", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "Hello"
        assert "prompt" not in json_body


class TestUnknownGoogleModelFallback:
    def test_unknown_google_model_uses_native(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-future-1"}

        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-2.5-pro-tts",
                voice="Puck",
                style_prompt="expressive style",
            )
            p.synthesize_chunk("Future model", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "Future model"
        assert json_body["prompt"] == "expressive style"


class TestPromptModeResolution:
    def test_gemini_flash_tts_resolves_to_native(self):
        mode = resolve_prompt_mode("openrouter-tts", "google/gemini-3.1-flash-tts-preview")
        assert mode == TTS_PROMPT_MODE_NATIVE

    def test_unknown_google_resolves_to_native(self):
        mode = resolve_prompt_mode("openrouter-tts", "google/gemini-2.5-pro-tts")
        assert mode == TTS_PROMPT_MODE_NATIVE

    def test_openai_resolves_to_none(self):
        mode = resolve_prompt_mode("openrouter-tts", "openai/gpt-4o-mini-tts-2025-12-15")
        assert mode == TTS_PROMPT_MODE_NONE

    def test_explicit_none_overrides(self):
        mode = resolve_prompt_mode("openrouter-tts", "google/gemini-3.1-flash-tts-preview", TTS_PROMPT_MODE_NONE)
        assert mode == TTS_PROMPT_MODE_NONE

    def test_explicit_prefix_overrides(self):
        mode = resolve_prompt_mode("openrouter-tts", "google/gemini-3.1-flash-tts-preview", TTS_PROMPT_MODE_PREFIX)
        assert mode == TTS_PROMPT_MODE_PREFIX

    def test_unknown_provider_model_resolves_to_none(self):
        mode = resolve_prompt_mode("polza-tts", "elevenlabs/some-model")
        assert mode == TTS_PROMPT_MODE_NONE


class TestBuildRequestBody:
    def test_native_mode_body(self):
        body = build_request_body(
            model="google/gemini-3.1-flash-tts-preview",
            text="Hello world",
            voice="Puck",
            response_format="pcm",
            style_prompt="Be expressive",
            prompt_mode=TTS_PROMPT_MODE_NATIVE,
        )
        assert body["input"] == "Hello world"
        assert body["prompt"] == "Be expressive"
        assert body["model"] == "google/gemini-3.1-flash-tts-preview"
        assert body["voice"] == "Puck"

    def test_native_mode_no_prompt_when_none(self):
        body = build_request_body(
            model="google/gemini-3.1-flash-tts-preview",
            text="Hello world",
            voice="Puck",
            response_format="pcm",
            style_prompt=None,
            prompt_mode=TTS_PROMPT_MODE_NATIVE,
        )
        assert body["input"] == "Hello world"
        assert "prompt" not in body

    def test_prefix_mode_body(self):
        body = build_request_body(
            model="google/gemini-3.1-flash-tts-preview",
            text="Hello world",
            voice="Puck",
            response_format="pcm",
            style_prompt="Be expressive",
            prompt_mode=TTS_PROMPT_MODE_PREFIX,
        )
        assert body["input"] == "Be expressive\n\nHello world"
        assert "prompt" not in body

    def test_prefix_mode_no_prompt(self):
        body = build_request_body(
            model="google/gemini-3.1-flash-tts-preview",
            text="Hello world",
            voice="Puck",
            response_format="pcm",
            style_prompt=None,
            prompt_mode=TTS_PROMPT_MODE_PREFIX,
        )
        assert body["input"] == "Hello world"
        assert "prompt" not in body

    def test_none_mode_body(self):
        body = build_request_body(
            model="google/gemini-3.1-flash-tts-preview",
            text="Hello world",
            voice="Puck",
            response_format="pcm",
            style_prompt="should be ignored",
            prompt_mode=TTS_PROMPT_MODE_NONE,
        )
        assert body["input"] == "Hello world"
        assert "prompt" not in body


class TestBuildPromptedInput:
    def test_none_mode_returns_text(self):
        result = build_prompted_input("Hello", "style", TTS_PROMPT_MODE_NONE)
        assert result == "Hello"

    def test_prefix_mode_concatenates(self):
        result = build_prompted_input("Hello", "style", TTS_PROMPT_MODE_PREFIX)
        assert result == "style\n\nHello"

    def test_prefix_mode_returns_text_when_prompt_none(self):
        result = build_prompted_input("Hello", None, TTS_PROMPT_MODE_PREFIX)
        assert result == "Hello"

    def test_native_mode_returns_text_only(self):
        result = build_prompted_input("Hello", "style", TTS_PROMPT_MODE_NATIVE)
        assert result == "Hello"


class TestReadStylePromptFromFile:
    def test_reads_file_content(self, tmp_path):
        pf = tmp_path / "prompt.txt"
        pf.write_text("Custom podcast narration", encoding="utf-8")
        content = read_style_prompt_from_file(pf)
        assert content == "Custom podcast narration"

    def test_strips_whitespace(self, tmp_path):
        pf = tmp_path / "prompt.txt"
        pf.write_text("  Padded prompt  \n", encoding="utf-8")
        content = read_style_prompt_from_file(pf)
        assert content == "Padded prompt"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            read_style_prompt_from_file("nonexistent.txt")

    def test_raises_on_empty_file(self, tmp_path):
        pf = tmp_path / "empty.txt"
        pf.write_text("   ", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            read_style_prompt_from_file(pf)
