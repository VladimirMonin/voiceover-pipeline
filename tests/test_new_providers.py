import pytest
from unittest.mock import patch, MagicMock

from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider


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
        assert "style_prompt" not in json_body["input"].lower()

    def test_gemini_model_uses_style_prompt(self):
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
        assert "podcast narration style" in json_body["input"]
        assert "Hello gemini" in json_body["input"]
