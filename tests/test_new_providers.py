import base64
import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
import requests

from voiceover_pipeline.config import (
    POLZA_BASE_URL,
    TTS_PROMPT_MODE_NATIVE,
    TTS_PROMPT_MODE_NONE,
    TTS_PROMPT_MODE_PREFIX,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider
from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
from voiceover_pipeline.tts_prompting import (
    build_prompted_input,
    build_request_body,
    read_style_prompt_from_file,
    resolve_prompt_mode,
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

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_submit
        ) as mock_post:
            with patch(
                "voiceover_pipeline.providers.polza_tts.requests.get",
                side_effect=[mock_poll, mock_dl],
            ) as mock_get:
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
        import base64
        import json

        from voiceover_pipeline.config import POLZA_BASE_URL

        audio_b64 = base64.b64encode(b"fake-audio-bytes").decode()
        resp_json = {"audio": audio_b64, "contentType": "audio/mpeg", "model": "gpt-4o-mini-tts"}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps(resp_json).encode()
        mock_response.json.return_value = resp_json
        mock_response.headers = {"X-Generation-Id": "gen-123"}

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response
        ) as mock_post:
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

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response
        ):
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

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response
        ):
            p = PolzaTTSProvider(api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash")
            with pytest.raises(RuntimeError, match="missing"):
                p.synthesize_chunk("Hello", "chunk_01")

    def test_synthesize_chunk_speech_keeps_unquoted_usage_numbers_exact(self):
        import base64
        from decimal import Decimal

        import requests

        from voiceover_pipeline.config import POLZA_BASE_URL

        audio_b64 = base64.b64encode(b"fake-audio").decode()
        body = (
            f'{{"audio": "{audio_b64}", "contentType": "audio/mpeg", '
            '"usage": {"cost_rub": 0.1234567890123456789, "tokens": 3}}'
        )
        response = requests.Response()
        response.status_code = 200
        response.encoding = "utf-8"
        response._content = body.encode("utf-8")
        response.headers["X-Generation-Id"] = "gen-123"

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=response
        ) as mock_post:
            p = PolzaTTSProvider(api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash")
            result = p.synthesize_chunk("Hello", "chunk_01")

        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == f"{POLZA_BASE_URL}/audio/speech"
        usage = result.raw_metadata["usage_direct"]
        assert isinstance(usage["cost_rub"], Decimal)
        assert usage["cost_rub"] == Decimal("0.1234567890123456789")
        assert result.generation_id == "gen-123"

    def test_poll_media_completed_keeps_unquoted_usage_numbers_exact(self):
        from decimal import Decimal

        import requests

        submit = requests.Response()
        submit.status_code = 200
        submit.encoding = "utf-8"
        submit._content = b'{"id": "task-1", "status": "pending"}'

        poll = requests.Response()
        poll.status_code = 200
        poll.encoding = "utf-8"
        poll._content = (
            b'{"id": "task-1", "status": "completed", '
            b'"data": [{"url": "https://s3.polza.ai/fake.mp3"}], '
            b'"usage": {"cost_rub": 0.1234567890123456789}}'
        )

        download = requests.Response()
        download.status_code = 200
        download._content = b"fake-elevenlabs-audio"

        with patch("voiceover_pipeline.providers.polza_tts.requests.post", return_value=submit):
            with patch(
                "voiceover_pipeline.providers.polza_tts.requests.get",
                side_effect=[poll, download],
            ):
                with patch("voiceover_pipeline.providers.polza_tts.time.sleep", return_value=None):
                    p = PolzaTTSProvider(
                        api_key="sk-test",
                        model="elevenlabs/text-to-speech-turbo-2-5",
                        voice="Rachel",
                    )
                    result = p.synthesize_chunk("Hello", "chunk_01")

        assert result.audio_bytes == b"fake-elevenlabs-audio"
        usage = result.raw_metadata["usage_direct"]
        assert isinstance(usage["cost_rub"], Decimal)
        assert usage["cost_rub"] == Decimal("0.1234567890123456789")


class TestPolzaMediaProviderBoundary:
    """Accepted media tasks stay recoverable with GET-only requests.

    A media POST that was accepted is already paid for, so the provider reports
    the validated id through ``on_media_task_accepted`` before any poll or
    download, and completing that known id later must never issue a second POST.
    """

    POST_TARGET = "voiceover_pipeline.providers.polza_tts.requests.post"
    GET_TARGET = "voiceover_pipeline.providers.polza_tts.requests.get"
    SLEEP_TARGET = "voiceover_pipeline.providers.polza_tts.time.sleep"

    @staticmethod
    def _provider(
        model: str = "elevenlabs/text-to-speech-turbo-2-5",
        voice: str = "Rachel",
        **callbacks,
    ) -> PolzaTTSProvider:
        return PolzaTTSProvider(api_key="sk-test", model=model, voice=voice, **callbacks)

    @staticmethod
    def _json_response(
        body: bytes, status_code: int = 200, headers: dict | None = None
    ) -> requests.Response:
        response = requests.Response()
        response.status_code = status_code
        response.encoding = "utf-8"
        response._content = body
        if headers:
            response.headers.update(headers)
        return response

    def test_accepted_task_id_is_reported_before_poll_and_post_is_sent_once(self):
        accepted: list[str] = []
        gets_when_accepted: list[int] = []
        submit = self._json_response(b'{"id": "task-1", "status": "pending"}')

        def poll_timeout(*_args, **_kwargs):
            raise requests.Timeout("read timed out")

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET, return_value=submit) as mock_post:
                with patch(self.GET_TARGET, side_effect=poll_timeout) as mock_get:

                    def on_accepted(task_id: str) -> None:
                        accepted.append(task_id)
                        gets_when_accepted.append(mock_get.call_count)

                    with pytest.raises(requests.Timeout):
                        self._provider(on_media_task_accepted=on_accepted).synthesize_chunk(
                            "Hello", "chunk_01"
                        )

        assert accepted == ["task-1"]
        assert gets_when_accepted == [0]
        assert mock_post.call_count == 1
        assert mock_get.call_count == 1

    def test_recover_media_task_polls_and_downloads_without_post(self):
        poll = self._json_response(
            b'{"id": "task-1", "status": "completed", '
            b'"data": [{"url": "https://s3.polza.ai/fake.mp3"}], '
            b'"usage": {"cost_rub": 0.1234567890123456789}}'
        )
        download = self._json_response(b"fake-elevenlabs-audio")

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET) as mock_post:
                with patch(self.GET_TARGET, side_effect=[poll, download]) as mock_get:
                    result = self._provider().recover_media_task("task-1", "Hello", "chunk_01")

        mock_post.assert_not_called()
        assert mock_get.call_args_list[0].args[0] == f"{POLZA_BASE_URL}/media/task-1"
        assert mock_get.call_count == 2
        assert result.audio_bytes == b"fake-elevenlabs-audio"
        assert result.audio_format == "mp3"
        assert result.transcript == "Hello"
        assert result.generation_id == "task-1"
        assert result.client_path == "requests"
        usage = result.raw_metadata["usage_direct"]
        assert isinstance(usage["cost_rub"], Decimal)
        assert usage["cost_rub"] == Decimal("0.1234567890123456789")

    def test_recovered_completed_task_surfaces_usage_before_download_failure(self):
        completed: list[tuple[str, dict | None, str | None]] = []
        gets_when_completed: list[int] = []
        poll = self._json_response(
            b'{"id": "task-1", "status": "completed", '
            b'"data": [{"url": "https://s3.polza.ai/signed-secret.mp3"}], '
            b'"usage": {"cost_rub": 0.5}}'
        )

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET) as mock_post:
                with patch(
                    self.GET_TARGET,
                    side_effect=[poll, requests.Timeout("read timed out")],
                ) as mock_get:

                    def on_completed(task_id, usage, generation_id) -> None:
                        gets_when_completed.append(mock_get.call_count)
                        completed.append((task_id, usage, generation_id))

                    provider = self._provider(on_media_completed=on_completed)
                    with pytest.raises(requests.Timeout):
                        provider.recover_media_task("task-1", "Hello", "chunk_01")

        mock_post.assert_not_called()
        assert gets_when_completed == [1]
        assert len(completed) == 1
        task_id, usage, generation_id = completed[0]
        assert task_id == "task-1"
        assert generation_id == "task-1"
        assert usage == {"cost_rub": Decimal("0.5")}
        assert "s3.polza.ai" not in repr(completed)
        assert "signed-secret" not in repr(completed)

    def test_initial_route_surfaces_completed_task_without_signed_url(self):
        completed: list[tuple[str, dict | None, str | None]] = []
        submit = self._json_response(b'{"id": "task-1", "status": "pending"}')
        poll = self._json_response(
            b'{"id": "task-1", "status": "completed", '
            b'"data": [{"url": "https://s3.polza.ai/signed.mp3"}], '
            b'"usage": {"cost_rub": 0.25}}'
        )
        download = self._json_response(b"fake-elevenlabs-audio")

        def on_completed(task_id, usage, generation_id) -> None:
            completed.append((task_id, usage, generation_id))

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET, return_value=submit) as mock_post:
                with patch(self.GET_TARGET, side_effect=[poll, download]) as mock_get:
                    result = self._provider(on_media_completed=on_completed).synthesize_chunk(
                        "Hello", "chunk_01"
                    )

        assert mock_post.call_count == 1
        assert mock_get.call_count == 2
        assert completed == [("task-1", {"cost_rub": Decimal("0.25")}, "task-1")]
        assert "s3.polza.ai" not in repr(completed)
        assert result.audio_bytes == b"fake-elevenlabs-audio"
        assert result.generation_id == "task-1"

    @pytest.mark.parametrize(
        "unsafe_task_id",
        [
            "",
            "task-1/extra",
            "task-1?status=completed",
            "../../etc/passwd",
            "https://evil.example/media/1",
            "task 1",
        ],
    )
    def test_recover_media_task_rejects_unsafe_task_id_before_any_request(self, unsafe_task_id):
        with patch(self.POST_TARGET) as mock_post:
            with patch(self.GET_TARGET) as mock_get:
                with pytest.raises(ValueError, match="safe opaque token"):
                    self._provider().recover_media_task(unsafe_task_id, "Hello", "chunk_01")

        mock_post.assert_not_called()
        mock_get.assert_not_called()

    def test_submit_response_with_unsafe_task_id_fails_closed(self):
        accepted: list[str] = []
        submit = self._json_response(b'{"id": "../escape", "status": "pending"}')

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET, return_value=submit) as mock_post:
                with patch(self.GET_TARGET) as mock_get:
                    with pytest.raises(ValueError, match="safe opaque token"):
                        self._provider(on_media_task_accepted=accepted.append).synthesize_chunk(
                            "Hello", "chunk_01"
                        )

        assert mock_post.call_count == 1
        assert accepted == []
        mock_get.assert_not_called()

    def test_persistence_failure_propagates_without_poll_or_second_post(self):
        submit = self._json_response(b'{"id": "task-1", "status": "pending"}')

        def on_accepted(_task_id: str) -> None:
            raise OSError("state directory is read-only")

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET, return_value=submit) as mock_post:
                with patch(self.GET_TARGET) as mock_get:
                    with pytest.raises(OSError, match="read-only"):
                        self._provider(on_media_task_accepted=on_accepted).synthesize_chunk(
                            "Hello", "chunk_01"
                        )

        assert mock_post.call_count == 1
        mock_get.assert_not_called()

    TOKEN = "X-Amz-Signature=deadbeefcafe"
    SIGNED_URL = f"https://s3.polza.ai/audio.mp3?{TOKEN}"

    @staticmethod
    def _completed_body(url: str, *, generation_id: str = "task-1", cost: float = 0.5) -> bytes:
        return json.dumps(
            {
                "id": generation_id,
                "status": "completed",
                "data": [{"url": url}],
                "usage": {"cost_rub": cost},
            },
            separators=(",", ":"),
        ).encode()

    def test_unsafe_task_id_error_does_not_echo_signed_url(self):
        with patch(self.POST_TARGET) as mock_post:
            with patch(self.GET_TARGET) as mock_get:
                with pytest.raises(ValueError) as excinfo:
                    self._provider().recover_media_task(self.SIGNED_URL, "Hello", "chunk_01")

        assert self.TOKEN not in str(excinfo.value)
        assert "s3.polza.ai" not in repr(excinfo.value)
        assert "safe opaque token" in str(excinfo.value)
        mock_post.assert_not_called()
        mock_get.assert_not_called()

    def test_submit_error_body_does_not_echo_signed_url(self):
        submit = self._json_response(
            json.dumps(
                {"error": f"bad request at {self.SIGNED_URL}"}, separators=(",", ":")
            ).encode(),
            status_code=400,
        )
        with patch(self.POST_TARGET, return_value=submit) as mock_post:
            with patch(self.GET_TARGET) as mock_get:
                with pytest.raises(RuntimeError) as excinfo:
                    self._provider().synthesize_chunk("Hello", "chunk_01")

        assert self.TOKEN not in str(excinfo.value)
        assert "s3.polza.ai" not in str(excinfo.value)
        assert str(excinfo.value) == "HTTP 400"
        assert mock_post.call_count == 1
        mock_get.assert_not_called()

    def test_poll_http_error_body_does_not_echo_signed_url(self):
        submit = self._json_response(b'{"id": "task-1", "status": "pending"}')
        poll = self._json_response(
            json.dumps({"error": f"upstream {self.SIGNED_URL}"}, separators=(",", ":")).encode(),
            status_code=500,
        )
        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET, return_value=submit):
                with patch(self.GET_TARGET, return_value=poll) as mock_get:
                    with pytest.raises(RuntimeError) as excinfo:
                        self._provider().synthesize_chunk("Hello", "chunk_01")

        assert self.TOKEN not in str(excinfo.value)
        assert "s3.polza.ai" not in str(excinfo.value)
        assert str(excinfo.value) == "HTTP 500"
        assert mock_get.call_count == 1

    def test_failed_status_payload_does_not_echo_signed_url(self):
        submit = self._json_response(b'{"id": "task-1", "status": "pending"}')
        poll = self._json_response(
            json.dumps(
                {"id": "task-1", "status": "failed", "error": f"broke at {self.SIGNED_URL}"},
                separators=(",", ":"),
            ).encode()
        )
        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET, return_value=submit):
                with patch(self.GET_TARGET, return_value=poll) as mock_get:
                    with pytest.raises(RuntimeError) as excinfo:
                        self._provider().synthesize_chunk("Hello", "chunk_01")

        assert self.TOKEN not in str(excinfo.value)
        assert "s3.polza.ai" not in str(excinfo.value)
        assert "task-1" in str(excinfo.value)
        assert mock_get.call_count == 1

    def test_signed_url_download_timeout_does_not_echo_url(self):
        poll = self._json_response(self._completed_body(self.SIGNED_URL))
        download = requests.Timeout(f"HTTPSConnectionPool read timed out for {self.SIGNED_URL}")

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET) as mock_post:
                with patch(self.GET_TARGET, side_effect=[poll, download]) as mock_get:
                    with pytest.raises(requests.Timeout) as excinfo:
                        self._provider().recover_media_task("task-1", "Hello", "chunk_01")

        assert self.TOKEN not in str(excinfo.value)
        assert "s3.polza.ai" not in str(excinfo.value)
        assert mock_get.call_count == 2
        mock_post.assert_not_called()

    @pytest.mark.parametrize("status_code,body", [(403, b"AccessDenied"), (200, b"")])
    def test_download_failure_does_not_echo_signed_url(self, status_code, body):
        poll = self._json_response(self._completed_body(self.SIGNED_URL))
        download = self._json_response(body, status_code=status_code)

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET) as mock_post:
                with patch(self.GET_TARGET, side_effect=[poll, download]) as mock_get:
                    with pytest.raises(RuntimeError) as excinfo:
                        self._provider().recover_media_task("task-1", "Hello", "chunk_01")

        assert self.TOKEN not in str(excinfo.value)
        assert "s3.polza.ai" not in str(excinfo.value)
        assert mock_get.call_count == 2
        mock_post.assert_not_called()

    def test_malicious_completed_generation_id_is_not_exposed(self):
        reported: list[tuple] = []
        poll = self._json_response(
            self._completed_body("https://s3.polza.ai/real.mp3", generation_id=self.SIGNED_URL)
        )
        download = self._json_response(b"fake-elevenlabs-audio")

        with patch(self.SLEEP_TARGET, return_value=None):
            with patch(self.POST_TARGET) as mock_post:
                with patch(self.GET_TARGET, side_effect=[poll, download]) as mock_get:
                    result = self._provider(
                        on_media_completed=lambda *args: reported.append(args)
                    ).recover_media_task("task-1", "Hello", "chunk_01")

        mock_post.assert_not_called()
        assert mock_get.call_count == 2
        assert reported == [("task-1", {"cost_rub": Decimal("0.5")}, None)]
        assert self.TOKEN not in repr(reported)
        assert "s3.polza.ai" not in repr(reported)
        assert result.generation_id == "task-1"
        assert result.audio_bytes == b"fake-elevenlabs-audio"

    def test_openai_speech_route_never_invokes_media_callbacks(self):
        accepted: list[str] = []
        completed: list[tuple] = []
        audio_b64 = base64.b64encode(b"fake-audio").decode()
        body = json.dumps(
            {"audio": audio_b64, "contentType": "audio/mpeg"}, separators=(",", ":")
        ).encode()
        response = self._json_response(body, headers={"X-Generation-Id": "gen-123"})

        with patch(self.POST_TARGET, return_value=response) as mock_post:
            result = self._provider(
                model="openai/gpt-4o-mini-tts",
                voice="ash",
                on_media_task_accepted=accepted.append,
                on_media_completed=lambda *args: completed.append(args),
            ).synthesize_chunk("Hello", "chunk_01")

        assert mock_post.call_args[0][0] == f"{POLZA_BASE_URL}/audio/speech"
        assert accepted == []
        assert completed == []
        assert result.generation_id == "gen-123"


class TestPolzaChatAudioProviderSinglePaidSubmit:
    """The chat-audio route must never send a second paid POST by itself.

    ``fallback_voice`` is a constructor-compatibility argument only: this route
    is a single paid streaming submit, so a timeout or a broken stream that may
    already have been accepted and billed is never followed by an automatic
    request for another voice. Choosing a different voice means a new run.
    """

    POST_TARGET = "voiceover_pipeline.providers.polza_chat_audio.requests.post"

    @staticmethod
    def _provider(fallback_voice: str = "onyx"):
        from voiceover_pipeline.providers.polza_chat_audio import PolzaChatAudioProvider

        return PolzaChatAudioProvider(
            api_key="sk-test",
            model="openai/gpt-audio-mini",
            voice="ash",
            fallback_voice=fallback_voice,
        )

    @staticmethod
    def _streaming_response(lines, status_code: int = 200):
        response = MagicMock()
        response.status_code = status_code
        response.headers = {}
        response.iter_lines.return_value = [line.encode("utf-8") for line in lines]
        return response

    def test_constructor_keeps_distinct_fallback_voice(self):
        provider = self._provider()
        assert provider.voice == "ash"
        assert provider.fallback_voice == "onyx"

    def test_timeout_on_first_submit_posts_once_and_propagates(self, capsys):
        import requests

        posted_voices: list[str] = []

        def post(_url, **kwargs):
            posted_voices.append(kwargs["json"]["audio"]["voice"])
            raise requests.Timeout("read timed out")

        with patch(self.POST_TARGET, side_effect=post):
            with pytest.raises(requests.Timeout):
                self._provider().synthesize_chunk("Hello", "chunk_01")

        assert posted_voices == ["ash"]
        assert capsys.readouterr().out == ""

    def test_post_submit_http_error_posts_once(self):
        response = MagicMock()
        response.status_code = 500
        response.text = "Internal Server Error"
        response.headers = {}

        with patch(self.POST_TARGET, return_value=response) as mock_post:
            with pytest.raises(RuntimeError, match="HTTP 500"):
                self._provider().synthesize_chunk("Hello", "chunk_01")

        assert mock_post.call_count == 1
        assert mock_post.call_args.kwargs["json"]["audio"]["voice"] == "ash"

    def test_empty_audio_stream_posts_once(self):
        with patch(
            self.POST_TARGET, return_value=self._streaming_response(["data: [DONE]"])
        ) as mock_post:
            with pytest.raises(RuntimeError, match="without audio chunks"):
                self._provider().synthesize_chunk("Hello", "chunk_01")

        assert mock_post.call_count == 1
        assert mock_post.call_args.kwargs["json"]["audio"]["voice"] == "ash"

    def test_primary_voice_success_never_posts_fallback(self):
        import base64

        audio_b64 = base64.b64encode(b"\x00\x01").decode()
        chunk = json.dumps(
            {"choices": [{"delta": {"audio": {"data": audio_b64, "transcript": "Hello"}}}]}
        )
        response = self._streaming_response([f"data: {chunk}", "", "data: [DONE]"])

        with patch(self.POST_TARGET, return_value=response) as mock_post:
            result = self._provider().synthesize_chunk("Hello", "chunk_01")

        assert mock_post.call_count == 1
        assert mock_post.call_args.kwargs["json"]["audio"]["voice"] == "ash"
        assert result.audio_bytes == b"\x00\x01"
        assert result.audio_format == "pcm16"
        assert result.raw_metadata["voice"] == "ash"


class TestOpenRouterTTSProviderOpenAI:
    def test_stale_openai_speech_model_fails_before_billing(self):
        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post") as mock_post:
            with pytest.raises(ValueError, match="not in the current OpenRouter speech catalog"):
                OpenRouterTTSProvider(
                    api_key="sk-or",
                    model="openai/gpt-4o-mini-tts-2025-12-15",
                    voice="alloy",
                )

        mock_post.assert_not_called()

    def test_is_openai_model_false_for_gemini(self):
        p = OpenRouterTTSProvider(
            api_key="sk-or",
            model="google/gemini-3.1-flash-tts-preview",
            voice="Puck",
        )
        assert p._is_openai_model is False

    def test_gemini_model_uses_documented_speech_payload(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"ID3fake-audio-gemini"
        mock_response.headers = {
            "Content-Type": "audio/mpeg",
            "X-Generation-Id": "gen-gemini-1",
        }

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="podcast narration style",
            )
            result = p.synthesize_chunk("Hello gemini", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body == {
            "model": "google/gemini-3.1-flash-tts-preview",
            "input": "Hello gemini",
            "voice": "Puck",
            "response_format": "pcm",
        }
        assert mock_post.call_args[1]["headers"]["X-Title"] == "Voiceover Pipeline"
        assert result.audio_bytes == b"ID3fake-audio-gemini"
        assert result.audio_format == "mp3"

    def test_gemini_model_ignores_legacy_multispeaker_config(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio-gemini"
        mock_response.headers = {"X-Generation-Id": "gen-gemini-multi"}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="podcast style",
                speaker_voice_map={"Speaker1": "Puck", "Speaker2": "Kore"},
            )
            result = p.synthesize_chunk("Speaker1: Hello\nSpeaker2: Hi", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["voice"] == "Puck"
        assert "multi_speaker_voice_config" not in json_body
        assert result.audio_bytes == b"fake-audio-gemini"

    def test_gemini_no_style_prompt_sends_none_prompt(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio-gemini"
        mock_response.headers = {"X-Generation-Id": "gen-gemini-2"}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
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
    def test_prefix_mode_cannot_change_openrouter_synthesis_input(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-1"}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="podcast style",
                prompt_mode="prefix",
            )
            p.synthesize_chunk("Hello", "chunk_01")

        json_body = mock_post.call_args[1]["json"]
        assert json_body["input"] == "Hello"
        assert "prompt" not in json_body

    def test_native_explicit_mode_cannot_add_prompt_field(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-native"}
        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
            p = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Puck",
                style_prompt="podcast style",
                prompt_mode="native",
            )
            p.synthesize_chunk("Hello", "chunk_01")

        assert mock_post.call_args.kwargs["json"] == {
            "model": "google/gemini-3.1-flash-tts-preview",
            "input": "Hello",
            "voice": "Puck",
            "response_format": "pcm",
        }

    def test_none_mode_sends_no_prompt_field(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.headers = {"X-Generation-Id": "gen-3"}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
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
    def test_unknown_google_model_fails_before_billing(self):
        with patch("voiceover_pipeline.providers.openrouter_tts.requests.post") as mock_post:
            with pytest.raises(ValueError, match="not in the current OpenRouter speech catalog"):
                OpenRouterTTSProvider(
                    api_key="sk-or",
                    model="google/gemini-2.5-pro-tts",
                    voice="Puck",
                    style_prompt="expressive style",
                )

        mock_post.assert_not_called()


class TestPromptModeResolution:
    def test_gemini_flash_tts_resolves_to_none(self):
        mode = resolve_prompt_mode("openrouter-tts", "google/gemini-3.1-flash-tts-preview")
        assert mode == TTS_PROMPT_MODE_NONE

    def test_unknown_google_resolves_to_native(self):
        mode = resolve_prompt_mode("openrouter-tts", "google/gemini-2.5-pro-tts")
        assert mode == TTS_PROMPT_MODE_NATIVE

    def test_openai_resolves_to_none(self):
        mode = resolve_prompt_mode("openrouter-tts", "openai/gpt-4o-mini-tts-2025-12-15")
        assert mode == TTS_PROMPT_MODE_NONE

    def test_explicit_none_overrides(self):
        mode = resolve_prompt_mode(
            "openrouter-tts", "google/gemini-3.1-flash-tts-preview", TTS_PROMPT_MODE_NONE
        )
        assert mode == TTS_PROMPT_MODE_NONE

    def test_explicit_prefix_overrides(self):
        mode = resolve_prompt_mode(
            "openrouter-tts", "google/gemini-3.1-flash-tts-preview", TTS_PROMPT_MODE_PREFIX
        )
        assert mode == TTS_PROMPT_MODE_PREFIX

    def test_unknown_provider_model_resolves_to_none(self):
        mode = resolve_prompt_mode("polza-tts", "elevenlabs/some-model")
        assert mode == TTS_PROMPT_MODE_NONE


class TestGeminiMultiSpeakerRequestShape:
    def _write_validated_dialogue(self, tmp_path):
        from voiceover_pipeline.gemini_dialogue import validate_gemini_dialogue_file

        script = tmp_path / "podcast.md"
        script.write_text(
            "\n".join(
                [
                    "---",
                    "format: gemini-dialogue",
                    "language: ru",
                    "model: google/gemini-3.1-flash-tts-preview",
                    "speakers:",
                    "  Host:",
                    "    display_name: Ведущая",
                    "    voice: Kore",
                    "    profile: warm host",
                    "  Guest:",
                    "    display_name: Гость",
                    "    voice: Puck",
                    "    profile: calm expert",
                    "vibe: >",
                    "  Russian technical podcast. Natural question-and-answer conversation.",
                    "allowed_tags:",
                    "  - warmly",
                    "  - curious",
                    "max_chunk_bytes: 3500",
                    "---",
                    "Host: [warmly] Что умеет утилита?",
                    "Guest: Она создаёт озвучку и субтитры.",
                    "******",
                    "Host: [curious] Можно работать локально?",
                    "Guest: Да, для одноголосой озвучки есть OmniVoice.",
                ]
            ),
            encoding="utf-8",
        )
        report = validate_gemini_dialogue_file(script)
        assert report["valid"] is True
        return report

    def _request_body(self, tmp_path, report):
        from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio-gemini"
        mock_response.headers = {"X-Generation-Id": "gen-gemini-multi"}
        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
            provider = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice=next(iter(report["speaker_voice_map"].values())),
                style_prompt=report["style_prompt"],
                speaker_voice_map=report["speaker_voice_map"],
            )
            provider.synthesize_chunk(
                "Host: Что умеет утилита?\nGuest: Она создаёт озвучку.", "chunk_01"
            )
        return mock_post.call_args[1]["json"]

    def test_request_has_no_undocumented_multispeaker_config(self, tmp_path):
        report = self._write_validated_dialogue(tmp_path)
        body = self._request_body(tmp_path, report)
        assert "multi_speaker_voice_config" not in body

    def test_request_is_single_voice(self, tmp_path):
        report = self._write_validated_dialogue(tmp_path)
        body = self._request_body(tmp_path, report)
        assert body["voice"] == "Kore"

    def test_request_contains_only_documented_fields(self, tmp_path):
        report = self._write_validated_dialogue(tmp_path)
        body = self._request_body(tmp_path, report)
        assert set(body) == {"model", "input", "voice", "response_format"}

    def test_top_level_compatibility_voice_equals_first_validated_voice(self, tmp_path):
        report = self._write_validated_dialogue(tmp_path)
        body = self._request_body(tmp_path, report)
        first_voice = next(iter(report["speaker_voice_map"].values()))
        assert body["voice"] == first_voice

    def test_no_third_speaker_or_raw_frontmatter_in_request(self, tmp_path):
        report = self._write_validated_dialogue(tmp_path)
        body = self._request_body(tmp_path, report)
        assert "multi_speaker_voice_config" not in body
        serialized = json.dumps(body, ensure_ascii=False)
        assert "max_chunk_bytes" not in serialized
        assert "allowed_tags" not in serialized
        assert "vibe" not in serialized
        assert "display_name" not in serialized

    def test_single_speaker_map_is_rejected_at_validation(self, tmp_path):
        from voiceover_pipeline.gemini_dialogue import validate_speakers

        errors: list[dict] = []
        validate_speakers({"Host": "Kore"}, errors)
        codes = [item["code"] for item in errors]
        assert "SPEAKER_COUNT_INVALID" in codes
        assert "DUPLICATE_SPEAKER_VOICE" not in codes


class TestOpenRouterDocumentedAudioResponses:
    @pytest.mark.parametrize(
        ("content_type", "audio_bytes", "expected_format"),
        [
            ("audio/mpeg", b"ID3fixture", "mp3"),
            ("audio/pcm", b"\x00\x01", "pcm16"),
        ],
    )
    def test_accepts_documented_raw_audio_streams(self, content_type, audio_bytes, expected_format):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = audio_bytes
        mock_response.headers = {"Content-Type": content_type}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ):
            provider = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Kore",
                style_prompt=None,
            )
            result = provider.synthesize_chunk("Hello", "turn_0001")

        assert result.audio_bytes == audio_bytes
        assert result.audio_format == expected_format

    @pytest.mark.parametrize(
        ("status_code", "content_type", "body", "error_match"),
        [
            (502, "application/json", b'{"error":"No successful provider responses"}', "HTTP 502"),
            (200, "audio/mpeg", b"", "empty audio body"),
            (200, "application/json", b'{"audio":"ZmFrZQ=="}', "non-audio response"),
            (200, "text/plain", b"upstream returned prose", "non-audio response"),
            (200, "audio/wav", b"RIFFfixtureWAVE", "non-audio response"),
            (200, "audio/L16", b"\x00\x01", "non-audio response"),
        ],
    )
    def test_failed_response_makes_exactly_one_request(
        self, status_code, content_type, body, error_match
    ):
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = body
        mock_response.text = body.decode("utf-8")
        mock_response.headers = {"Content-Type": content_type}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ) as mock_post:
            provider = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Kore",
                style_prompt="podcast style",
            )
            with pytest.raises(RuntimeError, match=error_match):
                provider.synthesize_chunk("Hello", "turn_0001")

        assert mock_post.call_count == 1

    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("application/json", b'{"audio":"ZmFrZQ=="}'),
            ("text/plain", b"data:audio/mpeg;base64,ZmFrZQ=="),
            ("text/event-stream", b'data: {"choices":[]}\n\n'),
        ],
    )
    def test_rejects_undocumented_wrapped_audio_without_guessing(self, content_type, body):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = body
        mock_response.headers = {"Content-Type": content_type}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ):
            provider = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Kore",
                style_prompt=None,
            )
            with pytest.raises(RuntimeError, match="non-audio response"):
                provider.synthesize_chunk("Hello", "turn_0001")

    def test_rejects_empty_raw_audio_stream(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b""
        mock_response.headers = {"Content-Type": "audio/mpeg"}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ):
            provider = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Kore",
                style_prompt=None,
            )
            with pytest.raises(RuntimeError, match="empty audio body"):
                provider.synthesize_chunk("Hello", "turn_0001")

    def test_gemini_missing_content_type_uses_requested_pcm_format(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"\x00\x01"
        mock_response.headers = {}

        with patch(
            "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=mock_response
        ):
            provider = OpenRouterTTSProvider(
                api_key="sk-or",
                model="google/gemini-3.1-flash-tts-preview",
                voice="Kore",
                style_prompt=None,
            )
            result = provider.synthesize_chunk("Hello", "turn_0001")

        assert result.audio_format == "pcm16"


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


def _wav_fixture() -> bytes:
    """A minimal valid 44-byte PCM RIFF/WAVE header with no samples."""
    return (
        b"RIFF"
        + (36).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (24000).to_bytes(4, "little")
        + (48000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + (0).to_bytes(4, "little")
    )


class TestPolzaSpeechAudioFormat:
    @staticmethod
    def _synthesize(resp_json, *, response_format="mp3"):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps(resp_json).encode()
        mock_response.json.return_value = resp_json
        mock_response.headers = {}

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response
        ):
            provider = PolzaTTSProvider(
                api_key="sk-test",
                model="openai/gpt-4o-mini-tts",
                voice="ash",
                response_format=response_format,
            )
            return provider.synthesize_chunk("Hello world", "chunk_01")

    @pytest.mark.parametrize(
        ("content_type", "audio_bytes", "expected_format"),
        [
            ("audio/mpeg", b"ID3fixture", "mp3"),
            ("audio/pcm", b"\x00\x01\x02\x03", "pcm16"),
            ("audio/wav", _wav_fixture(), "wav"),
        ],
    )
    def test_maps_documented_content_types(self, content_type, audio_bytes, expected_format):
        resp_json = {
            "audio": base64.b64encode(audio_bytes).decode(),
            "contentType": content_type,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps(resp_json).encode()
        mock_response.json.return_value = resp_json
        mock_response.headers = {}

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response
        ):
            provider = PolzaTTSProvider(
                api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash"
            )
            result = provider.synthesize_chunk("Hello world", "chunk_01")

        assert result.audio_format == expected_format
        assert result.audio_bytes == audio_bytes

    @pytest.mark.parametrize(
        ("content_type", "audio_bytes", "error_match"),
        [
            ("audio/wav", b"not-a-wav-header", "without a valid WAV header"),
            ("audio/flac", b"fLaCfixture", "unsupported audio content type"),
        ],
    )
    def test_unsupported_or_mislabelled_audio_fails_closed(
        self, content_type, audio_bytes, error_match
    ):
        resp_json = {
            "audio": base64.b64encode(audio_bytes).decode(),
            "contentType": content_type,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps(resp_json).encode()
        mock_response.json.return_value = resp_json
        mock_response.headers = {}

        with patch(
            "voiceover_pipeline.providers.polza_tts.requests.post", return_value=mock_response
        ):
            provider = PolzaTTSProvider(
                api_key="sk-test", model="openai/gpt-4o-mini-tts", voice="ash"
            )
            with pytest.raises(RuntimeError, match=error_match):
                provider.synthesize_chunk("Hello world", "chunk_01")

    def test_missing_content_type_with_real_wav_reports_wav(self):
        # ``response_format`` was mp3, but the bytes carry a real RIFF/WAVE
        # header, so the actual container is what the receipt must record.
        result = self._synthesize({"audio": base64.b64encode(_wav_fixture()).decode()})

        assert result.audio_format == "wav"
        assert result.audio_bytes == _wav_fixture()

    def test_declared_mp3_with_real_wav_reports_wav(self):
        result = self._synthesize(
            {
                "audio": base64.b64encode(_wav_fixture()).decode(),
                "contentType": "audio/mpeg",
            }
        )

        assert result.audio_format == "wav"
        assert result.audio_bytes == _wav_fixture()

    def test_l16_content_type_is_rejected_as_unsupported(self):
        # L16 is network big-endian while media.py decodes pcm16 as little-endian
        # s16le, so it must fail closed rather than byte-swap every sample.
        with pytest.raises(RuntimeError, match="unsupported audio content type"):
            self._synthesize(
                {
                    "audio": base64.b64encode(b"\x00\x01").decode(),
                    "contentType": "audio/l16",
                }
            )


class _FakeRun:
    def __init__(self, args):
        self.args = args
        self.returncode = 0
        self.stdout = b""
        self.stderr = b""


class TestRawAudioFormatRouting:
    def test_mp3_is_written_without_ffmpeg(self, tmp_path):
        from voiceover_pipeline import media

        out = tmp_path / "chunk.mp3"
        media.write_audio_as_mp3("ffmpeg", b"ID3fixture", "mp3", out)
        assert out.read_bytes() == b"ID3fixture"

    def test_genuine_pcm16_is_piped_as_s16le(self, tmp_path, monkeypatch):
        from voiceover_pipeline import media

        calls: list[list[str]] = []
        monkeypatch.setattr(
            media.subprocess, "run", lambda args, **kwargs: calls.append(args) or _FakeRun(args)
        )
        media.write_audio_as_mp3("ffmpeg", b"\x00\x01\x02\x03", "pcm16", tmp_path / "chunk.mp3")
        assert calls and "s16le" in calls[0]

    def test_wav_container_is_never_piped_as_pcm(self, tmp_path, monkeypatch):
        from voiceover_pipeline import media

        calls: list[list[str]] = []
        monkeypatch.setattr(
            media.subprocess, "run", lambda args, **kwargs: calls.append(args) or _FakeRun(args)
        )
        # Labelled PCM on purpose: the RIFF/WAVE container must still be decoded
        # as a file, never reinterpreted through the s16le demuxer.
        media.write_audio_as_mp3("ffmpeg", _wav_fixture(), "pcm16", tmp_path / "chunk.mp3")
        assert calls
        assert all("s16le" not in call for call in calls)

    def test_labeled_wav_is_decoded_as_a_file(self, tmp_path, monkeypatch):
        from voiceover_pipeline import media

        calls: list[list[str]] = []
        monkeypatch.setattr(
            media.subprocess, "run", lambda args, **kwargs: calls.append(args) or _FakeRun(args)
        )
        media.write_audio_as_mp3("ffmpeg", _wav_fixture(), "wav", tmp_path / "chunk.mp3")
        assert calls
        assert all("s16le" not in call for call in calls)

    def test_wav_header_beats_mp3_label_and_is_never_copied(self, tmp_path, monkeypatch):
        from voiceover_pipeline import media

        calls: list[list[str]] = []
        monkeypatch.setattr(
            media.subprocess, "run", lambda args, **kwargs: calls.append(args) or _FakeRun(args)
        )
        out = tmp_path / "chunk.mp3"
        # Labelled MP3 on purpose: real RIFF/WAVE bytes must be decoded through
        # ffmpeg, never fast-copied into the ``.mp3`` output.
        media.write_audio_as_mp3("ffmpeg", _wav_fixture(), "mp3", out)
        assert calls
        assert calls[0][calls[0].index("-i") + 1].endswith(".wav")
        assert all("s16le" not in call for call in calls)
        assert not out.exists()
