"""Regression tests for observed-cost handling in the pricing pipeline.

``cost_from_generation`` treats an observed zero cost (``0``/``0.0``/``Decimal("0")``)
as a real billing fact, not a missing value, so it wins over any fallback field.
``cli.attach_costs`` may bind a Polza cost by the chunk's own known remote
generation id only for ``polza-chat-audio``, whose id comes exclusively from the
response ``X-Generation-Id`` header. ``polza-tts`` ids mix media task ids and
generic ``/audio/speech`` body ids with resumed chunk artifacts, so they are not
comparable to history ids and must never trigger a history lookup; those chunks
keep their direct cost and generation id untouched. No lookup may erase an
existing direct cost. These tests cover the field-level selection, the exact
``Decimal`` parsing at both detail boundaries, the strict-JSON metadata
projection, and the Polza id boundary.
"""

import base64
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
import requests

from voiceover_pipeline import cli, pricing
from voiceover_pipeline.models import ChunkArtifact
from voiceover_pipeline.pricing import cost_from_generation


class TestPolzaGenerationCost:
    @pytest.mark.parametrize("provider", ["polza-chat-audio", "polza-tts"])
    def test_zero_client_cost_wins_over_nonzero_cost(self, provider):
        generation = {"clientCost": 0, "cost": 1.25}

        cost, cost_exact, currency = cost_from_generation(provider, generation)

        assert cost == 0.0
        assert cost_exact == "0"
        assert currency == "RUB"

    @pytest.mark.parametrize("provider", ["polza-chat-audio", "polza-tts"])
    def test_zero_float_client_cost_wins_over_nonzero_cost(self, provider):
        generation = {"clientCost": 0.0, "cost": 1.25}

        cost, cost_exact, currency = cost_from_generation(provider, generation)

        assert cost == 0.0
        assert cost_exact is None
        assert currency == "RUB"

    def test_missing_client_cost_falls_back_to_cost(self):
        generation = {"cost": 0.5}

        cost, cost_exact, currency = cost_from_generation("polza-tts", generation)

        assert cost == 0.5
        assert cost_exact is None
        assert currency == "RUB"

    def test_none_client_cost_falls_back_to_cost(self):
        generation = {"clientCost": None, "cost": 0.5}

        cost, cost_exact, currency = cost_from_generation("polza-tts", generation)

        assert cost == 0.5
        assert cost_exact is None
        assert currency == "RUB"

    def test_no_cost_fields_reports_unknown(self):
        assert cost_from_generation("polza-tts", {}) == (None, None, None)

    def test_exact_string_value_reports_legacy_float_and_exact_string(self):
        cost, cost_exact, currency = cost_from_generation("polza-tts", {"clientCost": "0.1575"})

        assert cost == 0.1575
        assert cost_exact == "0.1575"
        assert currency == "RUB"


class TestOpenRouterGenerationCost:
    def test_zero_total_cost_wins_over_nonzero_fallbacks(self):
        generation = {"total_cost": 0, "cost": 0.5, "usage": {"prompt_tokens": 10}}

        cost, cost_exact, currency = cost_from_generation("openrouter-tts", generation)

        assert cost == 0.0
        assert cost_exact == "0"
        assert currency == "USD"

    def test_zero_total_cost_wins_when_usage_is_non_numeric(self):
        generation = {"total_cost": 0, "cost": None, "usage": {"prompt_tokens": 10}}

        cost, cost_exact, currency = cost_from_generation("openrouter-tts", generation)

        assert cost == 0.0
        assert cost_exact == "0"
        assert currency == "USD"

    def test_missing_total_cost_falls_back_to_cost(self):
        cost, cost_exact, currency = cost_from_generation("openrouter-tts", {"cost": "0.0009"})

        assert cost == 0.0009
        assert cost_exact == "0.0009"
        assert currency == "USD"

    def test_none_total_cost_falls_back_to_cost(self):
        cost, cost_exact, currency = cost_from_generation(
            "openrouter-tts", {"total_cost": None, "cost": 0.25}
        )

        assert cost == 0.25
        assert cost_exact is None
        assert currency == "USD"

    def test_none_total_cost_and_cost_fall_back_to_numeric_usage(self):
        cost, cost_exact, currency = cost_from_generation(
            "openrouter-tts", {"total_cost": None, "cost": None, "usage": 0.1}
        )

        assert cost == 0.1
        assert cost_exact is None
        assert currency == "USD"

    def test_zero_numeric_usage_wins_when_other_fields_are_absent(self):
        cost, cost_exact, currency = cost_from_generation(
            "openrouter-tts", {"total_cost": None, "cost": None, "usage": 0}
        )

        assert cost == 0.0
        assert cost_exact == "0"
        assert currency == "USD"

    def test_exact_string_value_reports_legacy_float_and_exact_string(self):
        cost, cost_exact, currency = cost_from_generation("openrouter-tts", {"total_cost": "0.001"})

        assert cost == 0.001
        assert cost_exact == "0.001"
        assert currency == "USD"


class TestCostFromGenerationGuards:
    def test_missing_generation_reports_unknown(self):
        assert cost_from_generation("polza-tts", None) == (None, None, None)
        assert cost_from_generation("openrouter-tts", None) == (None, None, None)

    def test_unknown_provider_reports_unknown(self):
        assert cost_from_generation("qwen-local", {"cost": 1.0}) == (None, None, None)


_COST_PROVIDERS = ["polza-chat-audio", "polza-tts", "openrouter-tts"]


def _cost_field(provider: str, value: object) -> dict:
    key = "clientCost" if provider in ("polza-chat-audio", "polza-tts") else "total_cost"
    return {key: value}


def _raw_json_response(body: str, status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.encoding = "utf-8"
    response._content = body.encode("utf-8")
    return response


class TestObservedCostValueGuards:
    """A malformed observed value is unknown, never an exception or a float guess."""

    @pytest.mark.parametrize("provider", _COST_PROVIDERS)
    @pytest.mark.parametrize("value", [True, False, {"amount": 1}, ["0.5"]])
    def test_bool_and_container_values_are_unknown(self, provider, value):
        assert cost_from_generation(provider, _cost_field(provider, value)) == (None, None, None)

    @pytest.mark.parametrize("provider", _COST_PROVIDERS)
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_float_values_are_unknown(self, provider, value):
        assert cost_from_generation(provider, _cost_field(provider, value)) == (None, None, None)

    @pytest.mark.parametrize("provider", _COST_PROVIDERS)
    @pytest.mark.parametrize("value", ["abc", "", "nan", "Infinity", "1e2x"])
    def test_malformed_string_values_are_unknown(self, provider, value):
        assert cost_from_generation(provider, _cost_field(provider, value)) == (None, None, None)

    @pytest.mark.parametrize("provider", _COST_PROVIDERS)
    @pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
    def test_non_finite_decimal_values_are_unknown(self, provider, value):
        assert cost_from_generation(provider, _cost_field(provider, value)) == (None, None, None)

    @pytest.mark.parametrize("provider", _COST_PROVIDERS)
    def test_overflowing_decimal_is_unknown_not_infinity(self, provider):
        assert cost_from_generation(provider, _cost_field(provider, Decimal("1e400"))) == (
            None,
            None,
            None,
        )

    @pytest.mark.parametrize("provider", _COST_PROVIDERS)
    def test_int_and_exact_string_are_exact_but_float_is_legacy_only(self, provider):
        currency = "USD" if provider == "openrouter-tts" else "RUB"

        assert cost_from_generation(provider, _cost_field(provider, 7)) == (7.0, "7", currency)
        assert cost_from_generation(provider, _cost_field(provider, "0.1234567890123456789")) == (
            0.1234567890123456789,
            "0.1234567890123456789",
            currency,
        )
        assert cost_from_generation(provider, _cost_field(provider, 0.5)) == (0.5, None, currency)

    def test_fallback_advances_only_on_none(self):
        assert cost_from_generation("polza-tts", {"clientCost": 0, "cost": 1.25}) == (
            0.0,
            "0",
            "RUB",
        )
        assert cost_from_generation("polza-tts", {"clientCost": None, "cost": "0.5"}) == (
            0.5,
            "0.5",
            "RUB",
        )
        assert cost_from_generation(
            "openrouter-tts", {"total_cost": None, "cost": None, "usage": 0.5}
        ) == (0.5, None, "USD")


class TestGenerationDetailParsesExactNumbers:
    """Both detail boundaries keep unquoted JSON numbers out of binary float."""

    def test_polza_detail_parses_unquoted_numbers_as_decimal(self, monkeypatch):
        body = (
            '{"id": "gen-1", "model": "polza/model", '
            '"clientCost": 0.1234567890123456789, "generationTimeMs": 1500, '
            '"usage": {"cost_rub": 0.00000000000000000001, "tokens": 3}}'
        )
        monkeypatch.setattr(
            pricing.requests, "get", lambda *args, **kwargs: _raw_json_response(body)
        )

        detail = pricing.fetch_polza_generation_detail("key", "gen-1")

        assert detail is not None
        assert isinstance(detail["clientCost"], Decimal)
        assert isinstance(detail["usage"]["cost_rub"], Decimal)
        assert cost_from_generation("polza-chat-audio", detail) == (
            0.1234567890123456789,
            "0.1234567890123456789",
            "RUB",
        )

    def test_openrouter_detail_parses_unquoted_numbers_as_decimal(self, monkeypatch):
        body = (
            '{"data": {"id": "or-1", "total_cost": 0.0001234567890123456789, '
            '"generationTimeMs": 1500, '
            '"usage": {"cost": 0.0000000000000001, "prompt_tokens": 5}}}'
        )
        monkeypatch.setattr(
            pricing.requests, "get", lambda *args, **kwargs: _raw_json_response(body)
        )

        detail = pricing.fetch_openrouter_generation_detail("key", "or-1")

        assert detail is not None
        assert isinstance(detail["total_cost"], Decimal)
        assert isinstance(detail["usage"]["cost"], Decimal)
        assert cost_from_generation("openrouter-tts", detail) == (
            0.0001234567890123456789,
            "0.0001234567890123456789",
            "USD",
        )

    def test_polza_detail_http_error_returns_none_without_parsing(self, monkeypatch):
        monkeypatch.setattr(
            pricing.requests, "get", lambda *args, **kwargs: _raw_json_response("oops", 500)
        )

        assert pricing.fetch_polza_generation_detail("key", "gen-1") is None


class TestDecimalObservedCost:
    """Field-level check that an exact cost string survives decimal input.

    Only ``cost_from_generation`` is covered here; the surrounding cost pipeline
    still uses floats and is out of scope for this regression.
    """

    def test_decimal_zero_wins_over_nonzero_fallback(self):
        generation = {"clientCost": Decimal("0"), "cost": 1.25}

        cost, cost_exact, currency = cost_from_generation("polza-tts", generation)

        assert cost == 0.0
        assert cost_exact == "0"
        assert currency == "RUB"

    def test_decimal_sum_reports_exact_string_not_float_artifact(self):
        generation = {"clientCost": Decimal("0.1") + Decimal("0.2")}

        cost, cost_exact, currency = cost_from_generation("polza-tts", generation)

        assert cost == 0.3
        assert cost_exact == "0.3"
        assert currency == "RUB"


def _chunk(
    number: int,
    generation_id: str | None,
    *,
    cost: float | None = None,
    cost_exact: str | None = None,
    currency: str | None = None,
) -> ChunkArtifact:
    return ChunkArtifact(
        number=number,
        id=f"chunk-{number}",
        file=f"chunk-{number}.mp3",
        duration_ms=1000,
        duration_sec=1.0,
        start_ms=0,
        end_ms=1000,
        text_characters=10,
        transcript="text",
        client_path="requests",
        generation_id=generation_id,
        cost=cost,
        cost_exact=cost_exact,
        cost_currency=currency,
        cost_rub=cost if currency == "RUB" else None,
        cost_rub_exact=cost_exact if currency == "RUB" else None,
    )


class TestAttachCostsPolzaGenerationIdBoundary:
    """Only chat-audio may bind a cost by its own remote generation id.

    chat-audio ids come exclusively from the response ``X-Generation-Id`` header,
    so a detail addressed by that exact id is provably the chunk's own generation.
    polza-tts ids mix media task ids and generic ``/audio/speech`` body ids with
    resumed chunk artifacts, so no history lookup can be trusted to bind them; the
    unrelated ``B1``/``B2`` run must never supply costs for ``A1``/``A2``.
    """

    def test_chat_audio_own_generation_ids_map_per_chunk(self, monkeypatch):
        chunks = [_chunk(1, "A1"), _chunk(2, "A2")]
        requested: list[str | None] = []
        details = {
            "A1": {"id": "A1", "clientCost": "0.11"},
            "A2": {"id": "A2", "clientCost": "0.22"},
        }

        def fake_detail(api_key, generation_id):
            requested.append(generation_id)
            return details.get(generation_id)

        monkeypatch.setattr(cli, "fetch_polza_generation_detail", fake_detail)
        monkeypatch.setattr(
            cli,
            "fetch_polza_generation_costs",
            lambda *args, **kwargs: [
                {"id": "B2", "clientCost": 6.0},
                {"id": "B1", "clientCost": 5.0},
            ],
            raising=False,
        )

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert requested == ["A1", "A2"]
        assert [chunk.generation_id for chunk in result] == ["A1", "A2"]
        assert [chunk.cost for chunk in result] == [0.11, 0.22]
        assert [chunk.cost_exact for chunk in result] == ["0.11", "0.22"]
        assert [chunk.cost_currency for chunk in result] == ["RUB", "RUB"]

    @pytest.mark.parametrize(
        "generation_id",
        [
            "media-task-elevenlabs",
            "speech-openai-body-id",
            "legacy-resumed-body-id",
        ],
    )
    def test_polza_tts_never_looks_up_history(self, monkeypatch, generation_id):
        chunks = [
            _chunk(1, generation_id, cost=0.75, cost_exact="0.75", currency="RUB"),
            _chunk(2, generation_id, cost=0.0, cost_exact="0", currency="RUB"),
            _chunk(3, generation_id),
        ]
        calls: list[str | None] = []

        def fake_detail(api_key, requested_id):
            calls.append(requested_id)
            return {"id": requested_id, "clientCost": 9.0}

        monkeypatch.setattr(cli, "fetch_polza_generation_detail", fake_detail)

        result = cli.attach_costs("polza-tts", "test-key", "polza/model", None, chunks)

        assert calls == []
        assert [chunk.generation_id for chunk in result] == [generation_id] * 3
        assert [chunk.cost for chunk in result] == [0.75, 0.0, None]
        assert [chunk.cost_exact for chunk in result] == ["0.75", "0", None]
        assert [chunk.cost_currency for chunk in result] == ["RUB", "RUB", None]

    def test_chat_audio_missing_generation_id_skips_lookup_and_preserves_direct_cost(
        self, monkeypatch
    ):
        chunks = [_chunk(1, None, cost=0.5, cost_exact="0.5", currency="RUB")]
        calls: list[tuple] = []
        monkeypatch.setattr(
            cli, "fetch_polza_generation_detail", lambda *args: calls.append(args) or None
        )

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert calls == []
        assert result[0].generation_id is None
        assert result[0].cost == 0.5
        assert result[0].cost_exact == "0.5"

    @pytest.mark.parametrize("direct_cost", [0.5, None])
    def test_chat_audio_missing_detail_preserves_existing_cost(self, monkeypatch, direct_cost):
        chunks = [
            _chunk(
                1,
                "A1",
                cost=direct_cost,
                cost_exact="0.5" if direct_cost is not None else None,
                currency="RUB" if direct_cost is not None else None,
            )
        ]
        monkeypatch.setattr(cli, "fetch_polza_generation_detail", lambda *args: None)

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert result[0].cost == direct_cost
        assert result[0].generation_id == "A1"

    def test_chat_audio_mismatched_declared_id_is_not_assigned(self, monkeypatch):
        chunks = [_chunk(1, "A1", cost=0.5, cost_exact="0.5", currency="RUB")]
        monkeypatch.setattr(
            cli,
            "fetch_polza_generation_detail",
            lambda *args: {"id": "B1", "clientCost": 9.0},
        )

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert result[0].generation_id == "A1"
        assert result[0].cost == 0.5
        assert result[0].cost_exact == "0.5"

    def test_chat_audio_mismatched_model_is_not_assigned(self, monkeypatch):
        chunks = [_chunk(1, "A1", cost=0.5, cost_exact="0.5", currency="RUB")]
        monkeypatch.setattr(
            cli,
            "fetch_polza_generation_detail",
            lambda *args: {"id": "A1", "model": "other/model", "clientCost": 9.0},
        )

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert result[0].cost == 0.5
        assert result[0].generation_id == "A1"

    @pytest.mark.parametrize("direct_cost", [0.5, None])
    def test_chat_audio_detail_without_id_is_not_assigned(self, monkeypatch, direct_cost):
        chunks = [
            _chunk(
                1,
                "A1",
                cost=direct_cost,
                cost_exact="0.5" if direct_cost is not None else None,
                currency="RUB" if direct_cost is not None else None,
            )
        ]
        monkeypatch.setattr(
            cli, "fetch_polza_generation_detail", lambda *args: {"clientCost": 0.75}
        )

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert result[0].cost == direct_cost
        assert result[0].generation_id == "A1"

    def test_chat_audio_zero_cost_with_matching_id_stays_zero(self, monkeypatch):
        chunks = [_chunk(1, "A1")]
        monkeypatch.setattr(
            cli,
            "fetch_polza_generation_detail",
            lambda *args: {"id": "A1", "clientCost": 0},
        )

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert result[0].cost == 0.0
        assert result[0].cost_exact == "0"
        assert result[0].cost_currency == "RUB"

    def test_chat_audio_detail_without_cost_does_not_erase_direct_cost(self, monkeypatch):
        chunks = [_chunk(1, "A1", cost=0.5, cost_exact="0.5", currency="RUB")]
        monkeypatch.setattr(
            cli,
            "fetch_polza_generation_detail",
            lambda *args: {"id": "A1", "usage": {"tokens": 3}},
        )

        result = cli.attach_costs("polza-chat-audio", "test-key", "polza/model", None, chunks)

        assert result[0].cost == 0.5
        assert result[0].cost_exact == "0.5"
        assert result[0].cost_currency == "RUB"


def test_openrouter_path_still_uses_per_chunk_generation_detail(monkeypatch):
    chunks = [_chunk(1, "or-1")]
    requested: list[str | None] = []
    monkeypatch.setattr(
        cli,
        "fetch_openrouter_generation_detail",
        lambda api_key, generation_id: requested.append(generation_id) or {"total_cost": 0.002},
    )

    result = cli.attach_costs("openrouter-tts", "test-key", "openrouter/model", None, chunks)

    assert requested == ["or-1"]
    assert result[0].cost == 0.002
    assert result[0].cost_currency == "USD"


class TestAttachCostsOpenRouterGenerationIdBoundary:
    """An OpenRouter detail may only charge the chunk whose id was requested.

    The detail GET is targeted by ``params={"id": chunk.generation_id}``, so a
    response that declares a different ``id`` belongs to another generation and
    must never supply a cost, replace the chunk's generation id, or erase an
    already trusted direct/state-derived cost. A detail without a declared id
    stays allowed because the request itself was addressed by the chunk id.
    """

    def test_foreign_declared_id_never_charges_or_renames(self, monkeypatch):
        chunks = [_chunk(1, "A1", cost=0.5, cost_exact="0.5", currency="USD")]
        monkeypatch.setattr(
            cli,
            "fetch_openrouter_generation_detail",
            lambda *args: {"id": "B1", "total_cost": 9.0},
        )

        result = cli.attach_costs("openrouter-tts", "test-key", "openrouter/model", None, chunks)

        assert result[0].__dict__ == chunks[0].__dict__
        assert result[0].generation_id == "A1"
        assert result[0].cost == 0.5
        assert result[0].cost_exact == "0.5"
        assert result[0].cost_currency == "USD"

    def test_foreign_declared_id_leaves_uncosted_chunk_unknown(self, monkeypatch):
        chunks = [_chunk(1, "A1")]
        monkeypatch.setattr(
            cli,
            "fetch_openrouter_generation_detail",
            lambda *args: {"id": "B1", "total_cost": 9.0},
        )

        result = cli.attach_costs("openrouter-tts", "test-key", "openrouter/model", None, chunks)

        assert result[0].__dict__ == chunks[0].__dict__
        assert result[0].generation_id == "A1"
        assert result[0].cost is None
        assert result[0].cost_exact is None
        assert result[0].cost_currency is None

    def test_missing_declared_id_keeps_targeted_lookup_contract(self, monkeypatch):
        chunks = [_chunk(1, "A1")]
        monkeypatch.setattr(
            cli,
            "fetch_openrouter_generation_detail",
            lambda *args: {"total_cost": "0.002"},
        )

        result = cli.attach_costs("openrouter-tts", "test-key", "openrouter/model", None, chunks)

        assert result[0].generation_id == "A1"
        assert result[0].cost == 0.002
        assert result[0].cost_exact == "0.002"
        assert result[0].cost_currency == "USD"

    def test_missing_generation_id_skips_lookup_and_preserves_cost(self, monkeypatch):
        chunks = [_chunk(1, None, cost=0.5, cost_exact="0.5", currency="USD")]
        calls: list[tuple] = []
        monkeypatch.setattr(
            cli,
            "fetch_openrouter_generation_detail",
            lambda *args: calls.append(args) or None,
        )
        monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

        result = cli.attach_costs("openrouter-tts", "test-key", "openrouter/model", None, chunks)

        assert calls == []
        assert result[0].generation_id is None
        assert result[0].cost == 0.5
        assert result[0].cost_exact == "0.5"
        assert result[0].cost_currency == "USD"

    def test_unavailable_detail_preserves_resumed_cost_metadata(self, monkeypatch):
        chunks = [_chunk(1, "A1", cost=0.125, cost_exact="0.125", currency="USD")]
        monkeypatch.setattr(cli, "fetch_openrouter_generation_detail", lambda *args: None)
        monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

        result = cli.attach_costs("openrouter-tts", "test-key", "openrouter/model", None, chunks)

        assert result[0].__dict__ == chunks[0].__dict__
        assert result[0].cost == 0.125
        assert result[0].cost_exact == "0.125"
        assert result[0].cost_currency == "USD"


class TestAttachCostsStrictJsonMetadata:
    """Decimal detail metadata is projected before it can reach a manifest."""

    def test_chat_audio_decimal_usage_and_timing_are_projected(self, monkeypatch):
        chunks = [_chunk(1, "A1")]
        detail = {
            "id": "A1",
            "model": "polza/model",
            "clientCost": Decimal("0.1000000000000000055511151231257827"),
            "generationTimeMs": Decimal("1500.5"),
            "usage": {"cost_rub": Decimal("0.2"), "tokens": 3, "nested": [Decimal("0.3")]},
            "createdAt": "2026-01-01T00:00:00Z",
        }
        monkeypatch.setattr(cli, "fetch_polza_generation_detail", lambda *args: detail)

        result = cli.attach_costs("polza-chat-audio", "key", "polza/model", None, chunks)

        chunk = result[0]
        assert chunk.cost_exact == "0.1000000000000000055511151231257827"
        assert chunk.usage == {"cost_rub": 0.2, "tokens": 3, "nested": [0.3]}
        assert chunk.generation_time_ms == 1500.5
        assert not isinstance(chunk.usage["cost_rub"], Decimal)
        assert not isinstance(chunk.usage["nested"][0], Decimal)
        assert not isinstance(chunk.generation_time_ms, Decimal)
        json.dumps(chunk.__dict__, allow_nan=False)

    def test_chat_audio_non_finite_metadata_becomes_null(self, monkeypatch):
        chunks = [_chunk(1, "A1")]
        detail = {
            "id": "A1",
            "clientCost": 0.5,
            "generationTimeMs": float("inf"),
            "usage": {"score": float("nan")},
        }
        monkeypatch.setattr(cli, "fetch_polza_generation_detail", lambda *args: detail)

        result = cli.attach_costs("polza-chat-audio", "key", "polza/model", None, chunks)

        assert result[0].usage == {"score": None}
        assert result[0].generation_time_ms is None
        json.dumps(result[0].__dict__, allow_nan=False)

    def test_openrouter_decimal_usage_and_timing_are_projected(self, monkeypatch):
        chunks = [_chunk(1, "or-1")]
        detail = {
            "id": "or-1",
            "total_cost": Decimal("0.002"),
            "generationTimeMs": Decimal("1500.5"),
            "usage": {"cost": Decimal("0.002"), "prompt_tokens": 5},
            "createdAt": "2026-01-01T00:00:00Z",
        }
        monkeypatch.setattr(cli, "fetch_openrouter_generation_detail", lambda *args: detail)

        result = cli.attach_costs("openrouter-tts", "key", "openrouter/model", None, chunks)

        chunk = result[0]
        assert chunk.cost == 0.002
        assert chunk.cost_exact == "0.002"
        assert chunk.usage == {"cost": 0.002, "prompt_tokens": 5}
        assert chunk.generation_time_ms == 1500.5
        assert not isinstance(chunk.usage["cost"], Decimal)
        json.dumps(chunk.__dict__, allow_nan=False)


def test_summarize_costs_reports_direct_polza_tts_source_without_history():
    priced = [_chunk(1, "media-task-id", cost=0.0, cost_exact="0", currency="RUB")]

    total, total_exact, currency, source = cli.summarize_costs("polza-tts", priced)

    assert total == 0.0
    assert total_exact == "0"
    assert currency == "RUB"
    assert source == "Polza API usage.cost_rub (direct)"

    unknown = [_chunk(2, "media-task-id")]
    assert cli.summarize_costs("polza-tts", unknown) == (None, None, None, None)


class TestSummarizeCostsCanonicalTotal:
    """The canonical total is a Decimal sum of observed exact cost strings.

    Legacy float fields stay compatible, but must never become a pseudo-exact
    total, and a mixed or missing currency must not be summed as if it were one.
    """

    def test_exact_cost_strings_sum_to_the_decimal_total(self):
        chunks = [
            _chunk(1, "a", cost=0.1, cost_exact="0.1", currency="RUB"),
            _chunk(2, "b", cost=0.2, cost_exact="0.2", currency="RUB"),
        ]

        total, total_exact, currency, _source = cli.summarize_costs("polza-tts", chunks)

        assert total == 0.3
        assert total_exact == "0.3"
        assert currency == "RUB"

    def test_high_precision_exact_total_is_not_rounded_to_a_float_total(self):
        chunks = [
            _chunk(
                1,
                "a",
                cost=0.1234567890123456789,
                cost_exact="0.1234567890123456789",
                currency="RUB",
            ),
            _chunk(2, "b", cost=0.2, cost_exact="0.2", currency="RUB"),
        ]

        _total, total_exact, _currency, _source = cli.summarize_costs("polza-tts", chunks)

        assert total_exact == "0.3234567890123456789"

    def test_exact_total_keeps_digits_beyond_the_default_decimal_precision(self):
        chunks = [
            _chunk(
                1,
                "a",
                cost=0.12345678901234567890123456789,
                cost_exact="0.12345678901234567890123456789",
                currency="RUB",
            ),
            _chunk(2, "b", cost=0.2, cost_exact="0.2", currency="RUB"),
        ]

        _total, total_exact, _currency, _source = cli.summarize_costs("polza-tts", chunks)

        assert total_exact == "0.32345678901234567890123456789"

    def test_observed_zero_exact_string_makes_a_canonical_zero(self):
        chunks = [_chunk(1, "a", cost=0.0, cost_exact="0", currency="RUB")]

        total, total_exact, _currency, _source = cli.summarize_costs("polza-tts", chunks)

        assert total == 0.0
        assert total_exact == "0"

    @pytest.mark.parametrize("cost_exact", [None, "abc", "nan", "inf", ""])
    def test_unusable_exact_keeps_float_total_without_canonical_total(self, cost_exact):
        chunks = [_chunk(1, "a", cost=0.5, cost_exact=cost_exact, currency="RUB")]

        total, total_exact, currency, _source = cli.summarize_costs("polza-tts", chunks)

        assert total == 0.5
        assert total_exact is None
        assert currency == "RUB"

    @pytest.mark.parametrize(
        "cost_exact", [0.5, 0.1, True, False, ["0.5"], {"v": "0.5"}, Decimal("0.5")]
    )
    def test_non_string_exact_is_not_canonical_and_never_raises(self, cost_exact):
        chunks = [_chunk(1, "a", cost=0.5, cost_exact=cost_exact, currency="RUB")]

        total, total_exact, currency, _source = cli.summarize_costs("polza-tts", chunks)

        assert total == 0.5
        assert total_exact is None
        assert currency == "RUB"

    def test_mixed_currency_fails_closed_to_no_total(self):
        chunks = [
            _chunk(1, "a", cost=0.5, cost_exact="0.5", currency="RUB"),
            _chunk(2, "b", cost=0.5, cost_exact="0.5", currency="USD"),
        ]

        assert cli.summarize_costs("polza-tts", chunks) == (None, None, None, None)

    def test_missing_currency_fails_closed_to_no_total(self):
        chunks = [_chunk(1, "a", cost=0.5, cost_exact="0.5")]

        assert cli.summarize_costs("polza-tts", chunks) == (None, None, None, None)


def test_polza_tts_speech_raw_response_direct_cost_is_exact_end_to_end(monkeypatch):
    """A raw speech body's unquoted cost reaches the canonical total exactly."""
    from voiceover_pipeline.providers import polza_tts
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider

    audio_b64 = base64.b64encode(b"fake").decode()
    body = (
        '{"audio": "' + audio_b64 + '", "contentType": "audio/mpeg", '
        '"usage": {"cost_rub": 0.1234567890123456789}}'
    )
    monkeypatch.setattr(
        polza_tts.requests, "post", lambda *args, **kwargs: _raw_json_response(body)
    )

    provider = PolzaTTSProvider(api_key="k", model="openai/gpt-4o-mini-tts", voice="ash")
    result = provider.synthesize_chunk("text", "chunk_01")
    kwargs = cli._direct_cost_kwargs("polza-tts", result)
    artifact = ChunkArtifact(
        number=1,
        id="chunk_01",
        file="chunk_01.mp3",
        duration_ms=1000,
        duration_sec=1.0,
        start_ms=0,
        end_ms=1000,
        text_characters=4,
        transcript=None,
        client_path="requests",
        generation_id=None,
        **kwargs,
    )

    total, total_exact, currency, source = cli.summarize_costs("polza-tts", [artifact])

    assert total_exact == "0.1234567890123456789"
    assert currency == "RUB"
    assert source == "Polza API usage.cost_rub (direct)"


class TestDirectCostProjectionService:
    """The extracted ``services.costs`` keeps the CLI's exact cost projection."""

    def test_direct_cost_kwargs_matches_legacy_values(self):
        from voiceover_pipeline.services import costs

        result = SimpleNamespace(
            raw_metadata={"usage_direct": {"cost_rub": Decimal("0.1234567890123456789")}}
        )
        kwargs = costs.direct_cost_kwargs("polza-tts", result)
        assert kwargs["cost"] == 0.1234567890123456789
        assert kwargs["cost_exact"] == "0.1234567890123456789"
        assert kwargs["generation_detail_source"] == "Polza API usage.cost_rub (direct)"

        assert costs.direct_cost_kwargs("openrouter-tts", result) == {}

    @pytest.mark.parametrize(
        ("value", "expected_cost", "expected_exact"),
        [
            (0, 0.0, "0"),
            ("0.1575", 0.1575, "0.1575"),
            (0.25, 0.25, None),
            ("bad", None, None),
            (True, None, None),
            ([0.5], None, None),
        ],
    )
    def test_direct_cost_kwargs_covers_zero_exact_string_and_unknown(
        self, value, expected_cost, expected_exact
    ):
        from voiceover_pipeline.services import costs

        result = SimpleNamespace(raw_metadata={"usage_direct": {"cost_rub": value}})
        kwargs = costs.direct_cost_kwargs("polza-tts", result)
        if expected_cost is None:
            assert kwargs == {}
        else:
            assert kwargs["cost"] == expected_cost
            assert kwargs["cost_exact"] == expected_exact
            assert kwargs["cost_currency"] == "RUB"

    def test_direct_cost_kwargs_projects_nonfinite_metadata_json_safe(self):
        from voiceover_pipeline.services import costs

        result = SimpleNamespace(
            raw_metadata={
                "usage_direct": {
                    "cost_rub": Decimal("0.1"),
                    "nested": {"inf": float("inf"), "nan": float("nan")},
                }
            }
        )
        kwargs = costs.direct_cost_kwargs("polza-tts", result)
        assert kwargs["usage"]["nested"] == {"inf": None, "nan": None}
        json.dumps(kwargs, allow_nan=False)

    def test_media_and_recovered_projection_helpers(self):
        from voiceover_pipeline.services import costs

        assert costs.media_observed_cost({"cost": "0.3"}) == (0.3, "0.3")
        assert costs.media_observed_cost(None) == (None, None)
        assert costs.recovered_attempt_cost_kwargs({"cost": True}) == {}
        recovered = costs.recovered_attempt_cost_kwargs({"cost": 0.5, "cost_exact": "0.5"})
        assert recovered["cost_exact"] == "0.5"
        assert recovered["generation_detail_source"] == "Polza API usage.cost_rub (direct)"


class TestCostEnrichmentService:
    """The extracted enrichment body keeps its id/model guards and injects lookups.

    ``services.cost_enrichment.attach_costs`` owns the history lookup body but
    receives both lookups explicitly, so ``cli.attach_costs`` can forward its
    currently bound (and monkeypatched) callables without importing ``cli``.
    """

    def test_polza_chat_audio_binds_own_id_and_guards_id_and_model(self, monkeypatch):
        from voiceover_pipeline.services import cost_enrichment

        monkeypatch.setattr(cost_enrichment.time, "sleep", lambda _seconds: None)
        chunks = [
            _chunk(1, "A1"),
            _chunk(2, "A2", cost=0.5, cost_exact="0.5", currency="RUB"),
            _chunk(3, "A3", cost=0.5, cost_exact="0.5", currency="RUB"),
            _chunk(4, "A4"),
        ]
        details = {
            "A1": {"id": "A1", "model": "polza/model", "clientCost": 0},
            "A2": {"id": "B2", "model": "polza/model", "clientCost": 9.0},
            "A3": {"id": "A3", "model": "other/model", "clientCost": 9.0},
            "A4": {"id": "A4", "model": "polza/model", "clientCost": "0.25"},
        }
        requested: list[str | None] = []

        def fetch_polza_detail(api_key, generation_id):
            requested.append(generation_id)
            return details.get(generation_id)

        result = cost_enrichment.attach_costs(
            "polza-chat-audio",
            "key",
            "polza/model",
            None,
            chunks,
            fetch_polza_detail=fetch_polza_detail,
            fetch_openrouter_detail=lambda *_args: None,
        )

        assert requested == ["A1", "A2", "A3", "A4"]
        assert [chunk.generation_id for chunk in result] == ["A1", "A2", "A3", "A4"]
        # Zero clientCost is a real fact; a foreign id/model keeps the direct cost.
        assert [chunk.cost for chunk in result] == [0.0, 0.5, 0.5, 0.25]
        assert [chunk.cost_exact for chunk in result] == ["0", "0.5", "0.5", "0.25"]
        assert [chunk.cost_currency for chunk in result] == ["RUB", "RUB", "RUB", "RUB"]
        assert result[1].__dict__ == chunks[1].__dict__
        assert result[2].__dict__ == chunks[2].__dict__

    def test_openrouter_foreign_or_missing_detail_preserves_direct_cost(self, monkeypatch):
        from voiceover_pipeline.services import cost_enrichment

        monkeypatch.setattr(cost_enrichment.time, "sleep", lambda _seconds: None)
        chunks = [
            _chunk(1, "A1", cost=0.5, cost_exact="0.5", currency="USD"),
            _chunk(2, "A2"),
        ]
        calls: list[str | None] = []

        def fetch_openrouter_detail(api_key, generation_id):
            calls.append(generation_id)
            if generation_id == "A1":
                return {"id": "B1", "total_cost": 9.0}
            return None

        result = cost_enrichment.attach_costs(
            "openrouter-tts",
            "key",
            "openrouter/model",
            None,
            chunks,
            fetch_polza_detail=lambda *_args: None,
            fetch_openrouter_detail=fetch_openrouter_detail,
        )

        # A foreign declared id is not retried; a missing detail is polled four times.
        assert calls == ["A1"] + ["A2"] * 4
        assert result[0].__dict__ == chunks[0].__dict__
        assert result[1].__dict__ == chunks[1].__dict__
        assert result[0].cost == 0.5
        assert result[0].cost_currency == "USD"

    def test_cli_wrapper_forwards_currently_bound_lookups(self, monkeypatch):
        chunks = [_chunk(1, "A1")]
        seen: list[tuple[str, str | None]] = []

        def fake_detail(api_key, generation_id):
            seen.append((api_key, generation_id))
            return {"id": "A1", "clientCost": "0.11"}

        monkeypatch.setattr(cli, "fetch_polza_generation_detail", fake_detail)

        result = cli.attach_costs("polza-chat-audio", "patched-key", "polza/model", None, chunks)

        assert seen == [("patched-key", "A1")]
        assert result[0].cost_exact == "0.11"


class TestCostSummaryAndStateMergeService:
    """``services.cost_enrichment`` owns the exact summary and the trusted-state merge.

    ``cli`` keeps thin wrappers so its exit envelope and existing call sites stay
    identical, while the merge binds an artifact by its own ``id`` and ``number``
    and a missing or ambiguous match fails closed instead of writing a cost onto
    the wrong state chunk.
    """

    def test_summarize_costs_service_owns_exact_decimal_total_and_unknown_fallback(self):
        from voiceover_pipeline.services import cost_enrichment

        chunks = [
            _chunk(1, "a", cost=0.1, cost_exact="0.1", currency="RUB"),
            _chunk(2, "b", cost=0.2, cost_exact="0.2", currency="RUB"),
        ]

        total, total_exact, currency, source = cost_enrichment.summarize_costs("polza-tts", chunks)

        assert (total, total_exact, currency) == (0.3, "0.3", "RUB")
        assert source == "Polza API usage.cost_rub (direct)"

        zero = [_chunk(1, "a", cost=0.0, cost_exact="0", currency="RUB")]
        assert cost_enrichment.summarize_costs("polza-tts", zero) == (
            0.0,
            "0",
            "RUB",
            "Polza API usage.cost_rub (direct)",
        )

        unknown = [_chunk(2, "b")]
        assert cost_enrichment.summarize_costs("polza-tts", unknown) == (None, None, None, None)

    def test_merge_attached_costs_service_binds_by_id_and_number_and_fails_closed(self):
        from copy import deepcopy

        from voiceover_pipeline.services import cost_enrichment

        state = {
            "chunks": [
                {
                    "status": "completed",
                    "id": "chunk-1",
                    "number": 1,
                    "cost": 0.5,
                    "cost_exact": "0.5",
                },
                {"status": "completed", "id": "chunk-2", "number": 2},
            ]
        }
        artifact = _chunk(1, "g", cost=0.25, cost_exact="0.25", currency="RUB")

        cost_enrichment.merge_attached_costs_into_state(state, [artifact])

        entry = state["chunks"][0]
        assert entry["cost"] == 0.25
        assert entry["cost_exact"] == "0.25"
        assert entry["cost_rub"] == 0.25
        assert entry["cost_rub_exact"] == "0.25"
        assert entry["cost_currency"] == "RUB"
        # A field the artifact does not carry is not written, and the unmatched
        # second entry stays byte-identical.
        assert "usage" not in entry
        assert state["chunks"][1] == {"status": "completed", "id": "chunk-2", "number": 2}

        for chunks in (
            [{"status": "running", "id": "chunk-1", "number": 1}],
            [{"status": "completed", "id": "chunk-1", "number": 2}],
            [
                {"status": "completed", "id": "chunk-1", "number": 1},
                {"status": "completed", "id": "chunk-1", "number": 1},
            ],
        ):
            mismatched = {"chunks": chunks}
            before = deepcopy(chunks)
            with pytest.raises(cost_enrichment.AttachedCostStateMismatchError) as exc_info:
                cost_enrichment.merge_attached_costs_into_state(mismatched, [artifact])
            assert str(exc_info.value) == (
                "Cannot persist observed costs: run state has no single completed "
                "chunk matching chunk-1/1."
            )
            assert mismatched["chunks"] == before

    def test_cli_merge_wrapper_delegates_to_service_and_keeps_provider_exit(self, monkeypatch):
        from voiceover_pipeline.services import cost_enrichment

        state = {"chunks": []}
        seen: list[tuple[dict, list]] = []

        def recording_merge(state_arg, artifacts_arg):
            seen.append((state_arg, artifacts_arg))

        monkeypatch.setattr(cost_enrichment, "merge_attached_costs_into_state", recording_merge)
        cli._merge_attached_costs_into_state(state, [])

        assert seen == [(state, [])]

        def mismatching_merge(state_arg, artifacts_arg):
            raise cost_enrichment.AttachedCostStateMismatchError(
                "Cannot persist observed costs: run state has no single completed "
                "chunk matching x/9."
            )

        monkeypatch.setattr(cost_enrichment, "merge_attached_costs_into_state", mismatching_merge)
        with pytest.raises(cli.CliError) as exc_info:
            cli._merge_attached_costs_into_state({"chunks": []}, [])

        assert exc_info.value.code == cli._EXIT_PROVIDER
        assert str(exc_info.value) == (
            "Cannot persist observed costs: run state has no single completed chunk matching x/9."
        )


class TestPricingSnapshotRouting:
    """``services.cost_enrichment.fetch_pricing_snapshot`` owns the provider routing.

    The CLI keeps a thin wrapper so ``cli.fetch_pricing_snapshot`` stays the
    patchable paid-preflight seam and forwards its currently bound (and
    monkeypatched) pricing lookups, but the provider-to-lookup decision itself
    lives in the service and performs no I/O on its own.
    """

    def test_service_routes_provider_to_injected_lookup_without_network(self):
        from voiceover_pipeline.services import cost_enrichment

        calls: list[tuple[str, str, str]] = []

        def fetch_polza_pricing(api_key, model):
            calls.append(("polza", api_key, model))
            return {"model": model}

        def fetch_openrouter_pricing(model):
            calls.append(("openrouter", model, model))
            return None

        assert cost_enrichment.fetch_pricing_snapshot(
            "polza-tts",
            "sk-test",
            "openai/gpt-4o-mini-tts",
            fetch_polza_pricing=fetch_polza_pricing,
            fetch_openrouter_pricing=fetch_openrouter_pricing,
        ) == {"model": "openai/gpt-4o-mini-tts"}
        assert cost_enrichment.fetch_pricing_snapshot(
            "polza-chat-audio",
            "sk-test",
            "polza/model",
            fetch_polza_pricing=fetch_polza_pricing,
            fetch_openrouter_pricing=fetch_openrouter_pricing,
        ) == {"model": "polza/model"}
        assert (
            cost_enrichment.fetch_pricing_snapshot(
                "openrouter-tts",
                "sk-test",
                "openrouter/model",
                fetch_polza_pricing=fetch_polza_pricing,
                fetch_openrouter_pricing=fetch_openrouter_pricing,
            )
            is None
        )
        assert (
            cost_enrichment.fetch_pricing_snapshot(
                "qwen-local",
                "sk-test",
                "qwen/model",
                fetch_polza_pricing=fetch_polza_pricing,
                fetch_openrouter_pricing=fetch_openrouter_pricing,
            )
            is None
        )
        assert calls == [
            ("polza", "sk-test", "openai/gpt-4o-mini-tts"),
            ("polza", "sk-test", "polza/model"),
            ("openrouter", "openrouter/model", "openrouter/model"),
        ]

    def test_cli_wrapper_forwards_currently_bound_pricing_lookups(self, monkeypatch):
        polza_calls: list[tuple[str, str]] = []
        openrouter_calls: list[str] = []

        def fake_polza(api_key, model):
            polza_calls.append((api_key, model))
            return {"source": "polza"}

        def fake_openrouter(model):
            openrouter_calls.append(model)
            return {"source": "openrouter"}

        monkeypatch.setattr(cli, "fetch_polza_model_pricing", fake_polza)
        monkeypatch.setattr(cli, "fetch_openrouter_model_pricing", fake_openrouter)

        assert cli.fetch_pricing_snapshot("polza-tts", "sk-test", "m") == {"source": "polza"}
        assert cli.fetch_pricing_snapshot("openrouter-tts", "sk-test", "m") == {
            "source": "openrouter"
        }
        assert cli.fetch_pricing_snapshot("qwen-local", "sk-test", "m") is None
        assert polza_calls == [("sk-test", "m")]
        assert openrouter_calls == ["m"]
