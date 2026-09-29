"""Regression tests for observed-cost handling in the pricing pipeline.

``cost_from_generation`` treats an observed zero cost (``0``/``0.0``/``Decimal("0")``)
as a real billing fact, not a missing value, so it wins over any fallback field.
``cli.attach_costs`` may bind a Polza cost by the chunk's own known remote
generation id only for ``polza-chat-audio``, whose id comes exclusively from the
response ``X-Generation-Id`` header. ``polza-tts`` ids mix media task ids and
generic ``/audio/speech`` body ids with resumed chunk artifacts, so they are not
comparable to history ids and must never trigger a history lookup; those chunks
keep their direct cost and generation id untouched. No lookup may erase an
existing direct cost. These tests cover the field-level selection and the Polza
id boundary only; the surrounding pipeline still uses floats and is out of scope
here.
"""

from decimal import Decimal

import pytest

from voiceover_pipeline import cli
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
        assert cost_exact == "0.0"
        assert currency == "RUB"

    def test_missing_client_cost_falls_back_to_cost(self):
        generation = {"cost": 0.5}

        cost, cost_exact, currency = cost_from_generation("polza-tts", generation)

        assert cost == 0.5
        assert cost_exact == "0.5"
        assert currency == "RUB"

    def test_none_client_cost_falls_back_to_cost(self):
        generation = {"clientCost": None, "cost": 0.5}

        cost, cost_exact, currency = cost_from_generation("polza-tts", generation)

        assert cost == 0.5
        assert cost_exact == "0.5"
        assert currency == "RUB"

    def test_no_cost_fields_reports_unknown(self):
        assert cost_from_generation("polza-tts", {}) == (None, None, None)

    def test_legacy_float_and_exact_string_fields_are_preserved(self):
        cost, cost_exact, currency = cost_from_generation("polza-tts", {"clientCost": 0.1575})

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
        assert cost_exact == "0.25"
        assert currency == "USD"

    def test_none_total_cost_and_cost_fall_back_to_numeric_usage(self):
        cost, cost_exact, currency = cost_from_generation(
            "openrouter-tts", {"total_cost": None, "cost": None, "usage": 0.1}
        )

        assert cost == 0.1
        assert cost_exact == "0.1"
        assert currency == "USD"

    def test_zero_numeric_usage_wins_when_other_fields_are_absent(self):
        cost, cost_exact, currency = cost_from_generation(
            "openrouter-tts", {"total_cost": None, "cost": None, "usage": 0}
        )

        assert cost == 0.0
        assert cost_exact == "0"
        assert currency == "USD"

    def test_legacy_float_and_exact_string_fields_are_preserved(self):
        cost, cost_exact, currency = cost_from_generation("openrouter-tts", {"total_cost": 0.001})

        assert cost == 0.001
        assert cost_exact == "0.001"
        assert currency == "USD"


class TestCostFromGenerationGuards:
    def test_missing_generation_reports_unknown(self):
        assert cost_from_generation("polza-tts", None) == (None, None, None)
        assert cost_from_generation("openrouter-tts", None) == (None, None, None)

    def test_unknown_provider_reports_unknown(self):
        assert cost_from_generation("qwen-local", {"cost": 1.0}) == (None, None, None)


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
            "A1": {"id": "A1", "clientCost": 0.11},
            "A2": {"id": "A2", "clientCost": 0.22},
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


def test_summarize_costs_reports_direct_polza_tts_source_without_history():
    priced = [_chunk(1, "media-task-id", cost=0.0, cost_exact="0", currency="RUB")]

    total, total_exact, currency, source = cli.summarize_costs("polza-tts", priced)

    assert total == 0.0
    assert total_exact == "0.0"
    assert currency == "RUB"
    assert source == "Polza API usage.cost_rub (direct)"

    unknown = [_chunk(2, "media-task-id")]
    assert cli.summarize_costs("polza-tts", unknown) == (None, None, None, None)
