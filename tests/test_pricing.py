"""Regression tests for observed-cost precedence in ``pricing.cost_from_generation``.

An observed zero cost (``0``/``0.0``/``Decimal("0")``) is a real billing fact, not a
missing value: it must win over any fallback field. These tests cover the field-level
selection only; they do not claim the whole cost pipeline is Decimal-based.
"""

from decimal import Decimal

import pytest

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
