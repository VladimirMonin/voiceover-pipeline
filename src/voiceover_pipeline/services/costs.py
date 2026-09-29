"""Pure direct and observed cost projection for the CLI entry point.

These helpers only reshape values a provider already reported or a recovery
marker already stored: they never call a provider, touch run state, or reach the
network. ``cli`` keeps thin wrappers under the former names so existing imports
and outputs stay identical. Canonical pricing logic and the history lookup stay
in ``pricing`` and ``cli``.
"""

import math
from decimal import Decimal
from typing import Any

from ..pricing import observed_cost


def json_safe_metadata(value: Any) -> Any:
    """Project provider metadata to strict-JSON-safe legacy values.

    The history detail boundaries parse JSON with ``parse_float=Decimal`` so that
    observed costs stay exact, but that conversion also applies to every other
    JSON float in the same payload, and a non-finite ``Infinity``/``NaN`` constant
    can still arrive. Manifests and run state use ``json.dumps`` and must stay
    serializable, so a ``Decimal`` or non-finite number is projected here to a
    finite legacy float or ``None``; ``str``/``int``/``bool`` and the list/dict
    shape are preserved.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        legacy = float(value)
        return legacy if math.isfinite(legacy) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_metadata(item) for item in value]
    return value


def polza_direct_cost_kwargs(cost: float, cost_exact: str | None) -> dict:
    """Build the Polza direct-cost fields from one observed value pair."""
    return {
        "cost": cost,
        "cost_exact": cost_exact,
        "cost_currency": "RUB",
        "cost_rub": cost,
        "cost_rub_exact": cost_exact,
        "generation_detail_source": "Polza API usage.cost_rub (direct)",
    }


def recovered_attempt_cost_kwargs(recovery: dict[str, Any]) -> dict:
    """Reuse the exact cost the recovered paid attempt already reported.

    A recovered chunk (a saved raw file or a GET-only media task) can read a
    payload without usage, so the bounded cost stored in the attempt marker
    before the recovery stays the observed amount for this chunk.
    """
    cost = recovery.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return {}
    return polza_direct_cost_kwargs(float(cost), recovery.get("cost_exact"))


def media_observed_cost(usage: Any) -> tuple[float | None, str | None]:
    """Project the completed media payload's billed cost through ``observed_cost``.

    Only the recognized cost fields are read; a missing or unparseable value is
    unknown, so an arbitrary usage field never becomes a cost.
    """
    if not isinstance(usage, dict):
        return None, None
    value = usage.get("cost_rub")
    if value is None:
        value = usage.get("cost")
    if value is None:
        return None, None
    return observed_cost(value)


def direct_cost_kwargs(provider: str, result: Any) -> dict:
    """Project a ``polza-tts`` result's direct usage into canonical cost fields."""
    if provider != "polza-tts":
        return {}
    usage = (result.raw_metadata or {}).get("usage_direct")
    if not isinstance(usage, dict):
        return {}
    cost_rub = usage.get("cost_rub")
    if cost_rub is None:
        cost_rub = usage.get("cost")
    if cost_rub is None:
        return {}
    # ``observed_cost`` keeps an exact value string and refuses to invent one from
    # a binary float, so the direct cost obeys the same rule as the history path.
    cost, cost_exact = observed_cost(cost_rub)
    if cost is None:
        return {}
    return {
        **polza_direct_cost_kwargs(cost, cost_exact),
        # Project the copied usage only after the exact cost was extracted, so a
        # Decimal from ``parse_float=Decimal`` never reaches run state/manifests.
        "usage": json_safe_metadata(usage),
    }
