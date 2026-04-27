from datetime import datetime
from typing import Any

import requests

from .config import OPENROUTER_BASE_URL, POLZA_BASE_URL


def fetch_polza_model_pricing(api_key: str, model: str) -> dict[str, Any] | None:
    response = requests.get(
        f"{POLZA_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if response.status_code >= 400:
        return None

    for item in response.json().get("data", []):
        if item.get("id") == model:
            pricing = item.get("top_provider", {}).get("pricing", {})
            return {
                "currency": pricing.get("currency"),
                "prompt_per_million": pricing.get("prompt_per_million"),
                "completion_per_million": pricing.get("completion_per_million"),
                "audio_per_million": pricing.get("audio_per_million"),
                "source": f"{POLZA_BASE_URL}/models",
                "model": model,
            }

    return None


def fetch_polza_generation_costs(
    api_key: str,
    model: str,
    date_from: datetime,
    expected_count: int,
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{POLZA_BASE_URL}/history/generations",
        headers={"Authorization": f"Bearer {api_key}"},
        params={
            "limit": min(max(expected_count * 2, expected_count), 100),
            "page": 1,
            "dateFrom": date_from.isoformat().replace("+00:00", "Z"),
            "sortBy": "createdAt",
            "sortOrder": "asc",
        },
        timeout=60,
    )
    if response.status_code >= 400:
        return []

    items = [
        item
        for item in response.json().get("items", [])
        if item.get("model") == model and item.get("status") == "completed"
    ]

    detailed_items = []
    for item in items[:expected_count]:
        detail = fetch_polza_generation_detail(api_key, item.get("id"))
        detailed_items.append(detail or item)

    return detailed_items


def fetch_polza_generation_detail(api_key: str, generation_id: str | None) -> dict[str, Any] | None:
    if not generation_id:
        return None

    response = requests.get(
        f"{POLZA_BASE_URL}/history/generations/{generation_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if response.status_code >= 400:
        return None

    return response.json()


def fetch_openrouter_model_pricing(model: str) -> dict[str, Any] | None:
    response = requests.get(
        f"{OPENROUTER_BASE_URL}/models?output_modalities=speech",
        timeout=60,
    )
    if response.status_code >= 400:
        return None

    for item in response.json().get("data", []):
        if item.get("id") == model:
            return {
                "currency": "USD",
                "prompt": item.get("pricing", {}).get("prompt"),
                "completion": item.get("pricing", {}).get("completion"),
                "source": f"{OPENROUTER_BASE_URL}/models?output_modalities=speech",
                "model": model,
            }

    return None


def fetch_openrouter_generation_detail(api_key: str, generation_id: str | None) -> dict[str, Any] | None:
    if not generation_id:
        return None

    response = requests.get(
        f"{OPENROUTER_BASE_URL}/generation",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"id": generation_id},
        timeout=60,
    )
    if response.status_code >= 400:
        return None

    return response.json().get("data")


def cost_from_generation(provider: str, generation: dict[str, Any] | None) -> tuple[float | None, str | None, str | None]:
    if not generation:
        return None, None, None

    if provider == "polza-chat-audio":
        value = generation.get("clientCost") or generation.get("cost")
        return (float(value), str(value), "RUB") if value is not None else (None, None, None)

    if provider == "openrouter-tts":
        value = generation.get("total_cost") or generation.get("cost") or generation.get("usage")
        return (float(value), str(value), "USD") if value is not None else (None, None, None)

    return None, None, None
