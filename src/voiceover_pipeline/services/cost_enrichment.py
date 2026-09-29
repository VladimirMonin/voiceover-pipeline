"""Provider-side history cost enrichment for the CLI entry point.

This module owns the post-generation lookup that binds an already-saved chunk
artifact to the cost its own provider generation reported: a Polza chat-audio
``X-Generation-Id`` history GET or a targeted OpenRouter generation GET by the
chunk's own body id. It performs no submit and no other provider request.

Both history lookups are injected as explicit keyword dependencies, so
``cli.attach_costs`` can forward its currently bound callables and existing
monkeypatch targets keep steering this body without importing ``cli``.
"""

import time

from ..models import ChunkArtifact
from ..pricing import cost_from_generation
from . import costs


def generation_source(provider: str) -> str:
    """Return the provenance label for a provider's observed cost source."""
    return {
        "polza-chat-audio": "Polza GET /api/v1/history/generations/{id}",
        "polza-tts": "Polza API usage.cost_rub (direct)",
        "openrouter-tts": "OpenRouter GET /api/v1/generation?id=...",
        "qwen-local": "qwen-local (free)",
        "omnivoice-local": "omnivoice-local (local model; no billing request)",
    }.get(provider, "unknown")


def attach_costs(
    provider,
    api_key,
    model,
    run_started_at,
    chunks,
    *,
    fetch_polza_detail,
    fetch_openrouter_detail,
):
    """Bind each saved chunk to the cost its own provider generation reported."""
    if provider in {"qwen-local", "omnivoice-local"}:
        enriched = []
        for chunk in chunks:
            enriched.append(
                ChunkArtifact(
                    **{**chunk.__dict__, "cost": 0.0, "cost_exact": "0.0", "cost_currency": "RUB"}
                )
            )
        return enriched
    if provider == "polza-tts":
        # polza-tts generation ids are not comparable to /history/generations ids:
        # they mix media submit/poll task ids and generic /audio/speech body ids,
        # and a resumed ChunkArtifact can carry an old ambiguous id. A history
        # lookup therefore cannot be bound to the chunk that produced it, so keep
        # the direct cost (including zero) for provenance and never fetch here.
        return chunks
    if provider == "polza-chat-audio":
        # chat-audio ids come exclusively from the response X-Generation-Id
        # header, so the detail addressed by that exact id is the chunk's own
        # generation. A detail without a matching explicit id, or one declaring a
        # different model, is foreign: keep the chunk's existing direct cost
        # untouched. A missing id, missing detail, or a payload without a cost
        # likewise preserves the existing direct cost and generation id.
        enriched = []
        for chunk in chunks:
            generation = (
                fetch_polza_detail(api_key, chunk.generation_id) if chunk.generation_id else None
            )
            if generation is not None:
                reported_id = generation.get("id")
                reported_model = generation.get("model")
                if (
                    reported_id is None
                    or str(reported_id) != chunk.generation_id
                    or (reported_model is not None and reported_model != model)
                ):
                    generation = None
            if generation is None:
                enriched.append(chunk)
                continue
            cost, cost_exact, currency = cost_from_generation(provider, generation)
            if cost is None:
                enriched.append(chunk)
                continue
            enriched.append(
                ChunkArtifact(
                    **{
                        **chunk.__dict__,
                        "cost_rub": cost if currency == "RUB" else None,
                        "cost_rub_exact": cost_exact if currency == "RUB" else None,
                        "cost": cost,
                        "cost_exact": cost_exact,
                        "cost_currency": currency,
                        "usage": costs.json_safe_metadata(generation.get("usage")),
                        "generation_time_ms": costs.json_safe_metadata(
                            generation.get("generationTimeMs") or generation.get("generation_time")
                        ),
                        "generated_at": generation.get("createdAt") or generation.get("created_at"),
                        "generation_detail_source": generation_source(provider),
                    }
                )
            )
        return enriched
    # openrouter-tts ids come from the response body and the detail GET is
    # targeted by ``params={"id": chunk.generation_id}``, so a detail that
    # declares a different id belongs to another generation: it must never
    # supply a cost or replace this chunk's generation id. A missing id stays
    # allowed because the request itself was addressed by the chunk id. A
    # missing/errored lookup or a payload without a cost keeps the artifact
    # (including any direct or state-derived cost) untouched, and a chunk with
    # no generation id never hits the network.
    enriched = []
    for chunk in chunks:
        detail = None
        if chunk.generation_id:
            for _ in range(4):
                detail = fetch_openrouter_detail(api_key, chunk.generation_id)
                if detail:
                    break
                time.sleep(3)
        if detail is not None:
            reported_id = detail.get("id")
            if reported_id is not None and str(reported_id) != chunk.generation_id:
                detail = None
        if detail is None:
            enriched.append(chunk)
            continue
        cost, cost_exact, currency = cost_from_generation(provider, detail)
        if cost is None:
            enriched.append(chunk)
            continue
        enriched.append(
            ChunkArtifact(
                **{
                    **chunk.__dict__,
                    "cost_rub": cost if currency == "RUB" else None,
                    "cost_rub_exact": cost_exact if currency == "RUB" else None,
                    "cost": cost,
                    "cost_exact": cost_exact,
                    "cost_currency": currency,
                    "usage": costs.json_safe_metadata(detail.get("usage")),
                    "generation_time_ms": costs.json_safe_metadata(
                        detail.get("generationTimeMs") or detail.get("generation_time")
                    ),
                    "generated_at": detail.get("createdAt") or detail.get("created_at"),
                    "generation_detail_source": generation_source(provider),
                }
            )
        )
    return enriched
