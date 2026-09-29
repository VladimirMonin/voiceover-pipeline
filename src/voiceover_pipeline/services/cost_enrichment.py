"""Provider-side history cost enrichment for the CLI entry point.

This module owns the post-generation lookup that binds an already-saved chunk
artifact to the cost its own provider generation reported: a Polza chat-audio
``X-Generation-Id`` history GET or a targeted OpenRouter generation GET by the
chunk's own body id. It performs no submit and no other provider request.

Both history lookups are injected as explicit keyword dependencies, so
``cli.attach_costs`` can forward its currently bound callables and existing
monkeypatch targets keep steering this body without importing ``cli``.

The same module also owns the pre-generation provider price-route selection in
``fetch_pricing_snapshot``, which picks the injected Polza or OpenRouter model
lookup and itself opens no connection.

The same module also owns the two summaries the observed cost feeds into: the
late cost metadata copied into the matching trusted state chunk, and the run
total whose canonical value is a Decimal sum of provider-reported exact strings.
Neither writes an unobserved cost or calls a provider.
"""

import time
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from ..models import ChunkArtifact
from ..pricing import cost_from_generation
from . import costs
from .recovery import state_entry_number


class AttachedCostStateMismatchError(RuntimeError):
    """A late-observed cost artifact matches no single completed state chunk."""


def generation_source(provider: str) -> str:
    """Return the provenance label for a provider's observed cost source."""
    return {
        "polza-chat-audio": "Polza GET /api/v1/history/generations/{id}",
        "polza-tts": "Polza API usage.cost_rub (direct)",
        "openrouter-tts": "OpenRouter GET /api/v1/generation?id=...",
        "qwen-local": "qwen-local (free)",
        "omnivoice-local": "omnivoice-local (local model; no billing request)",
    }.get(provider, "unknown")


def fetch_pricing_snapshot(
    provider: str,
    api_key: str,
    model: str,
    *,
    fetch_polza_pricing,
    fetch_openrouter_pricing,
) -> dict | None:
    """Select the model-price lookup for a provider without doing I/O itself.

    Both lookups are injected as explicit keyword dependencies so
    ``cli.fetch_pricing_snapshot`` can forward its currently bound callables and
    existing monkeypatch targets keep steering this body without importing
    ``cli``. Providers without a price route return ``None`` and issue no request.
    """
    if provider in ("polza-chat-audio", "polza-tts"):
        return fetch_polza_pricing(api_key, model)
    if provider == "openrouter-tts":
        return fetch_openrouter_pricing(model)
    return None


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


# Cost metadata that a late history lookup can add after a chunk was already
# saved to run state.
_ATTACHED_COST_STATE_FIELDS = (
    "cost",
    "cost_exact",
    "cost_currency",
    "cost_rub",
    "cost_rub_exact",
    "usage",
    "generation_time_ms",
    "generated_at",
    "generation_detail_source",
)


def merge_attached_costs_into_state(state: dict[str, Any], artifacts: list[ChunkArtifact]) -> None:
    """Copy late-observed cost metadata into the matching trusted state chunk.

    ``attach_costs`` runs after each chunk was already saved to ``run_state.json``,
    so a cost that only the history lookup revealed would otherwise live in the
    manifests alone and be lost by a resume whose lookup is unavailable. An
    artifact is bound by its own ``id`` *and* ``number``; a missing or ambiguous
    match raises ``AttachedCostStateMismatchError`` before applying that artifact;
    the caller persists state only after the full merge succeeds. Only values
    present on the enriched artifact overwrite state, so a lookup that reports
    nothing can never erase an already-observed direct cost.
    """
    entries = state.get("chunks", [])
    for artifact in artifacts:
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("status") == "completed"
            and entry.get("id") == artifact.id
            and state_entry_number(entry) == artifact.number
        ]
        if len(matches) != 1:
            raise AttachedCostStateMismatchError(
                "Cannot persist observed costs: run state has no single completed "
                f"chunk matching {artifact.id}/{artifact.number}."
            )
        entry = matches[0]
        for field in _ATTACHED_COST_STATE_FIELDS:
            value = getattr(artifact, field)
            if value is not None:
                entry[field] = value


def summarize_costs(provider: str, chunks: list[ChunkArtifact]) -> tuple:
    """Return the run cost total, exact total, currency, and provenance.

    The float total stays the legacy compatibility value, while the canonical
    total sums only finite, parsable canonical exact cost strings. A missing,
    non-finite, unparsable, or foreign exact value disables the canonical total
    rather than promoting the float total into a pseudo-exact number.
    """
    if not chunks or any(chunk.cost is None for chunk in chunks):
        return None, None, None, None
    currency = chunks[0].cost_currency
    if currency is None or any(chunk.cost_currency != currency for chunk in chunks):
        return None, None, None, None
    total = sum(float(chunk.cost or 0) for chunk in chunks)
    source = generation_source(provider)
    # The sum runs in a local context wide enough for every observed digit so it
    # is never rounded to the default 28 significant digits.
    values: list[Decimal] = []
    for chunk in chunks:
        # Only a canonical exact string can join the Decimal total. Any other
        # value (a legacy float, a bool, a list, or a Decimal from untrusted
        # resumed state) is not a canonical exact cost, so it disables the exact
        # total instead of guessing a value or raising.
        if not isinstance(chunk.cost_exact, str):
            return round(total, 8), None, currency, source
        try:
            value = Decimal(chunk.cost_exact)
        except (InvalidOperation, ValueError):
            return round(total, 8), None, currency, source
        if not value.is_finite():
            return round(total, 8), None, currency, source
        values.append(value)
    precision = max(
        28,
        max(value.adjusted() for value in values)
        - min(int(value.as_tuple().exponent) for value in values)
        + len(str(len(values)))
        + 1,
    )
    with localcontext() as context:
        context.prec = precision
        exact_total = sum(values, Decimal(0))
    return round(total, 8), str(exact_total), currency, source
