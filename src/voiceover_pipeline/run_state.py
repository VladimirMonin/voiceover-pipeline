from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ChunkArtifact, ScriptChunk


STATE_FILE = "run_state.json"
LOG_FILE = "generation.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def chunk_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def script_hash(chunks: list[ScriptChunk]) -> str:
    payload = [{"id": chunk.id, "text": chunk.text} for chunk in chunks]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def initial_state(
    *,
    provider: str,
    model: str,
    voice: str,
    script_path: Path,
    chunks: list[ScriptChunk],
    script_format: str,
    run_id: str,
    limited_to_chunks: int | None = None,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "artifact_type": "voiceover-run-state",
        "status": "running",
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "voice": voice,
        "script": str(script_path.resolve()),
        "script_format": script_format,
        "script_hash": script_hash(chunks),
        "chunk_count": len(chunks),
        "limited_to_chunks": limited_to_chunks,
        "completed_count": 0,
        "started_at": now,
        "updated_at": now,
        "chunks": [],
        "errors": [],
    }


def completed_numbers(state: dict[str, Any] | None) -> set[int]:
    if not state:
        return set()
    return {
        int(item["number"])
        for item in state.get("chunks", [])
        if isinstance(item, dict) and item.get("status") == "completed" and item.get("number") is not None
    }


def artifact_from_state(item: dict[str, Any]) -> ChunkArtifact:
    return ChunkArtifact(
        number=int(item["number"]),
        id=item["id"],
        file=item["file"],
        duration_ms=int(item["duration_ms"]),
        duration_sec=round(int(item["duration_ms"]) / 1000, 3),
        start_ms=int(item.get("start_ms", 0)),
        end_ms=int(item.get("end_ms", 0)),
        text_characters=int(item.get("text_characters", 0)),
        transcript=item.get("transcript"),
        client_path=item.get("client_path"),
        generation_id=item.get("generation_id"),
        cost_rub=item.get("cost_rub"),
        cost_rub_exact=item.get("cost_rub_exact"),
        cost=item.get("cost"),
        cost_exact=item.get("cost_exact"),
        cost_currency=item.get("cost_currency"),
        usage=item.get("usage"),
        generation_time_ms=item.get("generation_time_ms"),
        generated_at=item.get("generated_at"),
        generation_detail_source=item.get("generation_detail_source"),
    )


def state_chunks_as_artifacts(state: dict[str, Any] | None, chunks_dir: Path) -> list[ChunkArtifact]:
    if not state:
        return []
    artifacts = []
    for item in state.get("chunks", []):
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        file_name = item.get("file")
        if not file_name or not (chunks_dir / file_name).exists():
            continue
        artifacts.append(artifact_from_state(item))
    return sorted(artifacts, key=lambda artifact: artifact.number)


def upsert_completed_chunk(
    state: dict[str, Any],
    *,
    artifact: ChunkArtifact,
    model: str,
    voice: str,
    text: str,
) -> None:
    item = {
        "status": "completed",
        "number": artifact.number,
        "id": artifact.id,
        "file": artifact.file,
        "path": f"chunks/{artifact.file}",
        "duration_ms": artifact.duration_ms,
        "duration_sec": artifact.duration_sec,
        "start_ms": artifact.start_ms,
        "end_ms": artifact.end_ms,
        "generation_id": artifact.generation_id,
        "model": model,
        "voice": voice,
        "text": text,
        "text_hash": chunk_text_hash(text),
        "text_characters": artifact.text_characters,
        "transcript": artifact.transcript,
        "client_path": artifact.client_path,
        "cost": artifact.cost,
        "cost_exact": artifact.cost_exact,
        "cost_currency": artifact.cost_currency,
        "cost_rub": artifact.cost_rub,
        "cost_rub_exact": artifact.cost_rub_exact,
        "usage": artifact.usage,
        "generated_at": utc_now(),
    }
    state["chunks"] = [entry for entry in state.get("chunks", []) if entry.get("number") != artifact.number]
    state["chunks"].append({key: value for key, value in item.items() if value is not None})
    state["chunks"].sort(key=lambda entry: int(entry["number"]))
    state["completed_count"] = len([entry for entry in state["chunks"] if entry.get("status") == "completed"])
    state["updated_at"] = utc_now()


def append_error(state: dict[str, Any], *, chunk_id: str | None, message: str) -> None:
    state.setdefault("errors", []).append({"at": utc_now(), "chunk_id": chunk_id, "message": message})
    state["status"] = "failed"
    state["updated_at"] = utc_now()


class GenerationLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, event: str, **fields: Any) -> None:
        parts = [utc_now(), level.upper(), event]
        for key, value in fields.items():
            if value is None:
                continue
            text = str(value).replace("\n", " ")
            if any(ch.isspace() for ch in text) or text == "":
                text = json.dumps(text, ensure_ascii=False)
            parts.append(f"{key}={text}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(parts) + "\n")
