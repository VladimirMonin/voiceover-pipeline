from pathlib import Path

from .models import ScriptChunk


def split_markdown_by_delimiter(script_path: Path, delimiter: str = "******") -> list[ScriptChunk]:
    text = script_path.read_text(encoding="utf-8-sig")
    parts = [part.strip() for part in text.split(delimiter)]
    chunks = [part for part in parts if part]

    return [
        ScriptChunk(number=index, id=f"chunk_{index:02d}", text=chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]
