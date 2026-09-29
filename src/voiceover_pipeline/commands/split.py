from pathlib import Path

from ..models import ScriptChunk
from ..script_splitter import split_markdown_by_delimiter


class ScriptNotFoundError(FileNotFoundError):
    """Raised when the requested script path does not exist on disk."""


def prepare_split_chunks(script_path: Path, delimiter: str = "******") -> list[ScriptChunk]:
    """Validate and read a script file, returning its split chunks."""
    if not script_path.exists():
        raise ScriptNotFoundError(f"Script file not found: {script_path}")
    return split_markdown_by_delimiter(script_path, delimiter)
