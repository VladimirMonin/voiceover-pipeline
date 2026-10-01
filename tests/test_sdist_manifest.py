"""Source archives must not bundle the checkout's input scripts."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_sdist_include_omits_private_input_directory() -> None:
    project = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    includes = set(metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])

    assert {"/src", "/README.md", "/pyproject.toml"} <= includes
    assert "/in" not in includes
    assert not includes.intersection({"/out", "/.env", "/.pi", "/.serena"})
