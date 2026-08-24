import json
from pathlib import Path

from voiceover_pipeline.execution_identity import build_execution_identity


def test_execution_identity_is_path_free_and_hashes_package_bytes(tmp_path, monkeypatch):
    package = tmp_path / "checkout" / "src" / "voiceover_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('value = "one"\n', encoding="utf-8")
    (package / "module.py").write_text("answer = 42\n", encoding="utf-8")
    monkeypatch.setattr("voiceover_pipeline.execution_identity._package_root", lambda: package)
    monkeypatch.setattr(
        "voiceover_pipeline.execution_identity._distribution_direct_url",
        lambda: {"dir_info": {"editable": True}, "url": package.parent.as_uri()},
    )
    monkeypatch.setattr(
        "voiceover_pipeline.execution_identity._git_identity",
        lambda _root: ("a" * 40, True),
    )

    first = build_execution_identity()
    (package / "module.py").write_text("answer = 43\n", encoding="utf-8")
    second = build_execution_identity()

    assert first["package_version"] == "0.6.0"
    assert first["source_kind"] == "editable-checkout"
    assert first["source_revision"] == "a" * 40
    assert first["source_dirty"] is True
    assert first["package_tree_sha256"] != second["package_tree_sha256"]
    assert str(tmp_path) not in json.dumps(first)


def test_execution_identity_classifies_noneditable_distribution_as_installed_wheel(
    tmp_path, monkeypatch
):
    package = tmp_path / "site-packages" / "voiceover_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr("voiceover_pipeline.execution_identity._package_root", lambda: package)
    monkeypatch.setattr(
        "voiceover_pipeline.execution_identity._distribution_direct_url",
        lambda: {"archive_info": {"hash": "sha256=fixture"}, "url": "file:///private/wheel.whl"},
    )

    receipt = build_execution_identity()

    assert receipt["source_kind"] == "installed-wheel"
    assert receipt["source_revision"] is None
    assert receipt["source_dirty"] is None
    assert str(Path.home()) not in json.dumps(receipt)
