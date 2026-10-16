"""The CC0 clip library: what it fetches, from where, and that nothing unverified is kept."""

from __future__ import annotations

import hashlib
import io
import json
import re

import pytest

from image_to_3dlab import clip_library as cl


def test_every_file_is_pinned_to_a_commit_with_its_hash_and_size():
    assert re.fullmatch(r"[0-9a-f]{40}", cl.MESH2MOTION["commit"])
    for name, (sha, size) in cl.MESH2MOTION["files"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", sha) and size > 0
        assert cl.MESH2MOTION["commit"] in cl.file_url(name)
    assert cl.TOTAL_BYTES == sum(size for _, size in cl.MESH2MOTION["files"].values())


def test_only_the_cc0_animation_files_are_fetched():
    """The repository also ships a CMU sample and per-file licensed variations; neither
    is covered by its CC0 notice, so neither may ever be in the download list."""
    assert cl.MESH2MOTION["license"] == "CC0-1.0"
    assert cl.MESH2MOTION["license_url"].endswith(f"{cl.MESH2MOTION['commit']}/LICENSE-CC0.MD")
    for name in cl.MESH2MOTION["files"]:
        assert name.startswith("static/animations/") and name.endswith("-animations.glb")
        assert "CarnegieMellon" not in name and "variation" not in name


def test_an_empty_folder_is_not_installed(tmp_path):
    assert not cl.installed(tmp_path)


def _fake(files: dict[str, bytes]):
    table = {name: (hashlib.sha256(data).hexdigest(), len(data)) for name, data in files.items()}

    def opener(url):
        name = next(n for n in files if url.endswith(n))
        return io.BytesIO(files[name])

    return table, opener


def test_a_verified_download_is_kept_with_its_source_record(tmp_path, monkeypatch):
    import bootstrap_clip_library as boot

    table, opener = _fake({"static/animations/fox-animations.glb": b"fox", "static/animations/snake-animations.glb": b"sn"})
    monkeypatch.setitem(cl.MESH2MOTION, "files", table)
    boot.download(tmp_path, opener=opener)
    assert (tmp_path / "fox-animations.glb").read_bytes() == b"fox"
    record = json.loads((tmp_path / cl.SOURCE_FILE).read_text(encoding="utf-8"))
    assert record["license"] == "CC0-1.0" and record["commit"] == cl.MESH2MOTION["commit"]
    assert record["files"]["fox-animations.glb"]["sha256"] == table["static/animations/fox-animations.glb"][0]
    assert cl.installed(tmp_path)


def test_a_file_with_the_wrong_hash_is_never_kept(tmp_path, monkeypatch):
    import bootstrap_clip_library as boot

    table, opener = _fake({"static/animations/fox-animations.glb": b"fox"})
    table["static/animations/fox-animations.glb"] = ("0" * 64, 3)
    monkeypatch.setitem(cl.MESH2MOTION, "files", table)
    with pytest.raises(SystemExit, match="checksum"):
        boot.download(tmp_path, opener=opener)
    assert list(tmp_path.iterdir()) == []


def test_nothing_is_fetched_without_a_yes(tmp_path, monkeypatch, capsys):
    import bootstrap_clip_library as boot

    monkeypatch.setattr(cl, "pack_dir", lambda: tmp_path / "mesh2motion")
    monkeypatch.setattr(boot, "download", lambda *a, **k: pytest.fail("downloaded without a yes"))
    assert boot.main([], ask=lambda _: "n") == 1
    out = capsys.readouterr().out
    assert "CC0" in out and "MB" in out and cl.MESH2MOTION["commit"][:7] in out
    assert boot.main(["--check"]) == 1


def test_the_library_is_in_the_catalogue_with_its_licence_and_size():
    import backend_catalog as bc
    import download_api as dl

    entry = bc.BY_ID["clip-library"]
    assert entry.kind == "tool" and entry.license_name == "CC0-1.0"
    assert entry.bytes_expected == cl.TOTAL_BYTES
    assert entry.weights[0].path == cl.pack_dir()
    command = dl.COMMANDS["clip-library"]
    assert command[1].endswith("bootstrap_clip_library.py") and "--yes" in command
