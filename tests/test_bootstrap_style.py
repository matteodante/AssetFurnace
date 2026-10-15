"""scripts/bootstrap_style.py: the style judges' downloads, only after a yes."""

from __future__ import annotations

import hashlib
import io
import zipfile

import bootstrap_style
import pytest

from image_to_3dlab.style import models


def test_the_announcement_names_every_part_its_size_and_its_licence():
    text = bootstrap_style.announcement(["siglip2", "csd", "llama"], models.llama_build("Windows", "AMD64"))
    assert "SigLIP 2" in text and "Apache-2.0" in text and "1.40 GB" in text
    assert "CSD" in text and "CC-BY-4.0" in text and "2.27 GB" in text
    assert "llama.cpp" in text and "MIT" in text and "30.7 MB" in text
    assert "total" in text.lower()


def test_each_supported_machine_gets_a_pinned_llama_build():
    for system, machine in (("Windows", "AMD64"), ("Linux", "x86_64"), ("Darwin", "arm64")):
        build = models.llama_build(system, machine)
        assert build and len(build.sha256) == 64 and build.size > 1_000_000
    assert models.llama_build("Linux", "riscv64") is None


def test_parts_are_chosen_with_only_and_unknown_ones_are_refused():
    assert bootstrap_style.parts(None) == ["siglip2", "csd", "llama"]
    assert bootstrap_style.parts("csd,llama") == ["csd", "llama"]
    with pytest.raises(SystemExit):
        bootstrap_style.parts("dinov3")


def test_nothing_is_downloaded_without_a_yes(monkeypatch, capsys):
    fetched = []
    monkeypatch.setattr(bootstrap_style, "fetch_hub", lambda *a, **k: fetched.append(a))
    monkeypatch.setattr(bootstrap_style, "fetch_llama", lambda *a, **k: fetched.append(a))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert bootstrap_style.main(["--only", "csd"]) == 1
    assert fetched == [] and "--yes" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert bootstrap_style.main(["--only", "csd"]) == 1
    assert fetched == []


def test_yes_fetches_the_chosen_parts(monkeypatch):
    fetched = []
    monkeypatch.setattr(bootstrap_style, "fetch_hub", lambda model, **k: fetched.append(model.key))
    monkeypatch.setattr(bootstrap_style, "fetch_llama", lambda build, **k: fetched.append("llama"))
    assert bootstrap_style.main(["--only", "siglip2,llama", "--yes"]) == 0
    assert fetched == ["siglip2", "llama"]


def zipped(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, payload in files.items():
            bundle.writestr(name, payload)
    return buffer.getvalue()


def test_the_llama_archive_is_checked_against_its_checksum_before_unpacking(tmp_path):
    archive = zipped({"llama-server.exe": b"MZ", "LICENSE": b"MIT"})
    good = models.LlamaBuild("llama.zip", hashlib.sha256(archive).hexdigest(), len(archive))
    dest = tmp_path / "vendor" / "llama-cpp"
    bootstrap_style.unpack_llama(archive, good, dest)
    assert (dest / "llama-server.exe").read_bytes() == b"MZ"
    assert models.llama_server(tmp_path) == dest / "llama-server.exe"
    bad = models.LlamaBuild("llama.zip", "0" * 64, len(archive))
    with pytest.raises(SystemExit, match="checksum"):
        bootstrap_style.unpack_llama(archive, bad, tmp_path / "other")


def test_an_archive_cannot_write_outside_its_folder(tmp_path):
    archive = zipped({"../evil.exe": b"x"})
    build = models.LlamaBuild("llama.zip", hashlib.sha256(archive).hexdigest(), len(archive))
    with pytest.raises(SystemExit, match="outside"):
        bootstrap_style.unpack_llama(archive, build, tmp_path / "dest")
    assert not (tmp_path / "evil.exe").exists()


def test_require_names_the_command_when_a_model_is_missing(tmp_path):
    with pytest.raises(models.MissingModel, match="bootstrap_style.py --only csd"):
        models.require(models.CSD, hub=tmp_path)
    snapshot = models.CSD.cache_dir(tmp_path) / "snapshots" / models.CSD.revision
    snapshot.mkdir(parents=True)
    for name in models.CSD.files:
        (snapshot / name).write_bytes(b"x")
    assert set(models.require(models.CSD, hub=tmp_path)) == set(models.CSD.files)
