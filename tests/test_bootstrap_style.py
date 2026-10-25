"""scripts/bootstrap_style.py: the style judge's llama.cpp download, only after a yes."""

from __future__ import annotations

import hashlib
import io
import zipfile

import bootstrap_style
import pytest

from image_to_3dlab.style import models


def test_the_announcement_names_the_part_its_size_and_its_licence():
    text = bootstrap_style.announcement(["llama"], models.llama_build("Windows", "AMD64"))
    assert "llama.cpp" in text and "MIT" in text and "30.7 MB" in text
    assert "total" in text.lower()
    assert "SigLIP" not in text and "CSD" not in text


def test_each_supported_machine_gets_a_pinned_llama_build():
    for system, machine in (("Windows", "AMD64"), ("Linux", "x86_64"), ("Darwin", "arm64")):
        build = models.llama_build(system, machine)
        assert build and len(build.sha256) == 64 and build.size > 1_000_000
    assert models.llama_build("Linux", "riscv64") is None


def test_parts_are_chosen_with_only_and_unknown_ones_are_refused():
    assert bootstrap_style.parts(None) == ["llama"]
    assert bootstrap_style.parts("llama") == ["llama"]
    for retired in ("siglip2", "csd", "dinov3"):
        with pytest.raises(SystemExit):
            bootstrap_style.parts(retired)


def test_nothing_is_downloaded_without_a_yes(monkeypatch, capsys):
    fetched = []
    monkeypatch.setattr(bootstrap_style, "fetch_llama", lambda *a, **k: fetched.append(a))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert bootstrap_style.main(["--only", "llama"]) == 1
    assert fetched == [] and "--yes" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert bootstrap_style.main(["--only", "llama"]) == 1
    assert fetched == []


def test_yes_fetches_llama(monkeypatch):
    fetched = []
    monkeypatch.setattr(bootstrap_style, "fetch_llama", lambda build, **k: fetched.append("llama"))
    linux = models.llama_build("Linux", "x86_64")
    monkeypatch.setattr(models, "llama_build", lambda: linux)
    assert bootstrap_style.main(["--yes"]) == 0
    assert fetched == ["llama"]


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


def test_the_linux_builds_library_links_are_kept(tmp_path):
    """Codex review: links in the Linux tarball (libllama.so.0 -> libllama.so.0.0.1) were dropped,
    so llama-completion could not load its libraries on a fresh install."""
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        def add(name, payload=None, link=None):
            info = tarfile.TarInfo(name)
            if link is None:
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            else:
                info.type, info.linkname = tarfile.SYMTYPE, link
                bundle.addfile(info)
        add("build/bin/libllama.so.0.0.1", b"ELF")
        add("build/bin/libllama.so", link="libllama.so.0")      # a link to a link, before its target
        add("build/bin/libllama.so.0", link="libllama.so.0.0.1")
        add("build/bin/llama-completion", b"ELF")
    archive = buffer.getvalue()
    build = models.LlamaBuild("llama.tar.gz", hashlib.sha256(archive).hexdigest(), len(archive))
    dest = tmp_path / "llama-cpp"
    bootstrap_style.unpack_llama(archive, build, dest)
    for name in ("libllama.so.0", "libllama.so"):
        assert (dest / "build" / "bin" / name).read_bytes() == b"ELF"


def test_a_link_cannot_point_outside_its_folder(tmp_path):
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("bin/libx.so")
        info.type, info.linkname = tarfile.SYMTYPE, "../../../etc/passwd"
        bundle.addfile(info)
    archive = buffer.getvalue()
    build = models.LlamaBuild("llama.tar.gz", hashlib.sha256(archive).hexdigest(), len(archive))
    with pytest.raises(SystemExit, match="outside"):
        bootstrap_style.unpack_llama(archive, build, tmp_path / "dest")
