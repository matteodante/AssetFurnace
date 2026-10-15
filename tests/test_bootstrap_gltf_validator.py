"""scripts/bootstrap_gltf_validator.py: Khronos' glTF-Validator for the `gltf.valid` check."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile

import bootstrap_gltf_validator as boot
import pytest

from image_to_3dlab.checks import find_validator


def _zip(payload: bytes = b"MZ fake") -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("gltf_validator.exe", payload)
        archive.writestr("LICENSE", "Apache License")
        archive.writestr("docs/validation.schema.json", "{}")
    return data.getvalue()


def _tar(payload: bytes = b"#!/bin/sh\n") -> bytes:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:xz") as archive:
        for name, body in (("gltf_validator", payload), ("LICENSE", b"Apache License")):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return data.getvalue()


def test_every_supported_machine_has_a_pinned_build():
    assert boot.asset_for("Windows", "AMD64").archive.endswith("win64.zip")
    assert boot.asset_for("Linux", "x86_64").archive.endswith("linux64.tar.xz")
    assert boot.asset_for("Darwin", "arm64").archive.endswith("macos64.tar.xz")
    assert boot.asset_for("Linux", "aarch64") is None
    for asset in boot.ASSETS.values():
        assert len(asset.sha256) == 64 and asset.size > 1_000_000


def test_the_announcement_names_the_tool_route_size_and_licence():
    text = boot.announcement(boot.asset_for("Windows", "AMD64"))
    assert "glTF-Validator" in text and "Apache-2.0" in text
    assert "MB" in text and "vendor/gltf-validator" in text
    assert boot.RELEASES in text


def test_install_checks_the_pin_before_writing_anything(tmp_path):
    archive = _zip()
    asset = boot.Asset("x-win64.zip", "0" * 64, len(archive), "gltf_validator.exe")
    with pytest.raises(SystemExit, match="checksum"):
        boot.install(archive, asset, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_install_writes_the_binary_and_its_licence_from_a_zip(tmp_path):
    archive = _zip()
    asset = boot.Asset("x-win64.zip", hashlib.sha256(archive).hexdigest(), len(archive), "gltf_validator.exe")
    binary = boot.install(archive, asset, tmp_path / "vendor" / "gltf-validator")
    assert binary.read_bytes() == b"MZ fake"
    assert (binary.parent / "LICENSE").read_text() == "Apache License"
    assert find_validator({}, repo=tmp_path) == binary


def test_install_reads_a_tar_xz_too(tmp_path):
    archive = _tar()
    asset = boot.Asset("x-linux64.tar.xz", hashlib.sha256(archive).hexdigest(), len(archive), "gltf_validator")
    assert boot.install(archive, asset, tmp_path).read_bytes() == b"#!/bin/sh\n"


def test_nothing_is_fetched_without_a_yes(monkeypatch, capsys):
    fetched = []
    monkeypatch.setattr(boot, "download", lambda url: fetched.append(url))
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: False)
    assert boot.main([]) == 1
    assert not fetched
    assert "--yes" in capsys.readouterr().out


def test_a_validator_that_cannot_start_is_removed(tmp_path, monkeypatch, capsys):
    """Codex review: an x86-64 build on a Mac without Rosetta must not stay 'installed'."""
    binary = tmp_path / "gltf_validator"

    def fake_install(archive, asset, dest):
        binary.write_bytes(b"x")
        return binary

    def cannot_start(*args, **kwargs):
        raise OSError("Bad CPU type in executable")

    monkeypatch.setattr(boot, "download", lambda url: b"")
    monkeypatch.setattr(boot, "install", fake_install)
    monkeypatch.setattr(boot.subprocess, "run", cannot_start)
    assert boot.main(["--yes"]) == 1
    assert not binary.exists()
    assert "Rosetta" in capsys.readouterr().out
