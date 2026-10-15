#!/usr/bin/env python3
"""Install Khronos' glTF-Validator for the asset checks, from its GitHub release.

    python scripts/bootstrap_gltf_validator.py          # says what it will fetch, then asks
    python scripts/bootstrap_gltf_validator.py --yes    # non-interactive (agents, batch setup)

The `gltf.valid` check (image_to_3dlab/checks) asks the official validator whether a GLB
follows the glTF 2.0 specification: an engine importer is far less forgiving than a web
viewer. Without it that check is skipped, never passed. This fetches the native build for
this machine (Apache-2.0, about 2 MB), checks it against a pinned checksum, and puts it in
`vendor/gltf-validator/` with its licence, where the checks already look.

Nothing is fetched without an explicit yes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERSION = "2.0.0-dev.3.10"
RELEASES = f"https://github.com/KhronosGroup/glTF-Validator/releases/download/{VERSION}/"


@dataclass(frozen=True)
class Asset:
    archive: str
    sha256: str
    size: int
    binary: str


# Pinned 2026-10-15 so a new upstream release, or a swapped file, cannot change what gets
# installed. The macOS build is x86_64 and runs on Apple Silicon through Rosetta.
ASSETS = {
    ("windows", "amd64"): Asset(
        f"gltf_validator-{VERSION}-win64.zip",
        "c5068f51205deedc28acc3529ee7e11ee60e853454f673093398eba80142202c", 2320898, "gltf_validator.exe"),
    ("linux", "x86_64"): Asset(
        f"gltf_validator-{VERSION}-linux64.tar.xz",
        "168eba887964125abe17ae97899b38d0b3cfd73c266c78424c194929ddcbc522", 2082740, "gltf_validator"),
    ("darwin", "arm64"): Asset(
        f"gltf_validator-{VERSION}-macos64.tar.xz",
        "bce89ceea00b4d3191a8779018f41744b84a4539b127465d63ba83ad0f243fef", 1824740, "gltf_validator"),
    ("darwin", "x86_64"): Asset(
        f"gltf_validator-{VERSION}-macos64.tar.xz",
        "bce89ceea00b4d3191a8779018f41744b84a4539b127465d63ba83ad0f243fef", 1824740, "gltf_validator"),
}
EXTRAS = ("LICENSE", "NOTICES")


def asset_for(system: str, machine: str) -> Asset | None:
    return ASSETS.get((system.lower(), machine.lower()))


def target_dir(repo: Path = REPO) -> Path:
    return repo / "vendor" / "gltf-validator"


def announcement(asset: Asset) -> str:
    return (f"glTF-Validator {VERSION} for the asset checks (Apache-2.0, from the Khronos Group)\n"
            f"  from   {RELEASES}{asset.archive}\n"
            f"  size   {asset.size / 1024 ** 2:.1f} MB download\n"
            f"  to     vendor/gltf-validator/{asset.binary}")


def user_agent() -> str:
    sys.path.insert(0, str(REPO))
    from image_to_3dlab import __version__

    return f"image-to-3dlab/{__version__} (+https://github.com/Bingeljell/image-to-3dlab)"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _members(archive: bytes, name: str) -> dict[str, bytes]:
    """The binary and its licence files, from a .zip or a .tar.xz."""
    wanted = {name, *EXTRAS}
    if name.endswith(".exe") or archive[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            return {n: bundle.read(n) for n in bundle.namelist() if n in wanted}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as bundle:
        return {m.name: bundle.extractfile(m).read() for m in bundle.getmembers()
                if m.isfile() and m.name in wanted}


def install(archive: bytes, asset: Asset, dest: Path) -> Path:
    """Check the archive against its pinned checksum, then write the binary and licence into `dest`."""
    if hashlib.sha256(archive).hexdigest() != asset.sha256:
        raise SystemExit(f"{asset.archive} does not match its pinned checksum; nothing installed")
    members = _members(archive, asset.binary)
    if asset.binary not in members:
        raise SystemExit(f"{asset.archive} has no {asset.binary} in it; nothing installed")
    dest.mkdir(parents=True, exist_ok=True)
    for name, payload in members.items():
        (dest / name).write_bytes(payload)
    binary = dest / asset.binary
    binary.chmod(0o755)
    return binary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="do not ask")
    args = parser.parse_args(argv)
    asset = asset_for(platform.system(), platform.machine())
    if asset is None:
        raise SystemExit(f"No glTF-Validator build for {platform.system()} {platform.machine()}. "
                         f"See https://github.com/KhronosGroup/glTF-Validator/releases")
    print(announcement(asset), flush=True)
    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    print(f"Downloading {asset.archive}...", flush=True)
    binary = install(download(RELEASES + asset.archive), asset, target_dir())
    check = subprocess.run([str(binary), "--help"], capture_output=True, text=True, check=False)
    first = (check.stdout or check.stderr).splitlines()[:1]
    if not first or "glTF 2.0 Validator" not in first[0]:
        print(f"glTF-Validator is in {binary} but did not start: {first}", flush=True)
        return 1
    print(f"Done: {first[0]} in {binary.parent}. The gltf.valid check will run now.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
