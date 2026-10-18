#!/usr/bin/env python3
"""Install the style judges' models and llama.cpp, after saying what, how big and under which licence.

    python scripts/bootstrap_style.py                      # all of it, asks first
    python scripts/bootstrap_style.py --only siglip2,csd   # just these
    python scripts/bootstrap_style.py --yes                # non-interactive (agents, batch setup)

Parts: `siglip2` (SigLIP 2, a general image embedding, Apache-2.0), `csd` (CSD ViT-L, an
embedding trained for style, CC-BY-4.0), `llama` (llama.cpp, MIT, which runs the
Qwen3-VL judge). The Qwen3-VL model itself comes with Qwen-Image's setup. Hugging Face
files are fetched at a pinned revision into the usual cache; llama.cpp is checked
against a pinned checksum and unpacked into `vendor/llama-cpp/`.

Nothing is fetched without an explicit yes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab.style import models  # noqa: E402

PARTS = ("siglip2", "csd", "llama")


def parts(only: str | None) -> list[str]:
    if not only:
        return list(PARTS)
    chosen = [p.strip() for p in only.split(",") if p.strip()]
    unknown = [p for p in chosen if p not in PARTS]
    if unknown:
        raise SystemExit(f"unknown part {', '.join(unknown)}; choose from {', '.join(PARTS)}")
    return chosen


def _size(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GB" if value >= 1024 ** 3 else f"{value / 1024 ** 2:.1f} MB"


def announcement(chosen: list[str], build: models.LlamaBuild | None) -> str:
    lines, total = ["Style judges: this will download"], 0
    for key in chosen:
        if key == "llama":
            if build is None:
                lines.append("  llama.cpp: no prebuilt build for this machine; build it yourself into vendor/llama-cpp/")
                continue
            lines.append(f"  llama.cpp {models.LLAMA_TAG} ({models.LLAMA_LICENSE})  {_size(build.size)}\n"
                         f"      from {models.LLAMA_RELEASES}{build.archive}\n"
                         f"      to   vendor/llama-cpp/")
            total += build.size
        else:
            model = models.HUB_MODELS[key]
            lines.append(f"  {model.label} ({model.license})  {_size(model.size)}\n"
                         f"      from https://huggingface.co/{model.repo} at {model.revision[:12]}\n"
                         f"      {model.note}")
            total += model.size
    lines.append(f"  total {_size(total)}")
    return "\n".join(lines)


def fetch_hub(model: models.HubModel) -> None:
    from huggingface_hub import hf_hub_download

    for name in model.files:
        hf_hub_download(model.repo, name, revision=model.revision)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "assetfurnace-bootstrap-style"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def unpack_llama(archive: bytes, build: models.LlamaBuild, dest: Path) -> None:
    """Check the archive against its pinned checksum, then unpack it into `dest`, nowhere else."""
    if hashlib.sha256(archive).hexdigest() != build.sha256:
        raise SystemExit(f"{build.archive} does not match its pinned checksum; nothing installed")
    if archive[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            members = [(i.filename, i.is_dir(), lambda i=i: bundle.read(i)) for i in bundle.infolist()]
            _write_members(members, dest)
    else:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            members = [(m.name, m.isdir(), lambda m=m: bundle.extractfile(m).read())
                       for m in bundle.getmembers() if m.isdir() or m.isfile()]
            # the Linux build's versioned libraries (libllama.so.0 -> libllama.so.0.0.N): the loader
            # looks for the link's name, so a link left out is a tool that cannot start
            links = [(m.name, str(Path(m.name).parent / m.linkname) if m.issym() else m.linkname)
                     for m in bundle.getmembers() if m.issym() or m.islnk()]
            _write_members(members, dest)
            _write_links(links, dest)


def _write_members(members: list, dest: Path) -> None:
    root = dest.resolve()
    for name, _, _ in members:
        if not (root / name).resolve().is_relative_to(root):
            raise SystemExit(f"the archive has a file outside its folder ({name}); nothing installed")
    for name, is_dir, read in members:
        target = root / name
        if is_dir:
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read())
        if target.suffix in ("", ".so", ".dylib"):
            target.chmod(0o755)


def _write_links(links: list[tuple[str, str]], dest: Path) -> None:
    """Each link as a copy of the file it names (a copy works on every system and needs no
    right to make links), inside `dest` only; a link to a link waits for that one."""
    root = dest.resolve()
    for name, target in links:
        for path in (root / name, root / target):
            if not path.resolve().is_relative_to(root):
                raise SystemExit(f"the archive has a link outside its folder ({name}); nothing installed")
    waiting = list(links)
    while waiting:
        left = [(name, target) for name, target in waiting if not (root / target).is_file()]
        for name, target in waiting:
            if (root / target).is_file():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes((root / target).read_bytes())
                if (root / name).suffix in ("", ".so", ".dylib") or ".so." in (root / name).name:
                    (root / name).chmod(0o755)
        if len(left) == len(waiting):
            raise SystemExit(f"the archive links {left[0][0]} to {left[0][1]}, which it does not hold; "
                             "nothing usable installed")
        waiting = left


def fetch_llama(build: models.LlamaBuild) -> None:
    unpack_llama(download(models.LLAMA_RELEASES + build.archive), build, models.llama_dir())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help=f"comma-separated parts: {', '.join(PARTS)}")
    parser.add_argument("--yes", action="store_true", help="do not ask")
    args = parser.parse_args(argv)
    chosen = parts(args.only)
    build = models.llama_build()
    print(announcement(chosen, build), flush=True)
    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    for key in chosen:
        if key == "llama":
            if build is None:
                print("llama.cpp has no prebuilt release for this machine (Windows x64, Linux x64 and Apple "
                      "Silicon only): build it from https://github.com/ggml-org/llama.cpp into vendor/llama-cpp/",
                      flush=True)
                return 1
            if build is not None:
                print("Downloading llama.cpp...", flush=True)
                fetch_llama(build)
        else:
            print(f"Downloading {models.HUB_MODELS[key].label}...", flush=True)
            fetch_hub(models.HUB_MODELS[key])
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
