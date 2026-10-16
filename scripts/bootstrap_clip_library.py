#!/usr/bin/env python3
"""Install the CC0 creature clip library (Mesh2Motion's animated creatures).

    python scripts/bootstrap_clip_library.py          # says what it will fetch, then asks
    python scripts/bootstrap_clip_library.py --yes    # non-interactive
    python scripts/bootstrap_clip_library.py --check  # exit 0 if installed, 1 if not

Six GLB files, about 4 MB, CC0, from one pinned commit of Mesh2Motion. Each is checked
against its SHA-256 before it is kept, and a `SOURCE.json` beside them records where they
came from. See `image_to_3dlab/clip_library.py` for what is fetched and what is not.

Nothing is fetched without an explicit yes: AGENTS.md forbids downloads the user has not
chosen.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import clip_library as cl


def announcement() -> str:
    src = cl.MESH2MOTION
    return (f"Creature clip library: {src['name']} animations "
            f"({cl.TOTAL_BYTES / 1e6:.1f} MB, {src['license']})\n"
            f"  from   {src['repo']} at {src['commit'][:7]}\n"
            f"  files  {', '.join(cl.local_name(n) for n in src['files'])}\n"
            f"  to     {cl.pack_dir()}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(target: Path, opener=urllib.request.urlopen) -> Path:
    """Fetch every file to a temporary name and verify it; only then move them all into
    place. One bad file and nothing is kept."""
    target.mkdir(parents=True, exist_ok=True)
    fetched = []
    try:
        for name, (sha, _) in cl.MESH2MOTION["files"].items():
            partial = target / (cl.local_name(name) + ".part")
            fetched.append(partial)
            with opener(cl.file_url(name)) as response, partial.open("wb") as out:
                while block := response.read(1 << 20):
                    out.write(block)
            actual = sha256_of(partial)
            if actual != sha:
                raise SystemExit(f"checksum mismatch for {cl.local_name(name)} ({actual}); nothing was installed")
    except BaseException:
        for partial in fetched:
            partial.unlink(missing_ok=True)
        raise
    for partial in fetched:
        partial.replace(partial.with_suffix(""))
    cl.write_source(target)
    return target


def main(argv: list[str] | None = None, ask=input) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="download without asking")
    parser.add_argument("--check", action="store_true", help="report only")
    args = parser.parse_args(argv)

    target = cl.pack_dir()
    if cl.installed(target):
        print(f"The creature clip library is installed: {target}")
        return 0
    if args.check:
        print("The creature clip library is not installed; creatures get procedural test clips only")
        return 1

    print(announcement(), flush=True)
    if not args.yes and ask("Download it? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing downloaded.")
        return 1
    download(target)
    print(f"Installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
