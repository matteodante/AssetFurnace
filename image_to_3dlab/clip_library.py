"""The CC0 creature clip library: which hand-made animations we fetch, from where, and how
to tell they are the ones we checked.

Mesh2Motion (https://github.com/Mesh2Motion/mesh2motion-app) ships animated skeletons for
most of our creature families. Its `LICENSE-CC0.MD` dedicates "all 3d models, blend files,
rigs, animations" to the public domain (the code is MIT, and we use none of it). Only the
`static/animations/<creature>-animations.glb` files are fetched: the repository's CMU
sample is not CC0, and its `models-variation` carry a licence per file.

Every file is pinned to one commit and one SHA-256, so what lands on disk is exactly what
was read and licence-checked on 2026-10-16. `scripts/bootstrap_clip_library.py` fetches
them, after asking; `scripts/chain_retarget.py` will carry them onto a rig.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIBRARY_DIR = REPO / "vendor" / "clip-library"
SOURCE_FILE = "SOURCE.json"

MESH2MOTION = {
    "name": "Mesh2Motion",
    "repo": "https://github.com/Mesh2Motion/mesh2motion-app",
    "commit": "653a9698f5f315523072f1c3ff3496100401317b",
    "license": "CC0-1.0",
    "license_url": ("https://github.com/Mesh2Motion/mesh2motion-app/blob/"
                    "653a9698f5f315523072f1c3ff3496100401317b/LICENSE-CC0.MD"),
    # path in the repository: (sha256, bytes)
    "files": {
        "static/animations/bird-animations.glb":
            ("9624b0bf70abe400ed0445c2423c69d5e046bd7d60074e399e2be161763c4f99", 329_424),
        "static/animations/dragon-animations.glb":
            ("73262041b09aa0499778818162b34955e76e17548212825548799643d5c925b3", 752_088),
        "static/animations/fox-animations.glb":
            ("80e80641a5692a14aba7616a46d2a28d3595479764887c578c766b99dba8a0d0", 788_588),
        "static/animations/kaiju-animations.glb":
            ("675a098d8cdddf5247c55d49fa04712556cd0493facec0be586b6bb9aaeed410", 1_046_032),
        "static/animations/snake-animations.glb":
            ("795e4c62ba1e02f3f33e900832cdc6a1ca5a4aa4f6fe426fccd3aaa0f7f6ac25", 359_604),
        "static/animations/spider-animations.glb":
            ("a321c1617c7f74cf86c050694bc254e74409c6c69db26fe542338f2f7c43d07a", 597_156),
    },
}
TOTAL_BYTES = sum(size for _, size in MESH2MOTION["files"].values())


def pack_dir() -> Path:
    """Where the Mesh2Motion files live once fetched."""
    return LIBRARY_DIR / "mesh2motion"


def file_url(name: str) -> str:
    """The pinned raw download address of one repository file."""
    repo = MESH2MOTION["repo"].removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{repo}/{MESH2MOTION['commit']}/{name}"


def local_name(name: str) -> str:
    """`static/animations/fox-animations.glb` -> `fox-animations.glb`."""
    return name.rsplit("/", 1)[-1]


def installed(folder: Path | None = None) -> bool:
    """Every file present at its expected size. The hashes were checked when they were
    fetched; this is the cheap question a studio page asks on every load."""
    folder = folder or pack_dir()
    return all((folder / local_name(n)).is_file() and (folder / local_name(n)).stat().st_size == size
               for n, (_, size) in MESH2MOTION["files"].items())


def source_record() -> dict:
    """What `SOURCE.json` says, and what a clip's provenance quotes about its motion."""
    return {
        "name": MESH2MOTION["name"], "repo": MESH2MOTION["repo"], "commit": MESH2MOTION["commit"],
        "license": MESH2MOTION["license"], "license_url": MESH2MOTION["license_url"],
        "files": {local_name(n): {"path": n, "sha256": sha, "bytes": size}
                  for n, (sha, size) in MESH2MOTION["files"].items()},
    }


def write_source(folder: Path) -> Path:
    path = folder / SOURCE_FILE
    path.write_text(json.dumps(source_record(), indent=2) + "\n", encoding="utf-8")
    return path
