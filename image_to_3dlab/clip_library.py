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


# -- what each library creature is, chain by chain ---------------------------------------
#
# Mesh2Motion's creatures stand Y up and face +Z with their left on +X. Chains run from the
# body outwards (a serpent's from its head), by bone name, joint to joint: the same roles
# `creature_family.classify` gives a SkinTokens rig, so `scripts/chain_retarget.py` can pair
# them. A clip is (name in the GLB, whether it loops).

FRAME = {"up": [0.0, 1.0, 0.0], "forward": [0.0, 0.0, 1.0], "left": [1.0, 0.0, 0.0]}


def _legs(rows: dict[str, list[str]]) -> list[dict]:
    """{"front L": bones, ...} -> role dicts."""
    return [{"bones": bones, "side": key.split()[1], "end": key.split()[0]} for key, bones in rows.items()]


def _sided(fmt: list[str], sides=("L", "R"), case=str) -> list[dict]:
    return [{"bones": [b.format(s=case(side)) for b in fmt], "side": side} for side in sides]


FOX = {
    "file": "fox-animations.glb",
    "roles": {
        "legs": _legs({
            **{f"front {s}": [f"Front_Leg_{b}_{s}" for b in ("Upper", "Lower", "Ankle", "Foot", "Tip")] for s in "LR"},
            **{f"back {s}": [f"Back_Leg_{b}_{s}" for b in ("Upper", "Lower", "Ankle", "Foot", "Foot_1", "Tip")]
               for s in "LR"}}),
        "arms": [], "wings": [],
        "spine": ["Hips", "Spine_1", "Spine_2", "Spine_2.001", "Spine_3"], "neck": ["Spine_4", "Head", "Nose"],
        "tail": ["Tail_Base", "Tail_Mid", "Tail_Mid.001", "Tail_End", "Tail_Tip"], "body": [],
    },
    "clips": {"idle": ("Idle", True), "walk": ("Walk", True), "run": ("Run", True),
              "attack": ("Bite", False), "death": ("Death", False)},
}
KAIJU = {
    "file": "kaiju-animations.glb",
    "roles": {
        "legs": _legs({f"back {s}": [f"Back_Leg_{b}_{s}" for b in ("Upper", "Lower", "Ankle", "Foot", "Foot_1",
                                                                     "Foot_2", "Tip")] for s in "LR"}),
        "arms": _sided(["Front_Leg_Shoulder_{s}", "Front_Leg_Upper_{s}", "Front_Leg_Lower_{s}", "Hand_{s}",
                        "Finger_Index_{s}", "Finger_Index_Tip_{s}"]),
        "wings": [],
        "spine": ["Hips", "Spine_1", "Spine_2", "Spine_2.001", "Spine_3"], "neck": ["Spine_4", "Head", "Nose"],
        "tail": ["Tail_Base", "Tail_Mid", "Tail_Mid.001", "Tail_Mid.002", "Tail_Mid.003", "Tail_Tip"], "body": [],
    },
    "clips": {"idle": ("Idle", True), "walk": ("Walk", True), "attack": ("Attack", False),
              "hit": ("Hit", False), "death": ("Death", False)},
}
SPIDER = {
    "file": "spider-animations.glb",
    "roles": {
        # a list, not `_legs`: two rows are "mid", and as dictionary keys one would replace the other
        "legs": [{"bones": [f"leg_{row}_{k}_{s.lower()}" for k in ("1", "2", "3", "tip")], "side": s, "end": end}
                 for s in "LR" for row, end in zip("abcd", ("front", "mid", "mid", "back"))],
        "arms": [], "wings": [],
        "spine": ["hips", "spine_1", "spine_2"], "neck": ["head", "head_tip"],
        "tail": ["tail_1", "tail_2", "tail_3", "tail_tip"], "body": [],
    },
    "clips": {"idle": ("Idle", True), "walk": ("Walk", True), "attack": ("Attack", False),
              "hit": ("Hit", False), "death": ("Death", False)},
}
SNAKE = {
    "file": "snake-animations.glb",
    "roles": {
        "legs": [], "arms": [], "wings": [], "spine": [], "neck": [], "tail": [],
        "body": ["head", "neck", *[f"tail{k:02d}" for k in range(1, 21)], "tail_tip"],
    },
    "clips": {"idle": ("Idle", True), "slither": ("Side winding", True), "attack": ("Bite", False),
              "hit": ("Hit", False), "death": ("Death", False)},
}
DRAGON = {
    "file": "dragon-animations.glb",
    "roles": {
        "legs": _legs({
            **{f"front {s}": [f"front_{b}_{s.lower()}" for b in ("leg", "ankle", "foot", "toes", "toes_tip")]
               for s in "LR"},
            **{f"back {s}": [f"back_{b}_{s.lower()}" for b in ("leg", "ankle", "foot", "toes", "toes_tip")]
               for s in "LR"}}),
        "arms": [],
        # the shoulder swings the whole wing: on no chain, its turn would be lost
        "wings": _sided(["shoulder_{s}"] + [f"wing_{k}_{{s}}" for k in range(1, 6)] + ["wing_tip_{s}"], case=str.lower),
        "spine": ["hips", "spine_0", "spine_1", "spine_2", "spine_3"], "neck": ["spine_4", "head"],
        "tail": ["tail_1", "tail_2", "tail_3", "tail_4", "tail_5", "tail_6", "tail_tip"], "body": [],
    },
    "clips": {"idle": ("Idle", True), "walk": ("Walk", True), "fly": ("Fly Flap", True),
              "glide": ("Fly Glide", True)},
}
BIRD = {
    "file": "bird-animations.glb",
    "roles": {
        "legs": _legs({f"back {s}": [f"{b}_{s}" for b in ("UpperLeg", "LowerLeg", "AnkleLeg", "Foot", "Toes",
                                                           "Toes_tip")] for s in "LR"}),
        "arms": [],
        "wings": _sided([f"wing_{k}_{{s}}" for k in range(1, 6)] + ["wing_tip_{s}"]),
        "spine": ["hips", "spine_0", "spine_1", "spine_2"], "neck": ["spine_3", "head"],
        "tail": ["tail_1", "tail_2", "tail_3", "tail_tip"], "body": [],
    },
    "clips": {"idle": ("Idle", True), "walk": ("Walk", True), "fly": ("Flap", True), "glide": ("Glide", True)},
}
SOURCES = {"fox": FOX, "kaiju": KAIJU, "spider": SPIDER, "snake": SNAKE, "dragon": DRAGON, "bird": BIRD}


def source_for(family: str, legs: int = 0) -> str | None:
    """Which library creature drives a family: a winged creature on four legs is a dragon,
    on two or none a bird. Plants and humanoids have none."""
    if family == "winged":
        return "dragon" if legs >= 3 else "bird"
    return {"quadruped": "fox", "digitigrade": "kaiju", "multileg": "spider", "serpent": "snake"}.get(family)


def source_roles(name: str) -> dict:
    """A library creature's roles, with its frame, ready for `chain_retarget.Source`."""
    return {"family": None, **SOURCES[name]["roles"], "frame": FRAME}


# A looping clip also comes as a take twice as long, made of its own pieces in a new order
# (`motion_variation`, after GenMM), so a creature does not idle in a two-second loop.
LONGER = "-long"
LONGER_FACTOR = 2.0
LONGER_SEED = 0


def variations(name: str) -> list[str]:
    """The varied motions of a library creature: one longer take of each looping clip."""
    return [motion + LONGER for motion, (_, loop) in SOURCES[name]["clips"].items() if loop]


def base_motion(motion: str) -> str:
    return motion.removesuffix(LONGER)


def motion_record(name: str, motion: str, factor: float = LONGER_FACTOR, seed: int = LONGER_SEED) -> dict:
    """What a clip's provenance says about where its motion came from (`factor` and `seed`:
    what a longer take was varied with)."""
    src = SOURCES[name]
    clip, _ = src["clips"][base_motion(motion)]
    sha, _ = MESH2MOTION["files"][f"static/animations/{src['file']}"]
    record = {"made_with": f"chain retarget of a {MESH2MOTION['name']} clip", "source_clip": clip,
              "source_file": src["file"], "source_sha256": sha, "repo": MESH2MOTION["repo"],
              "commit": MESH2MOTION["commit"],
              "license": {"name": MESH2MOTION["license"], "url": MESH2MOTION["license_url"]}}
    if motion != base_motion(motion):
        record["varied_with"] = (f"GenMM patch synthesis (AssetFurnace's numpy port, Apache-2.0), "
                                 f"{factor:g}x as long, seed {seed}")
    return record
