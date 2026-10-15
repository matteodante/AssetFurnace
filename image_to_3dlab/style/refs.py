"""Which approved pictures go to the generator as style references.

Pool: the bible's style references (pictures that define the style, such as an original
game's art) and its approved pictures, references first. Order: pictures of the same kind, then pictures marked for any kind, then the rest (with
no kind asked, all approved pictures are one group).
Within a group, the most typical first when the judge has scored typicality (how close
a picture is to the rest of the bible), otherwise the oldest first: the pictures a
bible was founded on are the ones that define it. Rejected pictures never go, nor a
picture whose file no longer matches the hash that was approved.
"""

from __future__ import annotations

from pathlib import Path

from image_to_3dlab.style.bible import load, sha256_file

MAX_REFERENCES = 3


POOLS = ("references", "images")


def pick(root: Path, kind: str | None, k: int = 3,
         typicality: dict[str, float] | None = None, pool: tuple[str, ...] = POOLS) -> list[Path]:
    """Up to `k` (1 to 3) picture paths, best first; fewer when the bible has fewer."""
    if not 1 <= k <= MAX_REFERENCES:
        raise ValueError(f"ask for between 1 and {MAX_REFERENCES} references, got {k}")
    return ordered(root, kind, typicality, pool)[:k]


def ordered(root: Path, kind: str | None, typicality: dict[str, float] | None = None,
            pool: tuple[str, ...] = POOLS) -> list[Path]:
    """Every usable picture of `pool` (style references, then approved), in the order `pick` takes them."""
    unknown = [p for p in pool if p not in POOLS]
    if unknown or not pool:
        raise ValueError(f"the pool is made of {', '.join(POOLS)}, got {pool!r}")
    root = Path(root)
    typicality = typicality or {}

    def group(entry: dict) -> int:
        if kind is None or entry.get("kind") == kind:
            return 0
        return 1 if entry.get("kind") is None else 2

    # Only the bytes that were approved: a file edited or replaced since is not a reference.
    data = load(root)
    listed = [e for key in pool for e in data[key]]  # pool order first: references before approved
    entries = [(i, e) for i, e in enumerate(listed)
               if (root / e["file"]).is_file() and sha256_file(root / e["file"]) == e["sha256"]]
    entries.sort(key=lambda item: (group(item[1]), -typicality.get(item[1]["sha256"], 0.0), item[0]))
    return [(root / e["file"]).resolve() for _, e in entries]
