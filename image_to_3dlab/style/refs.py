"""Which approved pictures go to the generator as style references.

Order: pictures of the same kind, then pictures marked for any kind, then the rest (with
no kind asked, all approved pictures are one group).
Within a group, the most typical first when the judge has scored typicality (how close
a picture is to the rest of the bible), otherwise the oldest first: the pictures a
bible was founded on are the ones that define it. Rejected pictures never go.
"""

from __future__ import annotations

from pathlib import Path

from image_to_3dlab.style.bible import load

MAX_REFERENCES = 3


def pick(root: Path, kind: str | None, k: int = 3,
         typicality: dict[str, float] | None = None) -> list[Path]:
    """Up to `k` (1 to 3) picture paths, best first; fewer when the bible has fewer."""
    if not 1 <= k <= MAX_REFERENCES:
        raise ValueError(f"ask for between 1 and {MAX_REFERENCES} references, got {k}")
    root = Path(root)
    typicality = typicality or {}

    def group(entry: dict) -> int:
        if kind is None or entry.get("kind") == kind:
            return 0
        return 1 if entry.get("kind") is None else 2

    entries = [(i, e) for i, e in enumerate(load(root)["images"]) if (root / e["file"]).is_file()]
    entries.sort(key=lambda item: (group(item[1]), -typicality.get(item[1]["sha256"], 0.0), item[0]))
    return [(root / e["file"]).resolve() for _, e in entries[:k]]
