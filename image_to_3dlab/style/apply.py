"""The style bible applied to a drawing, when asked: its words after the prompt, its style
pictures with an edit, and a record of which bible (and which version of it) was used.

Off unless a route asks for it, the same way on every one: `bible` is "-" for the active
bible (`style_bible.py use NAME`), or a registered bible's name. Text to image
(`text_image.py --bible`, the studio's Create, the batch step `picture`) takes its words;
a concept from a reference (`reference_concept.py make --bible`, the studio's panel, the
batch step `concept`) also takes up to three of its style pictures; characters in pieces
(`slot_parts.py run --bible`, the Pieces page) take its words for every piece.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from image_to_3dlab.style import refs, registry
from image_to_3dlab.style.bible import load, sha256_file

OUTPUT = Path(__file__).resolve().parents[2] / "output"   # where bibles.json lives
# The kinds a bible words separately, from the kinds the routes use.
KINDS = {"prop": "prop", "creature": "creature", "equipment": "equipment_part", "equipment_part": "equipment_part",
         "character": "character", "set": "prop", "map_object": "map_object"}


@dataclass(frozen=True)
class Style:
    name: str
    root: Path
    sha256: str          # of bible.json: which version of the bible was used
    words: str
    kind: str | None
    references: list[Path] = field(default_factory=list)


def bible_kind(kind: str | None) -> str | None:
    return KINDS.get(kind or "")


def resolve(bible: str | None, kind: str | None = None, references: int = 0,
            output: Path | None = None) -> Style | None:
    """The bible a route asked for, or None when it asked for none. BibleError when the
    name is unknown or "-" is asked and no bible is active."""
    if not bible:
        return None
    root = Path(registry.find(output or OUTPUT, bible))
    data = load(root)
    wanted = bible_kind(kind)
    words = [(data.get("style") or {}).get("wording", ""),
             (((data.get("kinds") or {}).get(wanted) or {}).get("wording", "") if wanted else "")]
    pictures = refs.pick(root, wanted, references) if references > 0 else []
    return Style(name=data["name"], root=root, sha256=sha256_file(root / "bible.json"),
                 words=", ".join(w.strip().rstrip(",.") for w in words if w and w.strip()), kind=wanted,
                 references=list(pictures))


def styled(prompt: str, style: Style | None) -> str:
    """The prompt with the bible's words after it."""
    base = prompt.strip().rstrip(",.")
    return f"{base}, {style.words}" if style and style.words else base


def record(style: Style) -> dict:
    """What the provenance says about the bible used."""
    return {"bible": style.name, "bible_sha256": style.sha256, "kind": style.kind, "words": style.words,
            "references": [{"name": p.name, "sha256": sha256_file(p)} for p in style.references]}
