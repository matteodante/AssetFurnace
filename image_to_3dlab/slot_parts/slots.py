"""The fixed slot catalogue, and a person's choice of slots for one set.

A slot is drawn on the base body in the concept (helmet, torso, shoulders, gloves, legs,
boots) or made on its own (weapon, shield). Paired slots are cut from the character's
right side and fitted to each side; `split` only says whether the two shoulder pieces stay
separate in the output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

ROUTES = ("C", "B0")  # one clean picture per piece; or cut from the whole character in 3D


@dataclass(frozen=True)
class Slot:
    name: str
    roles: tuple[str, ...]      # bone roles a slot sits on; empty for pieces made on their own
    paired: bool = False
    rigid: str | None = None    # the one bone a rigid piece follows
    faces: int = 4000           # face budget of one piece


CATALOGUE = {s.name: s for s in (
    Slot("helmet", ("neck", "head"), rigid="head", faces=6000),
    Slot("torso", ("hips", "spine"), faces=8000),
    Slot("shoulders", ("clavicle", "upperarm"), paired=True),
    Slot("gloves", ("forearm", "hand"), paired=True),
    Slot("legs", ("thigh", "shin"), faces=8000),
    Slot("boots", ("foot",), paired=True),
    Slot("weapon", (), rigid="hand", faces=3000),
    Slot("shield", (), rigid="forearm"),
)}
DRAWN = tuple(n for n, s in CATALOGUE.items() if s.roles)

# Index of each name in a slot map (a uint8 picture): 0 is background, the last is bare body.
LABELS = ("background", *DRAWN, "uncovered")


def label(name: str) -> int:
    return LABELS.index(name)


def choose(raw: Mapping[str, Mapping] | Iterable[str]) -> dict[str, dict]:
    """The chosen slots in catalogue order, each as {"route", "split", "prompt"}."""
    items = dict(raw) if isinstance(raw, Mapping) else {name: {} for name in raw}
    if not items:
        raise ValueError("choose at least one slot")
    unknown = sorted(set(items) - set(CATALOGUE))
    if unknown:
        raise ValueError(f"unknown slot {', '.join(unknown)}; the slots are {', '.join(CATALOGUE)}")
    chosen = {}
    for name in CATALOGUE:
        if name not in items:
            continue
        options = dict(items[name] or {})
        route = options.get("route", "C")
        if route not in ROUTES:
            raise ValueError(f"{name}: route must be one of {', '.join(ROUTES)}, not {route!r}")
        split = bool(options.get("split", False))
        if split and name != "shoulders":
            raise ValueError(f"{name}: only the shoulders can be split into two pieces")
        chosen[name] = {"route": route, "split": split, "prompt": str(options.get("prompt", ""))}
    return chosen
