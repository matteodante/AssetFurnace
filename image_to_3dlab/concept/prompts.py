"""The words sent with the pictures, one function per way of using a reference.

Each one is the prompt of the trial of 16/10/2026 that worked, with the subject left
open. Image 1 is always the shape picture; `other_view` with depth adds image 2.

- `redraw`: a structure map (depth or line drawing) and the colours in words. Giving
  the original picture as well made the model copy it, so it is not offered.
- `dress`: clothes on the base body, from its render or from its pose map.
- `add_piece`: one piece of equipment on the base body, the rest untouched.
- `other_view`: the same creature from the side or the back, from its front picture,
  optionally with the depth map of that side (without it the body came out thinner).
- `complete_piece`: one armour piece cut out of a concept and flattened on grey, redrawn
  whole and alone (feature 4, route C; the second trial of 16/10/2026).
"""

from __future__ import annotations

# Plain and even, so the cut-out that follows has an easy edge.
BACKGROUND = "plain light grey background, soft even studio lighting"
STRUCTURES = {"depth": "depth map (lighter is nearer)", "lineart": "line drawing"}
GUIDES = ("body", "pose")
# Only the directions the trial drew; a three-quarter view was never tried.
VIEWS = {"side": "its left side, in profile", "back": "behind"}
MODES = ("redraw", "dress", "add_piece", "other_view", "complete_piece")


def _tail(style: str, *parts: str) -> str:
    words = [p.strip() for p in (style, *parts, BACKGROUND) if p and p.strip()]
    return ", ".join(words) + "."


def redraw(subject: str, colours: str, structure: str = "depth", parts: str = "",
           framing: str = "full body, front view", style: str = "") -> str:
    """A new `subject` filling the outline of image 1 (a structure map), in `colours`."""
    if structure not in STRUCTURES:
        raise ValueError(f"structure is one of {', '.join(STRUCTURES)}, got {structure!r}")
    if not colours.strip():
        raise ValueError("a redraw needs its colours in words: the original picture is not sent")
    detail = f" ({parts.strip()})" if parts.strip() else ""
    return (f"Image 1 is a {STRUCTURES[structure]} of a {subject}. Draw a new, original {subject} that "
            f"fills exactly the outline and pose of image 1, with {colours.strip()}, designed from "
            f"scratch{detail}. " + _tail(style, framing))


def dress(outfit: str, guide: str = "body", style: str = "") -> str:
    """`outfit` on the base body of image 1: its render (`body`) or its OpenPose map (`pose`)."""
    if guide not in GUIDES:
        raise ValueError(f"guide is one of {', '.join(GUIDES)}, got {guide!r}")
    if guide == "body":
        return (f"Dress the character in image 1 as {outfit}. Keep exactly the same body, proportions, "
                f"head size, T-pose, arm and leg positions and framing as image 1; only the clothing and "
                f"hair change. " + _tail(style, "full body, front view"))
    return (f"Image 1 is an OpenPose skeleton. Draw {outfit}, standing exactly in the pose of the "
            f"skeleton: arms straight out to the sides in a T-pose, legs slightly apart, with the head, "
            f"shoulders, elbows, wrists, hips, knees and ankles exactly at the skeleton's joints, at the "
            f"same size and position in the frame. " + _tail(style, "full body, front view"))


def add_piece(piece: str, style: str = "", wearing: str = "bodysuit") -> str:
    """`piece` drawn onto the character of image 1, everything else kept, the clothes it is
    `wearing` too (a base body's own, `slot_parts.body.dressing_words`)."""
    return (f"Add {piece} to the character in image 1, fitted closely to the body. Change nothing else: "
            f"same body, {wearing}, proportions, T-pose, framing and background. " + _tail(style))


def complete_piece(piece: str, style: str = "") -> str:
    """The cut-out `piece` of image 1 completed: hidden and cut-off parts filled in, nothing inside.

    The edit keeps the piece where and how large it found it, which is how the piece is
    placed back on the body afterwards.
    """
    return (f"Complete this cut-out armour piece into {piece}: fill in every part that was hidden or cut "
            f"off, keep its design, colours and style, no body, no person, nothing inside, front view, "
            f"centred. " + _tail(style))


def other_view(view: str, depth: bool = False, style: str = "") -> str:
    """The creature of image 1 (its front) seen from `view`; with `depth`, image 2 is that side's depth."""
    if view not in VIEWS:
        raise ValueError(f"view is one of {', '.join(VIEWS)}, got {view!r}")
    direction = VIEWS[view]
    text = (f"Image 1 shows a creature from the front. Draw exactly the same creature, with the same "
            f"design, colours, markings and proportions, seen from {direction}, full body, orthographic "
            f"{view} view, standing in the same pose. " + _tail(style))
    if depth:
        text += (f" Image 2 is a depth map of its body seen from {direction} (lighter is nearer): "
                 f"follow its outline exactly.")
    return text


def shape_pictures(mode: str, depth: bool = False) -> int:
    """How many shape pictures lead the references; style pictures, if any, follow them."""
    if mode not in MODES:
        raise ValueError(f"mode is one of {', '.join(MODES)}, got {mode!r}")
    return 2 if mode == "other_view" and depth else 1


def style_sentence(first: int, count: int) -> str:
    """The words for `count` style pictures that follow the shape pictures, from image `first`."""
    if count <= 0:
        return ""
    if count == 1:
        return (f" Image {first} is a style reference: paint it in the same style (same brushwork, colours "
                f"and proportions), but do not copy its subject.")
    which = f"{first} and {first + 1}" if count == 2 else f"{first} to {first + count - 1}"
    return (f" Images {which} are style references: paint it in the same style (same brushwork, colours "
            f"and proportions), but do not copy their subjects.")
