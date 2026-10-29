"""The words a video model gets for a two-camera clip: one action sentence, the rest fixed.

The skeleton is AnyMove's `tools/make_prompt.py` (Patch, Apache-2.0): it pins down what
video models drift on (both halves one moment from two locked cameras, the model exactly
as rendered, every limb keeping its colour, a fixed body count, the creature centred) and
only the action changes from move to move. Two rules come from video2rig's notes on
Seedance (the ideas only; that project is AGPL): say how the move ends in both views, or
models roll the creature back; and ask for clean frames, since a dust cloud or a light
trail hides the limbs from the reader while it still reads them with confidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TEMPLATE = (
    "Animate the exact {noun} game model shown twice in this split-screen image. Left: a front camera, the "
    "{noun} faces the viewer. Right: a side camera on its right side, the {noun} faces right. Both are "
    "locked-off cameras filming the same {noun} at the same moment, so every movement happens in both views at "
    "exactly the same time, frame for frame. Keep the model exactly as it is{describe}. Its limbs are "
    "colour-coded and the colours never change or swap: {colour_code}. {count}Action: {action}{end} It stays on "
    "the same spot, centred in both views, and its whole body stays inside each view. One continuous "
    "{seconds}-second shot at natural speed, no slow motion. The cameras never move, cut, pan or zoom; the plain "
    "grey studio background, the floor band and the dark vertical divider stay fixed. Every frame is crisp and "
    "clean: no motion blur, dust, smoke, sparks, trails or other effects, and every limb is clearly drawn."
)


class PromptError(ValueError):
    pass


def _sentence(text: str) -> str:
    text = " ".join(str(text).split())
    return text if not text or text.endswith((".", "!", "?")) else text + "."


def make_prompt(creature: Mapping[str, Any], action: str, end: str | None = None, seconds: int = 5) -> str:
    """The prompt for one move. `creature` is a prepared creature's `creature.json`
    (`noun`, `colour_code`, and optionally `describe` and `count`)."""
    action = _sentence(action)
    if not action:
        raise PromptError("say what the creature does: one sentence")
    if not creature.get("noun") or not creature.get("colour_code"):
        raise PromptError("creature.json needs a noun and a colour code: prepare the creature first")
    describe = creature.get("describe")
    return TEMPLATE.format(
        noun=creature["noun"], describe=f": {describe}" if describe else "",
        colour_code=creature["colour_code"], count=_sentence(creature["count"]) + " " if creature.get("count") else "",
        action=action, end=f" In both views it ends {_sentence(end)}" if end else "", seconds=int(seconds))
