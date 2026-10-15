"""The prompt for one asset: the subject, what the 3D step needs, then the project's style.

    <subject>, <recipe wording for the kind>, <bible style wording>, <bible wording for the kind>

The recipe (`image_to_3dlab/prompt_recipes.json`) is generic: framing, background and
pose that make a picture turn into a clean model. The bible is the project's look. Kept
apart, the same recipes serve every game and a bible never has to repeat them.

No negative prompt: Qwen-Image runs at cfg 1.0 here, where a negative prompt does
nothing, so a style has to be said in positive words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from image_to_3dlab.style.bible import load

RECIPES = Path(__file__).resolve().parents[1] / "prompt_recipes.json"
# Added when style references from the bible go to the generator with the prompt.
REFERENCE_SENTENCE = ("paint it in the same style as the reference pictures: same brushwork, "
                      "colours and proportions, but a new subject")


def load_recipes(path: Path = RECIPES) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean(text: str) -> str:
    """As the studio's composePrompt does: no trailing spaces, commas or full stops."""
    return re.sub(r"[\s,.]+$", "", str(text or "").strip())


def compose(subject: str, kind: str, bible: dict | Path | str | None = None,
            recipes: dict | None = None, references: int = 0) -> dict[str, Any]:
    """The prompt and its labelled parts (`subject`, `recipe`, `style`, `kind`, `references`).

    `bible` is a loaded bible or its folder. `references` is how many style pictures go
    with the prompt; any number above zero adds one sentence pointing at them.
    """
    recipes = recipes or load_recipes()
    goals = recipes["goals"]
    if kind not in goals or not goals[kind].get("wording"):
        known = sorted(k for k, g in goals.items() if g.get("wording"))
        raise ValueError(f"unknown asset kind {kind!r}; choose one of {', '.join(known)}")
    text = _clean(subject)
    if not text:
        raise ValueError("the subject is empty: say what to make")
    if isinstance(bible, (str, Path)):
        bible = load(Path(bible))
    parts = [("subject", text), ("recipe", goals[kind]["wording"])]
    if bible:
        parts.append(("style", (bible.get("style") or {}).get("wording", "")))
        parts.append(("kind", ((bible.get("kinds") or {}).get(kind) or {}).get("wording", "")))
    if references > 0:
        parts.append(("references", REFERENCE_SENTENCE))
    kept = [{"from": name, "text": _clean(value)} for name, value in parts if _clean(value)]
    return {"prompt": ", ".join(p["text"] for p in kept), "parts": kept}
