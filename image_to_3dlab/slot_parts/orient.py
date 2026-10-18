"""A weapon's picture turned handle down before it becomes 3D, checked by the picture judge.

A weapon is made in its canonical pose, handle at the bottom: its grip is found on the upright
picture (`grip.py`) and the Unreal mesh points its blade or head along +Z from the grip. The
picture is asked for that way, but the picture model
drew the first real set's sword point down in three seeds out of three. So Qwen3-VL (the
picture judge, `image_to_3dlab/style/judge.py`) is asked where the handle is, at the top or
at the bottom, and a picture with it at the top is flipped top to bottom. A learned check,
not a rule about shapes: an axe, a mace and a sword are told apart by what they look like.

Each picture is checked once (`<picture>.orient.json`). Without the judge the pictures stay
as drawn, and the run says so.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

BOTTOM = ("This picture shows one weapon. Is its handle, the part a hand holds, at the bottom of the picture? "
          "Answer yes or no.")
TOP = ("This picture shows one weapon. Is its handle, the part a hand holds, at the top of the picture? "
       "Answer yes or no.")


def record_path(picture: Path) -> Path:
    return Path(picture).with_name(Path(picture).name + ".orient.json")


def handle_at_top(picture: Path, ask: Callable[[Path, str], float]) -> tuple[bool, dict]:
    """Whether the judge sees the handle at the top, and its two answers (yes probabilities)."""
    top, bottom = float(ask(picture, TOP)), float(ask(picture, BOTTOM))
    return top > bottom, {"top": round(top, 3), "bottom": round(bottom, 3)}


def orient_held(pictures: list[Path], ask: Callable[[Path, str], float] | None) -> dict[Path, bool | None]:
    """{picture: turned (True/False), or None when there is no judge}; each picture once."""
    from PIL import Image, ImageOps

    from image_to_3dlab.slot_parts import candidates

    out: dict[Path, bool | None] = {}
    for picture in pictures:
        done = record_path(picture)
        if done.is_file():
            out[picture] = json.loads(done.read_text(encoding="utf-8")).get("turned")
            continue
        if ask is None:
            out[picture] = None
            continue
        top, answers = handle_at_top(picture, ask)
        if top:
            with Image.open(picture) as opened:
                ImageOps.flip(opened.copy()).save(picture)
        done.write_text(json.dumps({"handle": "top" if top else "bottom", "turned": top, "answers": answers,
                                    "by": candidates.JUDGE}, indent=1) + "\n", encoding="utf-8")
        out[picture] = top
    if ask is None and any(v is None for v in out.values()):
        print("weapon pictures kept as drawn: the picture judge is not installed to check that the handle is at the "
              "bottom (Setup & Status installs it)", flush=True)
    return out


def with_judge(pictures: list[Path]) -> dict[Path, bool | None]:
    """orient_held with the picture judge when it is installed and the GPU is free."""
    from image_to_3dlab.slot_parts import candidates

    waiting = [p for p in pictures if not record_path(p).is_file()]
    if not waiting:
        return orient_held(pictures, None)
    return candidates.with_judge(lambda server: orient_held(pictures, candidates.make_asker(server.post)
                                                            if server else None),
                                 "the weapon's pictures are kept as drawn")[0]
