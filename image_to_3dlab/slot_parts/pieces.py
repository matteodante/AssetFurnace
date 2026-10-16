"""Route C: one clean picture per piece, then the piece in 3D, and where it sits on the concept.

For each chosen slot the flattened cut goes to an edit ("complete this piece", see
`concept.prompts.complete_piece`), a few seeds at a time. The picked picture goes through
Pixal3D, the finish step (retopology to the slot's face budget, the photo projected back)
and `fix_winding.py`: Pixal3D sometimes turns every face of a closed piece inwards. The
piece's box in its own picture, mapped back through the cut's Placement, says where it
sits on the concept, hence on the body (measured in the second route C trial).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_to_3dlab.concept import prompts
from image_to_3dlab.slot_parts import camera, cuts, slots

# What each slot is, when the person gave no words of their own (the trial's wording).
WORDS = {
    "helmet": "a closed helmet",
    "torso": "a breastplate cuirass, complete with both sides where the arms and pauldrons hid it",
    "shoulders": "a single shoulder pauldron, without any part of the breastplate",
    "gloves": "an armoured gauntlet with its forearm guard",
    "legs": "a pair of leg armour pieces with knee guards and hip guards, without boots",
    "boots": "a single armoured boot",
}


HELD_WORDS = {"weapon": "a one-handed sword", "shield": "a round shield"}


def held_prompt(slot: str, words: str = "", style: str = "") -> str:
    """Text to image for a weapon or shield made on its own: alone, upright, front view."""
    what = words.strip() or HELD_WORDS[slot]
    parts = [f"{what}, a single standalone game asset shown alone, upright, front view, centred, complete",
             style.strip(), prompts.BACKGROUND]
    return ", ".join(p for p in parts if p) + "."


@dataclass(frozen=True)
class PieceJob:
    slot: str
    seed: int
    canvas: Path          # the flattened cut sent as the reference picture
    picture: Path         # the completed picture to write
    prompt: str


def picture_path(folder: Path, slot: str, seed: int) -> Path:
    return Path(folder) / slot / f"s{seed}.png"


def model_path(folder: Path, slot: str) -> Path:
    """Pixal3D's GLB for a slot; its camera folder is the same path with `.svviews`."""
    return Path(folder) / slot / "model" / f"{slot}.glb"


def finished_path(folder: Path, slot: str) -> Path:
    return Path(folder) / slot / "model" / f"{slot}_finished.glb"


def picture_jobs(cut_record: dict, cuts_dir: Path, folder: Path, seeds: list[int], style: str = "") -> list[PieceJob]:
    """Every (route C slot, seed) edit to run, in slot order then seed order."""
    if not seeds:
        raise ValueError("give at least one seed")
    jobs = []
    for slot, entry in cut_record["slots"].items():
        if entry.get("route", "C") != "C":
            continue
        words = entry.get("prompt") or WORDS[slot]
        text = prompts.complete_piece(words, style)
        for seed in seeds:
            jobs.append(PieceJob(slot, int(seed), Path(cuts_dir) / entry["picture"], picture_path(folder, slot, seed), text))
    return jobs


def model_commands(picture: Path, folder: Path, slot: str, python: str, scripts: Path) -> list[list[str]]:
    """Pixal3D, the finish to the slot's face budget with the photo projected back, then winding."""
    glb, done = model_path(folder, slot), finished_path(folder, slot)
    views = glb.with_suffix(".svviews")
    return [
        [python, str(scripts / "pixal3d_generate.py"), str(picture), str(glb)],
        [python, str(scripts / "retopo_repaint.py"), str(glb), str(picture), str(done),
         "--faces", str(slots.CATALOGUE[slot].faces), "--skip-paint", "--views", str(views)],
        [python, str(scripts / "fix_winding.py"), str(done)],
    ]


def b0_commands(concept: Path, folder: Path, python: str, scripts: Path, faces: int = 40000) -> list[list[str]]:
    """Route B0: the whole concept through Pixal3D and the finish step, once for every B0 slot."""
    glb = Path(folder) / "b0" / "model" / "concept.glb"
    return [
        [python, str(scripts / "pixal3d_generate.py"), str(concept), str(glb)],
        [python, str(scripts / "retopo_repaint.py"), str(glb), str(concept), str(glb.with_name("concept_finished.glb")),
         "--faces", str(faces), "--skip-paint", "--views", str(glb.with_suffix(".svviews"))],
    ]


def concept_box(piece_alpha: np.ndarray, placement: cuts.Placement) -> list[float]:
    """The completed piece's box in its picture, mapped back onto the concept."""
    return [round(v, 2) for v in placement.to_concept(camera.alpha_box(piece_alpha))]


def parse_picks(text: str) -> dict[str, int]:
    """"helmet=7,boots=3" -> {"helmet": 7, "boots": 3}."""
    picks = {}
    for item in filter(None, (p.strip() for p in (text or "").split(","))):
        slot, _, seed = item.partition("=")
        if slot not in slots.CATALOGUE or not seed.strip().lstrip("-").isdigit():
            raise ValueError(f"a pick is slot=seed, e.g. helmet=7; got {item!r}")
        picks[slot] = int(seed)
    return picks


def write_record(path: Path, data: dict) -> Path:
    Path(path).write_text(json.dumps(data, indent=1) + "\n")
    return Path(path)
