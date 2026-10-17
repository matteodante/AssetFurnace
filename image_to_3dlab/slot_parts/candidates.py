"""Several pictures per piece, and the one that goes to 3D.

Route C makes a few pictures of each piece (one per seed) from its cut. Only one goes on
to Pixal3D, and the wrong one costs a 3D run and a bad piece: a glove where the boot
should be, a pauldron with half the breastplate still on it. Each candidate is scored
twice:

- **shape**: its outline (cut out by the studio's matting model) against the slot's own
  mask in the cut. A completed piece may grow where the cut was hidden, so it is measured
  as how much of the slot it covers, less how much of it lies well outside the slot;
- **is it the piece**: Qwen3-VL (the studio's judge model, `style.judge`) answers yes or no
  to "is this one helmet / boot / ..., alone?", read as the probability of yes.

The best score is picked by itself; the person can change any pick on the Pieces page
before the 3D starts. Without Qwen3-VL the shape decides alone, and the record says so.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from image_to_3dlab.slot_parts import pieces

# What each slot's picture must show, in the words the yes/no question uses.
NOUNS = {
    "helmet": "one helmet, hood or hat, with no face or head inside",
    "torso": "one breastplate or chest armour, without shoulder pauldrons",
    "shoulders": "one single shoulder pauldron",
    "gloves": "one single armoured glove or gauntlet",
    "legs": "leg armour for both legs, without boots",
    "boots": "one single boot",
}
QUESTION = ("Is this picture {noun}, shown on its own and complete, with no person, no body and no other "
            "piece of armour? Answer with yes or no only.")
# A pauldron's cut takes in some breastplate, back plate or arm, and pictures that keep it pass the
# plain question; this one caught them (11 of 12 shoulder pictures right, against 10).
QUESTIONS = {"shoulders": ("Look closely. Is every part of this picture the same single shoulder pauldron, with "
                           "nothing else joined to it (no chest plate, no back plate, no strap across, no second "
                           "pauldron, no arm guard)? Answer with yes or no only.")}
DOUBT = 0.5  # below this for every candidate, none is taken to be the piece
GROW = 0.06  # how far round the slot (share of the canvas side) a completed piece may grow freely


def question(slot: str) -> str:
    return QUESTIONS.get(slot) or QUESTION.format(noun=NOUNS[slot])


def _grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    if pixels <= 0:
        return mask
    return binary_dilation(mask, iterations=pixels)


def shape_score(candidate: np.ndarray, slot: np.ndarray, grow: float = GROW) -> dict:
    """{cover, spill, score}: the share of the slot the candidate covers, the share of the
    candidate well outside the slot, and cover x (1 - spill)."""
    candidate, slot = np.asarray(candidate, bool), np.asarray(slot, bool)
    if candidate.shape != slot.shape:
        raise ValueError(f"the candidate is {candidate.shape}, the slot mask {slot.shape}")
    if not slot.any() or not candidate.any():
        return {"cover": 0.0, "spill": 1.0, "score": 0.0}
    cover = float((candidate & slot).sum() / slot.sum())
    spill = float((candidate & ~_grow(slot, round(grow * max(slot.shape)))).sum() / candidate.sum())
    return {"cover": round(cover, 4), "spill": round(spill, 4), "score": round(cover * (1.0 - spill), 4)}


def combined(shape: float, is_piece: float | None) -> float:
    """The number to rank by: the judge's yes to two decimals; the shape alone without a judge.

    The shape only breaks ties (`best`): it rewards a picture that copies its cut, and a
    pauldron's cut holds some breastplate. Weighed in, it let a pauldron with a chest plate
    still on it win; as a tie-break, the pick was right in 15 slots of 15.
    """
    if is_piece is None:
        return round(shape, 4)
    return round(is_piece, 2)


def best(scored: dict[int, dict]) -> int:
    """The seed with the highest score, then the better shape, then the lowest seed."""
    return min(scored, key=lambda seed: (-scored[seed]["score"], -scored[seed].get("shape", {}).get("score", 0.0), seed))


def ask_request(picture: Path, text: str) -> dict:
    """One picture and one yes/no question, one token back with its probabilities."""
    from image_to_3dlab.style import judge

    content = [{"type": "image_url", "image_url": {"url": judge._data_url(picture)}}, {"type": "text", "text": text}]
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 20}


def make_asker(post: Callable[[dict], dict]) -> Callable[[Path, str], float]:
    from image_to_3dlab.style.judge import yes_probability

    def ask(picture: Path, text: str) -> float:
        reply = post(ask_request(picture, text))
        tokens = reply["choices"][0]["logprobs"]["content"]
        return yes_probability(tokens[0]["top_logprobs"] if tokens else [])

    return ask


def outline(picture: Path, cut_out=None) -> np.ndarray:
    """The candidate's subject: the matting model on its grey ground."""
    from PIL import Image

    if cut_out is None:
        from image_to_3dlab.matte import cut_out
    with Image.open(picture) as image:
        rgba, _ = cut_out(image)
    return np.asarray(rgba)[..., 3] > 127


def score_slot(slot: str, slot_mask: np.ndarray, pictures: dict[int, Path], ask=None, cut_out=None) -> dict:
    """Every candidate of one slot scored, and the automatic pick."""
    scored = {}
    for seed, picture in sorted(pictures.items()):
        shape = shape_score(outline(picture, cut_out), slot_mask)
        is_piece = round(ask(picture, question(slot)), 4) if ask else None
        scored[seed] = {"picture": str(picture), "shape": shape, "is_piece": is_piece,
                        "score": combined(shape["score"], is_piece)}
    auto = best(scored)
    # every picture judged not to be the piece (a breastplate drawn with pauldrons each time):
    # the page says so, and more seeds or other words are the way out
    doubt = ask is not None and all(c["is_piece"] < DOUBT for c in scored.values())
    return {"candidates": scored, "auto": auto, "picked": auto, "by": "auto", "doubt": doubt}


def slot_mask(cuts_dir: Path, entry: dict) -> np.ndarray:
    """The slot's own cut where the canvas holds it: scaled and padded as `cuts.flatten` did."""
    from PIL import Image

    place = entry["placement"]
    with Image.open(Path(cuts_dir) / entry["cut"]) as cut:
        alpha = cut.convert("RGBA").getchannel("A")
        w, h = alpha.size
        scaled = alpha.resize((max(1, round(w * place["scale"])), max(1, round(h * place["scale"]))), Image.NEAREST)
    canvas = Image.new("L", (place["size"], place["size"]), 0)
    canvas.paste(scaled, tuple(place["pad"]))
    return np.asarray(canvas) > 127


def score_all(cut_record: dict, cuts_dir: Path, folder: Path, seeds: list[int], ask=None, cut_out=None) -> dict:
    """{slot: scored} for every route C slot of a cuts.json whose pictures exist."""
    out = {}
    for slot, entry in cut_record["slots"].items():
        if entry.get("route", "C") != "C":
            continue
        pictures = {s: pieces.picture_path(folder, slot, s) for s in seeds}
        pictures = {s: p for s, p in pictures.items() if p.is_file()}
        if pictures:
            out[slot] = score_slot(slot, slot_mask(cuts_dir, entry), pictures, ask, cut_out)
    return out


def path(folder: Path) -> Path:
    return Path(folder) / "candidates.json"


def write(folder: Path, slots_scored: dict, judge: str) -> Path:
    data = {"schema_version": 1, "judge": judge, "slots": slots_scored}
    target = path(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return target


def load(folder: Path) -> dict:
    target = path(folder)
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {"slots": {}}


def choose(folder: Path, picks: dict[str, int]) -> dict:
    """The person's picks over the automatic ones; a pick must be one of the slot's candidates."""
    data = load(folder)
    for slot, seed in picks.items():
        entry = data["slots"].get(slot)
        if entry is None:
            raise ValueError(f"{slot}: no candidates to choose from")
        if str(seed) not in {str(s) for s in entry["candidates"]}:
            raise ValueError(f"{slot}: seed {seed} is not one of its pictures")
        entry["picked"], entry["by"] = int(seed), "person" if int(seed) != int(entry["auto"]) else "auto"
    path(folder).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


def picks(folder: Path) -> dict[str, int]:
    """{slot: the seed to make in 3D}, from candidates.json."""
    return {slot: int(entry["picked"]) for slot, entry in load(folder)["slots"].items()}
