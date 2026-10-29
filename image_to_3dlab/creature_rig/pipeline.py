"""The skeleton step of the creature auto-rig: drawn skeletons (or none) in, `joints.json` out.

    joints_from_draws(W, [W/draws/draw0.json, ...])   the learned rigger's draws
    joints_without_unirig(W)                          the centre lines of the body alone

Both read `W/shell.npz` and try the many-legs rig first: six or more legs on the floor are
traced from the body and the draws are not used. Otherwise every draw is cleaned and
scored, the best is kept, a contact it missed gets a centre-line leg, the bones are named
by body plan and repaired. `joints.json` holds the bones, every draw's score and notes on
what was changed and why.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import manylegs
from .body import MIDDLE_DRAWN, analyse, centre_line_bones, name_bones
from .draws import best, clean, read_draw, score
from .repairs import add_missing_legs, exact_twins, middle_plane, repair_bones
from .shell import Shell
from .skeleton import Bones, all_samples


def _result(shell: Shell, body: str, bones: Bones, draws: list[dict], chosen: int | None, notes: list[str],
            facing: str = "-Y", plan: str | None = None) -> dict:
    still = shell.unreached(all_samples([b for b in bones if b.deform]))
    if still:
        notes = [*notes, f"{len(still)} floor contact(s) still without a bone reaching them"]
    out = {"body": body, "height": shell.height, "facing": facing, "bones": [b.to_json() for b in bones],
           "draws": draws, "chosen": chosen, "notes": notes}
    if plan is not None:
        out["plan"] = plan
    return out


def _write(work: Path, result: dict) -> dict:
    (Path(work) / "joints.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def _many_legs(shell: Shell) -> dict | None:
    many = manylegs.rig(shell)
    if many is None:
        return None
    bones, notes, facing = many
    notes += exact_twins(bones, shell)
    moved = middle_plane(bones, shell)
    if moved:
        notes.append(f"{moved} middle bone(s) put on the middle plane")
    return _result(shell, "manylegs", bones, [], None, notes, facing)


def _centre_line(shell: Shell, notes: list[str]) -> tuple[Bones, list[str], str]:
    bones, body = centre_line_bones(shell)
    notes = [*notes, *body.notes]
    notes += repair_bones(bones, shell)
    return bones, notes, body.kind


def joints_without_unirig(work: Path) -> dict:
    """`W/joints.json` from the body alone: many legs, else its centre lines."""
    shell = Shell.load(work)
    result = _many_legs(shell)
    if result is None:
        bones, notes, plan = _centre_line(shell, ["no drawn skeleton: the centre-line skeleton of the body"])
        result = _result(shell, "centreline", bones, [], None, notes, plan=plan)
    return _write(work, result)


def joints_from_draws(work: Path, draws: list[Path], prefer: int | None = None) -> dict:
    """`W/joints.json` from the learned rigger's draws: each cleaned and scored, the best
    kept, repaired and named. `chosen` is the index of the kept draw in `draws`. `prefer`:
    a draw a person picked over the score (the studio's "Use draw N"), tried first."""
    shell = Shell.load(work)
    result = _many_legs(shell)
    if result is not None:
        return _write(work, result)
    records, trees, scores = [], [], []
    for path in draws:
        record = {"file": Path(path).name}
        try:
            tree, info = clean(read_draw(Path(path)), shell)
            s = score(tree, shell)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:   # a broken draw is skipped
            records.append({**record, "score": None, "error": f"{type(exc).__name__}: {exc}"})
            trees.append(None)
            scores.append(None)
            continue
        records.append({**record, **{k: v for k, v in s.items() if k != "missed"}, **info})
        trees.append(tree)
        scores.append(s)
    notes: list[str] = []
    ranked = sorted((s["score"], k) for k, s in enumerate(scores) if s is not None)
    if prefer is not None and 0 <= prefer < len(scores) and scores[prefer] is not None:
        ranked = [(scores[prefer]["score"], prefer)] + [r for r in ranked if r[1] != prefer]
        notes.append(f"draw {prefer} picked by hand over the score")
    centre = None
    for _, k in ranked:
        tree, added = trees[k], []
        try:
            body = analyse(tree, shell, middle=MIDDLE_DRAWN)
            missed = scores[k]["missed"]
            if missed:
                if centre is None:
                    centre = centre_line_bones(shell)[0]
                tree, body, added = add_missing_legs(tree, body, shell, missed, centre, MIDDLE_DRAWN)
            bones, _ = name_bones(tree, body, shell)
        except (ValueError, IndexError, KeyError, StopIteration) as exc:    # an odd draw: try the next best
            notes.append(f"draw {k} could not be read as a body ({type(exc).__name__}: {exc})")
            continue
        notes = [f"draw {k} kept ({records[k]['file']}, score {scores[k]['score']:.1f}): {body.kind}",
                 *notes, *added, *body.notes]
        notes += repair_bones(bones, shell)
        return _write(work, _result(shell, body.kind, bones, records, k, notes))
    if best(scores) is None:
        notes.append("no usable draw")
    bones, notes, plan = _centre_line(shell, [*notes, "no draw could be used: the centre-line skeleton instead"])
    return _write(work, _result(shell, "centreline", bones, records, None, notes, plan=plan))
