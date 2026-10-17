"""Bone roles of a humanoid base body, and the bone segments a slot map is drawn from.

The roles come from the creature-family reading of the skeleton (`creature_family.classify`):
its tree shape and joint positions, never bone names, so a SkinTokens rig (`bone_0..N`)
works as it is. A bone runs from its head to its child's head: glTF tails are arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Where along an upper arm the shoulder piece stops, and along a shin the boot starts.
# Measured on the trial body: the slot map needed no touch-ups with these.
SHOULDER_T = 0.45
BOOT_T = 0.72


@dataclass(frozen=True)
class Segment:
    bone: str
    role: str
    side: str            # "L", "R" or "" on the middle line
    head: np.ndarray
    end: np.ndarray


def arm_roles(n: int) -> list[str]:
    """Roles along an arm chain of n bones: a clavicle only when there are four or more."""
    names = ["clavicle", "upperarm", "forearm", "hand"] if n >= 4 else ["upperarm", "forearm", "hand"]
    return [names[min(i, len(names) - 1)] for i in range(n)]


def bone_roles(found: dict) -> dict[str, tuple[str, int, str]]:
    """{bone: (role, index along its chain, side)} from a `creature_family.classify` result."""
    if len(found.get("arms", [])) != 2 or len(found.get("legs", [])) != 2:
        raise ValueError(f"slots need a humanoid base body; this skeleton reads as {found.get('detail', '?')}")
    roles: dict[str, tuple[str, int, str]] = {}
    spine = found.get("spine", [])
    for i, bone in enumerate(spine):
        roles[bone] = ("hips" if i == 0 else "spine", i, "")
    for i, bone in enumerate(found.get("neck", [])):
        roles[bone] = ("neck" if i == 0 else "head", i, "")
    for arm in found["arms"]:
        for i, (bone, role) in enumerate(zip(arm["bones"], arm_roles(len(arm["bones"])))):
            roles[bone] = (role, i, arm["side"])
    for leg in found["legs"]:
        for i, bone in enumerate(leg["bones"]):
            roles[bone] = (["thigh", "shin", "foot"][min(i, 2)], i, leg["side"])
    return roles


def inherit_roles(roles: dict, parents: dict[str, str | None]) -> dict[str, tuple[str, int, str]]:
    """Bones the classifier left out take the role and side of their nearest classified ancestor.

    The classifier keeps one chain per limb, so the other fingers of a hand are not in it;
    left out, they read as spine, the gloves lost them and B0 cut them into the breastplate.
    Further down a chain, the index goes on counting. A bone with no classified ancestor
    (a root above the hips) stays out.
    """
    out = {b: tuple(r) for b, r in roles.items()}
    for bone in parents:
        chain, up = [], bone
        while up is not None and up not in out:
            chain.append(up)
            up = parents.get(up)
        if up is None:
            continue
        role, index, side = out[up]
        for depth, b in enumerate(reversed(chain), start=1):
            out[b] = (role, index + depth, side)
    return out


def slot_for(role: str, t: float, shoulder_t: float = SHOULDER_T, boot_t: float = BOOT_T) -> str:
    """The slot of a point nearest a bone of `role`, at fraction t (0 head, 1 end) along it."""
    if role in ("neck", "head"):
        return "helmet"
    if role in ("hips", "spine"):
        return "torso"
    if role == "clavicle" or (role == "upperarm" and t < shoulder_t):
        return "shoulders"
    if role == "upperarm":
        return "uncovered"
    if role in ("forearm", "hand"):
        return "gloves"
    if role == "thigh" or (role == "shin" and t < boot_t):
        return "legs"
    return "boots"


def joint_segments(parents: dict[str, str | None], heads: dict, roles: dict) -> list[Segment]:
    """Every bone as a segment from its head to its child's head (the mean of its children).

    The hips and spine end at the next spine bone, not between the thighs and the arms. A
    bone with no children is a point.
    """
    roles = inherit_roles(roles, parents)
    kids: dict[str | None, list[str]] = {}
    for bone, parent in parents.items():
        kids.setdefault(parent, []).append(bone)
    out = []
    for bone in parents:
        role, _, side = roles.get(bone, ("spine", 0, ""))
        children = kids.get(bone, [])
        if role in ("hips", "spine") and len(children) > 1:
            children = [c for c in children if roles.get(c, ("",))[0] in ("spine", "neck")] or children
        head = np.asarray(heads[bone], float)
        end = np.mean([np.asarray(heads[c], float) for c in children], axis=0) if children else head
        out.append(Segment(bone, role, side, head, end))
    return out
