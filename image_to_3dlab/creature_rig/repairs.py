"""Repairs to the chosen skeleton, each measured on the body.

* a floor contact no bone reaches gets the leg of the centre-line skeleton;
* twin limbs become exact mirrors where the body round them mirrors itself, or at least
  equally long where it does not (a leg stepped forward);
* middle bones go onto the middle plane;
* a standing leg's last bone runs along its sole to the toe tips;
* a head drawn ending in an ear runs from the skull to the front of the face;
* a mouth modelled open gets a jaw;
* a short bone on one side of the head only goes (one brow ridge).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .body import Body, analyse, name_bones
from .shell import FORWARD, Shell
from .skeleton import (
    LIMB_KINDS,
    Bone,
    Bones,
    Tree,
    below,
    children_of,
    in_order,
    side_of,
    stem_of,
)

LEG_ENDS = ("hind", "front", "biped", "chain")
LEG_LOW = 0.15          # a centre-line leg ending this low ...
LEG_NEAR = 0.08         # ... and this close beside a missed contact replaces it
STUB_NEAR = 0.10        # a drawn end this close to the new leg ...
STUB_ROOT = 0.20        # ... leaving the trunk this close to its top was that leg drawn short
TWIN_MIRROR = 0.8       # the body round both chains mirrors itself this well: exact mirrors
TWIN_TOTAL = 0.75       # otherwise lengths made equal if the chains' totals compare this well ...
TWIN_BONE = 0.4         # ... and every bone pair this well
MIDDLE_NEAR = 0.08      # a middle bone this close to the plane goes onto it
SOLE_END = 0.05         # leg ends this low stand on a sole
SOLE_LOW = 0.03         # the sole: vertices this low ...
SOLE_NEAR = 0.12        # ... this close beside the bone
SOLE_AHEAD = 0.02       # used only if the toes reach this far ahead of the bone's start
FACE_OFF = 0.04         # a head end this far off the plane ended in an ear
FACE_SIBLING = 0.25
FACE_NEAR = 0.35
HEAD_SHORT = 0.22       # a head bone this short ...
HEAD_OFF = 0.04         # ... this far off the middle ...
HEAD_TWIN = 0.06        # ... with no twin end this close to its mirror image goes
HEAD_PARTS = ("head", "part", "neck", "chain")


# --- missing legs ------------------------------------------------------------------------


def legs_for_contacts(centre: Bones, shell: Shell, missed: list[int]) -> list[list[Bone]]:
    """Chains of the centre-line skeleton standing on the missed contacts: each limb end
    bone ending low and close beside one, with the limb bones above it."""
    h = shell.height
    kids = children_of(centre)
    by = {b.name: b for b in centre}
    out = []
    for b in centre:
        if b.kind not in LEG_ENDS or kids.get(b.name) or b.tail[2] > shell.floor + LEG_LOW * h:
            continue
        if not any(np.linalg.norm(shell.contacts[k] - b.tail[:2], axis=1).min() < LEG_NEAR * h for k in missed):
            continue
        chain = [b]
        while chain[-1].parent in by and by[chain[-1].parent].kind in LIMB_KINDS:
            chain.append(by[chain[-1].parent])
        out.append(chain[::-1])
    return out


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(p[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1)


def graft_legs(tree: Tree, body: Body, bones: Bones, joint_of: dict[str, int], chains: list[list[Bone]],
               shell: Shell) -> tuple[Tree, list[str]]:
    """The chains hung into the drawn tree from the trunk or neck bone whose middle is
    nearest each chain's top. A drawn branch ending close beside a new leg was that leg
    drawn short (to the knee): it goes, or it would pair with its twin as a pair of arms."""
    h = shell.height
    trunk = [b for b in bones if b.kind in ("spine", "neck") and b.name in joint_of]
    if not trunk or not chains:
        return tree, []
    notes, segments, tops = [], [], []
    for chain in chains:
        top = chain[0].head
        host = min(trunk, key=lambda b: float(np.linalg.norm((b.head + b.tail) / 2 - top)))
        up = joint_of[host.name]
        for b in chain:
            tree, up = tree.add(b.head, up, tip=False)
        tree, _ = tree.add(chain[-1].tail, up, tip=True)
        segments += [(b.head, b.tail) for b in chain]
        tops.append(top)
        notes.append(f"floor contact without a leg: {chain[0].name}..{chain[-1].name} added from the centre line")
    a = np.array([s[0] for s in segments])
    b = np.array([s[1] for s in segments])
    spine = set(body.spine)
    kids = tree.children()
    stale = []
    for e in [limb.end for pair in body.upper_pairs for limb in pair] + [limb.end for limb in body.singles + body.props]:
        if e in (body.head, body.tail) or _segment_distance(tree.points[[e]], a, b).min() > STUB_NEAR * h:
            continue
        n, branch = e, []
        while tree.parent[n] >= 0 and n not in spine and len(kids[n]) <= 1:
            branch.append(n)
            n = int(tree.parent[n])
        # leaving the trunk where the new leg does: a hand hanging beside a thigh does not
        if min(np.linalg.norm(tree.points[n] - t) for t in tops) < STUB_ROOT * h:
            stale += branch
    if stale:
        tree = tree.remove(stale)
        notes.append("a leg drawn short beside an added one: removed")
    return tree, notes


def add_missing_legs(tree: Tree, body: Body, shell: Shell, missed: list[int], centre: Bones,
                     middle: float) -> tuple[Tree, Body, list[str]]:
    """Missed contacts get centre-line legs, then the body plan is decided again on the
    whole tree, so the new leg pairs with its drawn twin and the plan counts it."""
    if not missed:
        return tree, body, []
    chains = legs_for_contacts(centre, shell, missed)
    if not chains:
        return tree, body, [f"{len(missed)} floor contact(s) without a leg, none found on the centre line either"]
    bones, joint_of = name_bones(tree, body, shell)
    tree, notes = graft_legs(tree, body, bones, joint_of, chains, shell)
    return tree, analyse(tree, shell, middle=middle), notes


# --- twins and the middle plane ----------------------------------------------------------


def _twin(name: str) -> str:
    return stem_of(name) + ".R"


def exact_twins(bones: Bones, shell: Shell) -> list[str]:
    """Each left chain matched to its right chain by name. Where the body round both is
    mirror-symmetric (TWIN_MIRROR) both become the mean of left and mirrored right; where
    it is not but the chains compare in length, only the lengths are made equal and each
    keeps its pose; otherwise both stay as drawn. A top bone only one side has (a scapula
    drawn on one side) is stepped over, the chains below it still twinned."""
    by = {b.name: b for b in bones}
    kids = children_of(bones)
    mirrored, equal, kept = [], [], []
    tops = [b for b in bones if b.side == "L" and (b.parent is None or side_of(b.parent) != "L")]
    for top in tops:
        t = top
        while _twin(t.name) not in by:
            down = [c for c in kids.get(t.name, []) if c.side == "L"]
            if len(down) != 1:
                t = None
                break
            t = down[0]
        if t is None:
            kept.append(stem_of(top.name))
            continue
        left = [b for b in below(bones, t.name) if b.side == "L"]
        right = [by.get(_twin(b.name)) for b in left]
        if any(r is None for r in right):
            kept.append(stem_of(t.name))
            continue
        pts = lambda chain: np.concatenate([b.samples(5) for b in chain])
        sym = min(shell.symmetric_near(pts(left)), shell.symmetric_near(pts(right)))
        if sym >= TWIN_MIRROR:
            for a, b in zip(left, right):
                for key in ("head", "tail"):
                    m = (getattr(a, key) + shell.mirror(getattr(b, key))) / 2
                    setattr(a, key, m)
                    setattr(b, key, shell.mirror(m))
            mirrored.append(stem_of(t.name))
            continue
        la, lb = np.array([b.length for b in left]), np.array([b.length for b in right])
        total = min(la.sum(), lb.sum()) / max(la.sum(), lb.sum(), 1e-12)
        each = np.min(np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1e-12))
        if total < TWIN_TOTAL or each < TWIN_BONE:
            kept.append(stem_of(t.name))
            continue
        mean = (la + lb) / 2
        for chain in (left, right):
            _set_lengths(chain, mean)
        equal.append(stem_of(t.name))
    notes = []
    if mirrored:
        notes.append(f"twin limbs made exact mirrors: {', '.join(mirrored)}")
    if equal:
        notes.append(f"twin limbs made equally long (posed differently): {', '.join(equal)}")
    if kept:
        notes.append(f"limbs kept as drawn (no twin to match): {', '.join(kept)}")
    return notes


def _set_lengths(chain: list[Bone], lengths: np.ndarray) -> None:
    """Each bone its new length along its own direction; a child that started at its
    parent's end still does, one hung elsewhere on it moves as the parent's start moved."""
    moved: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for b, length in zip(chain, lengths):
        old_head, old_tail = b.head.copy(), b.tail.copy()
        head = old_head
        if b.parent in moved:
            ph, pt, nh, nt = moved[b.parent]
            head = nt.copy() if np.linalg.norm(old_head - pt) < 1e-6 else old_head + (nh - ph)
        u = old_tail - old_head
        u = u / max(np.linalg.norm(u), 1e-12)
        b.head, b.tail = head, head + u * length
        moved[b.name] = (old_head, old_tail, b.head.copy(), b.tail.copy())


def middle_plane(bones: Bones, shell: Shell) -> int:
    """Middle bones (no side) with both ends within MIDDLE_NEAR of the plane, where the
    body round them mirrors itself, put onto the plane. Returns how many moved."""
    h = shell.height
    moved = 0
    for b in bones:
        if b.side or b.kind not in ("root", "spine", "neck", "head", "tail", "jaw"):
            continue
        if max(abs(shell.lateral(b.head)), abs(shell.lateral(b.tail))) > MIDDLE_NEAR * h:
            continue
        if shell.symmetric_near(b.samples(5)) < TWIN_MIRROR:
            continue
        b.head[0] = b.tail[0] = shell.mid
        moved += 1
    return moved


# --- feet and head -----------------------------------------------------------------------


def soles(bones: Bones, shell: Shell) -> list[str]:
    """A standing leg's last bone runs along its sole to the toe tips. The floor-near skin
    goes to the nearest such bone (seen from above); the bone's new end is the front 3% of
    its share. A wolf's toes were drawn as a stub straight down at the back of the paw, so
    nothing said where the paw pointed."""
    h = shell.height
    kids = children_of(bones)
    ends = [b for b in bones if not kids.get(b.name) and b.kind in LEG_ENDS and b.tail[2] < shell.floor + SOLE_END * h]
    if not ends:
        return []
    low = shell.verts[shell.verts[:, 2] < shell.floor + SOLE_LOW * h]
    if not len(low):
        return []
    low = np.unique(np.round(low / (0.1 * shell.size)), axis=0) * (0.1 * shell.size)
    a = np.array([b.head[:2] for b in ends])
    c = np.array([b.tail[:2] for b in ends])
    d = _segment_distance(low[:, :2], a, c).T                       # (ends, vertices)
    owner = np.argmin(d, axis=0)
    moved = []
    for k, b in enumerate(ends):
        mine = low[(owner == k) & (d[k] < SOLE_NEAR * h)]
        if len(mine) < 8:
            continue
        along = mine @ FORWARD
        tip = mine[along >= np.percentile(along, 97)].mean(0)
        if (tip - b.head) @ FORWARD < SOLE_AHEAD * h:
            continue
        b.tail = tip
        moved.append(b.name)
    return [f"feet along their soles to the toe tips: {', '.join(moved)}"] if moved else []


def _head(bones: Bones) -> Bone | None:
    return next((b for b in bones if b.name == "Head"), None)


def head_to_face(bones: Bones, shell: Shell) -> list[str]:
    """A head drawn ending in an ear (its end off the middle plane) runs instead from the
    skull (its joint and its siblings' joints, on the plane) to the front of the face. The
    bones beside it (ears) hang from it as HeadX<n>."""
    h = shell.height
    head = _head(bones)
    if head is None:
        return []
    if abs(shell.lateral(head.tail)) < FACE_OFF * h:
        return _head_to_snout(head, shell)
    by = {b.name: b for b in bones}
    sibs = [b for b in bones if b.parent == head.parent and b is not head and b.kind in HEAD_PARTS
            and np.linalg.norm(b.head - head.head) < FACE_SIBLING * h]
    start = np.mean([head.head] + [b.head for b in sibs], axis=0)
    start[0] = shell.mid
    tip = _face_front(shell, start)
    if tip is None or (tip - start) @ FORWARD < FACE_OFF * h:
        return []
    neck = by.get(head.parent)
    if neck is not None and np.linalg.norm(neck.tail - head.head) < 1e-6:
        neck.tail = start.copy()
    head.head, head.tail = start, tip
    count: dict[str, int] = {}
    for b in sibs:
        old, sfx = b.name, b.name[-2:] if b.side else ""
        count[sfx] = count.get(sfx, 0) + 1
        b.name, b.parent, b.kind = f"HeadX{count[sfx]}{sfx}", "Head", "head"
        for c in bones:
            if c.parent == old:
                c.parent = b.name
    bones[:] = in_order(bones)
    return [(f"the head ended off the middle (an ear?): now from the skull to the face; "
             f"{len(sibs)} bone(s) beside it hang from it")]


def _face_front(shell: Shell, start: np.ndarray) -> np.ndarray | None:
    """The front of the face: of the vertices near `start` and near the middle plane, the
    mean of the front 2%."""
    h, v = shell.height, shell.verts
    near = (np.linalg.norm(v - start, axis=1) < FACE_NEAR * h) & (np.abs(shell.lateral(v)) < FACE_OFF * h)
    if near.sum() < 8:
        return None
    along = v[near] @ FORWARD
    return v[near][along >= np.percentile(along, 98)].mean(0)


def _head_to_snout(head: Bone, shell: Shell) -> list[str]:
    """A head on the middle that stops short of the face (a dragon's drawn head ended 3% of
    the height long, at its brow, 6 cm behind the snout) reaches on to the face front, its
    start kept: the fit reads the head's turn from this bone."""
    h = shell.height
    start = head.head.copy()
    start[0] = shell.mid
    tip = _face_front(shell, start)
    if tip is None or (tip - head.tail) @ FORWARD < FACE_OFF * h:
        return []
    old = head.length
    head.tail = tip
    return [(f"the head stopped short of the face: now reaches its front, {100 * old / h:.0f}% -> "
             f"{100 * head.length / h:.0f}% of the height")]


def open_mouth_jaw(bones: Bones, shell: Shell) -> list[str]:
    """A mouth modelled open gets a jaw, from the mouth's back corner to the chin.

    On a side slice of the body at the middle plane (3 cells thick), around the head bone:
    in each column, empty cells between the lowest and highest solid cell are a gap; a gap
    open to the front and long enough is the mouth. The chin is the skin under the gap,
    followed forward while it stays thin (thicker is a chest). Small bones near the plane
    ending below the hinge were the drawn stub of that jaw: they go."""
    h, size = shell.height, shell.size
    head = _head(bones)
    if head is None or any("jaw" in b.name.lower() for b in bones):
        return []
    length = head.length
    if length < 4 * size:
        return []
    k = int(np.clip(np.floor((shell.mid - shell.lo[0]) / size), 1, shell.solid.shape[0] - 2))
    side = shell.solid[k - 1:k + 2].any(axis=0)[::-1]           # (forward, up): forward is -Y, so Y reversed
    nu, nz = side.shape
    fwd_of = lambda u: -(shell.lo[1] + (nu - u - 0.5) * size)     # forward coordinate of a column
    z_of = lambda z: shell.lo[2] + (z + 0.5) * size
    a = np.array([head.head @ FORWARD, head.head[2]])
    b = np.array([head.tail @ FORWARD, head.tail[2]])
    cols = np.array([fwd_of(u) for u in range(nu)])
    rows = np.array([z_of(z) for z in range(nz)])
    grid = np.stack(np.meshgrid(cols, rows, indexing="ij"), -1)
    ab = b - a
    t = np.clip(((grid - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    region = (np.linalg.norm(grid - (a + t[..., None] * ab), axis=-1) < 0.8 * length) & \
        (grid[..., 0] > a[0] - 0.25 * length)
    gap = np.zeros_like(side)
    for u in np.nonzero((side & region).any(1))[0]:
        zs = np.nonzero(side[u] & region[u])[0]
        if len(zs) >= 2:
            gap[u, zs.min():zs.max()] = ~side[u, zs.min():zs.max()]
    label, n = ndimage.label(gap)
    mouth = np.zeros_like(side)
    for c in range(1, n + 1):
        us, zs = np.nonzero(label == c)
        front = us.max()
        open_front = front + 1 >= nu or not side[front + 1, zs[us == front]].all()
        if len(us) >= 6 and open_front and (front - us.min() + 1) * size >= 0.25 * length:
            mouth |= label == c
    if not mouth.any():
        return []
    us, zs = np.nonzero(mouth)
    back, front = us.min(), us.max()
    hinge_z = z_of(zs[us == back].mean())

    def run_under(u: int, top: int):
        z1 = top
        while z1 >= 0 and not side[u, z1]:
            z1 -= 1
        if z1 < 0:
            return None
        z0 = z1
        while z0 > 0 and side[u, z0 - 1]:
            z0 -= 1
        return z0, z1

    run = run_under(front, zs[us == front].min())
    if run is None or (run[1] - run[0] + 1) * size > 0.7 * length:
        return []
    u = front
    while u + 1 < nu:
        lo_z = max(run[0] - 1, 0)
        hit = np.nonzero(side[u + 1, lo_z:run[1] + 2])[0]
        if not len(hit):
            break
        z = lo_z + int(hit[len(hit) // 2])
        z0 = z1 = z
        while z0 > 0 and side[u + 1, z0 - 1]:
            z0 -= 1
        while z1 + 1 < nz and side[u + 1, z1 + 1]:
            z1 += 1
        if (z1 - z0 + 1) * size > 0.7 * length:
            break
        run, u = (z0, z1), u + 1
    hinge = np.array([shell.mid, -fwd_of(back), hinge_z])
    chin = np.array([shell.mid, -(fwd_of(u) + 0.5 * size), z_of(0.5 * (run[0] + run[1]))])
    if np.linalg.norm(chin - hinge) < 0.3 * length or np.linalg.norm(chin - head.tail) > 0.7 * length:
        return []
    kids = children_of(bones)
    # the drawn stub hangs off the skull joint: the head's own, or its parent's when a snout was drawn
    near_head = kids.get("Head", []) + [c for c in kids.get(head.parent, []) if c is not head]
    stubs = [c for c in near_head if c.kind in HEAD_PARTS and not kids.get(c.name)
             and abs(shell.lateral(c.head)) < 0.04 * h and abs(shell.lateral(c.tail)) < 0.04 * h
             and c.tail[2] < hinge_z]
    bones[:] = [c for c in bones if not any(c is s for s in stubs)]
    at = next(i for i, c in enumerate(bones) if c is head) + 1
    bones.insert(at, Bone("Jaw", "Head", hinge, chin, "jaw"))
    note = f"an open mouth: a jaw from its corner to the chin, {100 * np.linalg.norm(chin - hinge) / length:.0f}% " \
           "of the head's length"
    return [note + (f"; {len(stubs)} drawn stub(s) in it removed" if stubs else "")]


def lone_head_bones(bones: Bones, shell: Shell) -> list[str]:
    """A short bone on the head, off the middle, with no twin on the other side goes: one
    brow ridge would take half the face from the head and give the fit turns to guess with
    nothing in a video to read them from. Ears and horns come in pairs and stay."""
    h = shell.height
    kids = children_of(bones)
    sibs = [b for b in kids.get("Head", []) if b.kind in HEAD_PARTS and not kids.get(b.name)]
    lone = []
    for b in sibs:
        off = shell.lateral(b.tail)
        if abs(off) < HEAD_OFF * h or b.length > HEAD_SHORT * h:
            continue
        image = shell.mirror(b.tail)
        if not any(o is not b and np.linalg.norm(o.tail - image) < HEAD_TWIN * h for o in sibs):
            lone.append(b)
    if not lone:
        return []
    bones[:] = [b for b in bones if not any(b is s for s in lone)]
    return [f"{len(lone)} short bone(s) on one side of the head only: removed"]


def repair_bones(bones: Bones, shell: Shell, feet: bool = True) -> list[str]:
    """The bone-level repairs in order: head, jaw, one-sided head bones, soles, twins,
    middle plane."""
    notes = head_to_face(bones, shell)
    notes += open_mouth_jaw(bones, shell)
    notes += lone_head_bones(bones, shell)
    if feet:
        notes += soles(bones, shell)
    notes += exact_twins(bones, shell)
    moved = middle_plane(bones, shell)
    if moved:
        notes.append(f"{moved} middle bone(s) put on the middle plane")
    return notes
