"""A piece made on its own loses the small bits floating away from it: the first real set's
glove had a 56-face fragment a fifth of its size away, which hung in the air next to the hand
in Unreal. Bits that touch the piece, and large separate parts, stay."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from image_to_3dlab.slot_parts import fragments, pieces, slots

REPO = Path(__file__).resolve().parents[1]


def _glove():
    hand = trimesh.creation.box(extents=(0.2, 0.1, 0.8)).subdivide().subdivide().subdivide()   # the piece
    rivet = trimesh.creation.box(extents=(0.01, 0.01, 0.01))                # a bit on its surface
    rivet.apply_translation((0.105, 0, 0))
    floating = trimesh.creation.box(extents=(0.03, 0.03, 0.03))             # a stray bit in the air
    floating.apply_translation((0.4, 0, 0.3))
    strap = trimesh.creation.box(extents=(0.2, 0.1, 0.3)).subdivide().subdivide()   # a big separate part
    strap.apply_translation((0, 0.3, 0))
    return trimesh.util.concatenate([hand, rivet, floating, strap]), len(floating.faces)


def test_the_floating_bit_goes_and_the_rest_stays():
    mesh, stray = _glove()
    keep, report = fragments.floating(mesh.vertices, mesh.faces)
    assert int((~keep).sum()) == stray
    assert report == {"dropped": 1, "faces": stray}


def test_the_script_rewrites_the_piece(tmp_path):
    mesh, stray = _glove()
    path = tmp_path / "glove.glb"
    mesh.export(path)
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "drop_fragments.py"), str(path)],
                          capture_output=True, text=True, check=True)
    assert "dropped 1" in done.stdout
    assert len(trimesh.load(path, force="mesh").faces) == len(mesh.faces) - stray


def test_a_piece_made_on_its_own_is_cleaned_after_its_finish():
    commands = pieces.model_commands(Path("p.png"), Path("pieces"), "gloves", "python", REPO / "scripts")
    names = [Path(c[1]).name for c in commands]
    assert names.index("drop_fragments.py") > names.index("retopo_repaint.py")
    assert slots.CATALOGUE["gloves"]


def test_a_bit_on_the_other_side_of_a_pair_stays():
    """Gloves and boots come as one file holding both sides: a buckle on the second boot is
    measured against that boot, not against the first one only."""
    left = trimesh.creation.box(extents=(0.2, 0.3, 0.1)).subdivide().subdivide()
    right = left.copy()
    right.apply_translation((0.6, 0, 0))
    buckle = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    buckle.apply_translation((0.6, 0, 0.06))               # resting on the right boot's top
    mesh = trimesh.util.concatenate([left, right, buckle])
    keep, report = fragments.floating(mesh.vertices, mesh.faces)
    assert keep.all() and report["dropped"] == 0


def test_a_bit_resting_inside_a_large_face_is_not_far():
    """Measured to the surface, not to its corners: a decimated piece has big triangles."""
    # a disc: its top is a fan of long thin triangles, so mid-radius no corner is near
    plate = trimesh.creation.cylinder(radius=1.0, height=0.05, sections=200)
    stud = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    stud.apply_translation((0.5, 0.0, 0.03))               # on the disc's top, half way out
    mesh = trimesh.util.concatenate([plate, stud])
    keep, _ = fragments.floating(mesh.vertices, mesh.faces)
    assert keep.all()
