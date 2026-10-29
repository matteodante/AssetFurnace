"""The bpy-free parts of `scripts/blender_creature_rig_build.py`: limbs, loose parts, the clean-up
of skin weights, the closed voxel copy bone heat falls back on, and the poses sheet. A real
Blender run is in `test_creature_rig_blender_run.py`."""

import blender_creature_rig_build as build
import numpy as np
import pytest


def bone(name, parent, kind, head=(0, 0, 0), tail=(0, 0, 1), side="", deform=True):
    return {"name": name, "parent": parent, "kind": kind, "head": list(head), "tail": list(tail),
            "side": side, "deform": deform}


def quadruped_bones():
    bones = [bone("Root", None, "root", deform=False), bone("Hips", "Root", "spine"),
             bone("Spine1", "Hips", "spine"), bone("Neck1", "Spine1", "neck"), bone("Head", "Neck1", "head"),
             bone("HeadX1", "Head", "part"), bone("Tail1", "Hips", "tail")]
    for s in "LR":
        bones += [bone(f"Thigh.{s}", "Hips", "hind", side=s), bone(f"Shin.{s}", f"Thigh.{s}", "hind", side=s),
                  bone(f"HindPaw.{s}", f"Shin.{s}", "hind", side=s),
                  bone(f"Scapula.{s}", "Spine1", "front", side=s),
                  bone(f"UpperArm.{s}", f"Scapula.{s}", "front", side=s),
                  bone(f"Forearm.{s}", f"UpperArm.{s}", "front", side=s)]
    return bones


def test_arguments():
    assert build.parse_args(["blender", "--", "W"]) == {"work": build.Path("W"), "sheet": True}
    assert build.parse_args(["x.py", "W", "--no-sheet"])["sheet"] is False
    with pytest.raises(SystemExit, match="usage"):
        build.parse_args(["blender", "--"])


def test_a_broken_joints_file_is_refused_before_blender_builds_anything():
    with pytest.raises(ValueError, match="no bones"):
        build.check_bones({"bones": []})
    with pytest.raises(ValueError, match="before its parent"):
        build.check_bones({"bones": [bone("Shin.L", "Thigh.L", "hind"), bone("Thigh.L", None, "hind")]})
    with pytest.raises(ValueError, match="two bones"):
        build.check_bones({"bones": [bone("Hips", None, "spine"), bone("Hips", None, "spine")]})
    assert len(build.check_bones({"bones": quadruped_bones()})) == len(quadruped_bones())


def test_each_limb_is_a_chain_of_its_own_and_the_trunk_is_none():
    bones = quadruped_bones()
    chains = build.limb_chains(bones)
    assert "Hips" not in chains and "Head" not in chains and "Root" not in chains
    assert chains["Thigh.L"] == chains["Shin.L"] == chains["HindPaw.L"]
    assert chains["Scapula.R"] == chains["Forearm.R"]
    limbs = {chains[n] for n in ("Thigh.L", "Thigh.R", "Scapula.L", "Scapula.R", "HeadX1")}
    assert len(limbs) == 5
    assert build.chain_ends(bones, chains) == {"HindPaw.L", "HindPaw.R", "Forearm.L", "Forearm.R", "HeadX1"}


def test_a_uv_seam_does_not_split_a_part():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],          # one triangle
                  [1, 0, 0], [0, 1, 0], [1, 1, 0],          # its neighbour across a seam: two vertices doubled
                  [5, 5, 5], [6, 5, 5], [5, 6, 5]], float)  # a separate piece
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [6, 7], [7, 8], [8, 6]])
    part = build.loose_parts(v, edges, 1e-5)
    assert part.tolist() == [0] * 6 + [1] * 3
    assert build.loose_parts(v, edges[:6], 1e-5).tolist()[6:] == [1, 2, 3]   # loose points, each alone


def test_heat_runs_on_the_big_parts():
    part = np.array([0] * 100 + [1] * 3 + [2])
    assert build.main_parts(part) == {0, 1}
    assert build.main_parts(np.array([0, 1, 1, 2, 3, 4]), share=0.5) == {1}


def test_nearest_points_and_bone_distances():
    idx, dist = build.nearest_brute(np.array([[0, 0, 0], [1, 0, 0]]), np.array([[0.9, 0, 0], [-1, 0, 0]]))
    assert idx.tolist() == [1, 0] and np.allclose(dist, [0.1, 1.0])
    d = build.segment_distances(np.array([[0, 0, 0.5], [0, 0, 2], [1, 0, 0]]), np.array([[0, 0, 0]]),
                                np.array([[0, 0, 1]]))
    assert np.allclose(d[:, 0], [0, 1, 1])


def test_weights_by_distance_take_the_two_nearest_bones_by_the_fourth_power():
    w = build.by_distance(np.array([[1.0, 2.0, 3.0]]), [4, 0, 2], 5)
    assert w[0, 2] == 0 and w.sum() == pytest.approx(1)
    assert w[0, 4] / w[0, 0] == pytest.approx(16)


def test_clean_up_keeps_four_bones_and_one_limb():
    chain_of = np.array([-1, 0, 0, 1, 0, -1])
    w = np.array([[0.30, 0.25, 0.20, 0.10, 0.10, 0.05],    # trunk strongest: limbs may share it
                  [0.10, 0.50, 0.10, 0.30, 0.00, 0.00],    # limb 0 strongest: limb 1 goes
                  [0.0] * 6])
    out = build.clean_weights(w, chain_of)
    assert ((out > 0).sum(1) <= 4).all()
    assert np.allclose(out[0], np.array([0.30, 0.25, 0.20, 0.10, 0, 0]) / 0.85)
    assert np.allclose(out[1], [1 / 7, 5 / 7, 1 / 7, 0, 0, 0])
    assert (out[2] == 0).all()


def scene():
    """A trunk and a head (trunk bones), two forelegs (limbs), skin along each bone, and three
    loose parts: a claw at the left paw, a sword held in the right, a horn on the head."""
    heads = np.array([[0, 0.2, 0.5], [0, -0.2, 0.5], [0.1, -0.1, 0.45], [0.1, -0.1, 0.25],
                      [-0.1, -0.1, 0.45], [-0.1, -0.1, 0.25]])
    tails = np.array([[0, -0.2, 0.5], [0, -0.4, 0.6], [0.1, -0.1, 0.25], [0.1, -0.1, 0.05],
                      [-0.1, -0.1, 0.25], [-0.1, -0.1, 0.05]])
    chain_of = np.array([-1, -1, 0, 0, 1, 1])
    ends = np.array([False, False, False, True, False, True])
    skin, owner_col = [], []
    for c in range(6):
        for t in np.linspace(0, 1, 12):
            skin.append(heads[c] + t * (tails[c] - heads[c]) + [0.02, 0, 0])
            owner_col.append(c)
    skin = np.array(skin)
    claw = np.array([[0.12, -0.12 - 0.01 * k, 0.05] for k in range(6)])
    sword = np.array([[-0.1, -0.1 - 0.02 * k, 0.04] for k in range(25)])
    horn = np.array([[0.02, -0.4, 0.6 + 0.01 * k] for k in range(10)])
    co = np.concatenate([skin, claw, sword, horn])
    part = np.array([0] * len(skin) + [1] * len(claw) + [2] * len(sword) + [3] * len(horn))
    weights = np.eye(6)[owner_col]
    near, dist = build.nearest_brute(skin, co)
    owner = chain_of[np.argmax(weights, 1)]
    return co, part, weights, near, dist, owner, heads, tails, chain_of, ends


def test_loose_parts_follow_their_limb_or_the_skin_they_sit_on():
    co, part, weights, near, dist, owner, heads, tails, chain_of, ends = scene()
    w, done = build.follow_limbs(co, part, {0}, weights, near, dist, owner, heads, tails, chain_of, ends, 1.0)
    assert done == {"limb": 1, "trunk": 1, "held": 1}
    skin = part == 0
    assert np.allclose(w[skin], weights[near[skin]])                 # the skin keeps its heat weights
    claw = w[part == 1]
    assert np.allclose(claw.sum(1), 1) and (claw[:, 3] > 0.9).all()  # the paw moves the claw ...
    assert (claw[:, [0, 1, 4, 5]] == 0).all()                        # ... and nothing else does
    sword = w[part == 2]
    assert (sword[:, 5] == 1).all() and (sword.sum(1) == 1).all()    # held rigidly by the right forearm
    horn = w[part == 3]
    assert np.allclose(horn.sum(1), 1) and (horn[:, 1] > 0.5).all()  # the head carries the horn
    assert horn[0, 1] == 1 and horn[-1, 0] > 0                       # blending to the trunk bones further out
    assert (horn[:, 2:] == 0).all()


def grid_box(n, skip=None):
    """A unit cube, each face a grid of n x n squares; `skip` = (face, i, j) leaves one square out."""
    verts, faces = [], []
    for f, (axis, at) in enumerate([(a, s) for a in range(3) for s in (0.0, 1.0)]):
        o1, o2 = (axis + 1) % 3, (axis + 2) % 3
        for i in range(n):
            for j in range(n):
                if skip == (f, i, j):
                    continue
                quad = []
                for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = [0.0, 0.0, 0.0]
                    p[axis], p[o1], p[o2] = at, (i + d1) / n, (j + d2) / n
                    quad.append(len(verts))
                    verts.append(p)
                faces += [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
    return np.array(verts), np.array(faces)


def test_the_voxel_copy_is_solid_inside_a_closed_skin():
    v, f = grid_box(4)
    solid, lo = build.sealed_solid(v, f, 0.1)
    centre = np.floor((np.array([0.5, 0.5, 0.5]) - lo) / 0.1).astype(int)
    assert solid[tuple(centre)] and not solid[0, 0, 0]
    assert solid.sum() * 0.001 == pytest.approx(1.0, rel=0.8)


def test_a_hole_smaller_than_a_voxel_is_sealed_and_a_big_one_is_not():
    small = grid_box(20, skip=(1, 10, 10))           # a 5 cm hole, voxels of 10 cm
    solid, lo = build.sealed_solid(*small, 0.1)
    assert solid[tuple(np.floor((np.array([0.5, 0.5, 0.5]) - lo) / 0.1).astype(int))]
    v, f = grid_box(1)
    open_box = f[[k for k in range(len(f)) if k not in (2, 3)]]   # one whole side missing
    solid, lo = build.sealed_solid(v, open_box, 0.1)
    assert not solid[tuple(np.floor((np.array([0.5, 0.5, 0.5]) - lo) / 0.1).astype(int))]


def test_an_inner_wall_does_not_hollow_the_copy():
    v, f = grid_box(4)
    wall = np.array([[0.5, 0.1, 0.1], [0.5, 0.9, 0.1], [0.5, 0.9, 0.9], [0.5, 0.1, 0.9]])
    v2 = np.concatenate([v, wall])
    f2 = np.concatenate([f, [[len(v), len(v) + 1, len(v) + 2], [len(v), len(v) + 2, len(v) + 3]]])
    solid, lo = build.sealed_solid(v2, f2, 0.1)
    for x in (0.25, 0.75):
        assert solid[tuple(np.floor((np.array([x, 0.5, 0.5]) - lo) / 0.1).astype(int))]


def signed_volume(verts, quads):
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    return float((a * np.cross(b, c)).sum() / 6)


def test_the_voxel_skin_is_closed_and_faces_out():
    one = np.zeros((3, 3, 3), bool)
    one[1, 1, 1] = True
    v, q = build.voxel_skin(one, np.array([10.0, 0, 0]), 0.5)
    assert len(v) == 8 and len(q) == 6
    assert signed_volume(v, q) == pytest.approx(0.125)
    assert np.allclose(v.min(0), [10.5, 0.5, 0.5])
    two = np.zeros((2, 1, 1), bool)
    two[:] = True
    v, q = build.voxel_skin(two, np.zeros(3), 1.0)
    assert len(v) == 12 and len(q) == 10 and signed_volume(v, q) == pytest.approx(2.0)


def test_the_sheet_poses():
    bones = quadruped_bones() + [bone("Coxa1.L", "Hips", "chain", side="L"),
                                 bone("Femur1.L", "Coxa1.L", "chain", side="L"),
                                 bone("Tibia1.R", "Femur1.L", "chain", side="R")]
    assert build.sheet_pose(bones, "rest") == {}
    bend = build.sheet_pose(bones, "bend")
    assert bend["Shin.L"] == (45.0, 0.0, 0.0) and bend["Shin.R"] == (-45.0, 0.0, 0.0)
    assert bend["Forearm.L"] == (45.0, 0.0, 0.0)
    assert bend["Thigh.L"] == bend["Thigh.R"] == bend["UpperArm.R"] == (-30.0, 0.0, 0.0)
    assert "Coxa1.L" not in bend and bend["Femur1.L"] == (30.0, 0.0, 0.0) and bend["Tibia1.R"] == (-30.0, 0.0, 0.0)
    assert "Hips" not in bend and "HindPaw.L" not in bend
    turn = build.sheet_pose(bones, "turn")
    assert set(turn) == {"Spine1", "Neck1", "Tail1"} and turn["Neck1"] == (12.0, 0.0, 10.0)
    both = build.sheet_pose(bones, "all")
    assert both["Shin.L"] == bend["Shin.L"] and both["Tail1"] == turn["Tail1"]
    with pytest.raises(ValueError):
        build.sheet_pose(bones, "dance")


def test_numbered_bones_pose_like_their_kind():
    assert build.base_name("Thigh2.L") == "Thigh" and build.base_name("Tail3") == "Tail"
    bend = build.sheet_pose([bone("Shin2.R", None, "hind", side="R")], "bend")
    assert bend["Shin2.R"] == (-45.0, 0.0, 0.0)


def test_tiles_are_laid_out_rows_first():
    rows = [[np.full((2, 2, 4), 10 * i + j, np.float32) for j in range(3)] for i in range(2)]
    sheet = build.tile_sheet(rows)
    assert sheet.shape == (4, 6, 4)
    assert sheet[0, 0, 0] == 0 and sheet[0, 5, 0] == 2 and sheet[3, 0, 0] == 10 and sheet[3, 5, 0] == 12


def test_the_weights_report_names_bones_that_move_nothing_on_a_mesh():
    w = np.array([[1, 0], [0.5, 0.5], [1, 0], [0, 1], [0, 1]], float)
    report = build.weights_report(["Head", "Jaw"], [("body", 3, 1), ("teeth", 2, 2)], w)
    assert report == {"body": {"vertices": 3, "loose_parts": 1, "empty_bones": []},
                      "teeth": {"vertices": 2, "loose_parts": 2, "empty_bones": ["Head"]}}
