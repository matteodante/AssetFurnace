"""Route B0's face labels: visible faces from the slot map, hidden ones from the body's bones
carried into the model's frame, then a majority clean-up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab.slot_parts import segment, slots
from image_to_3dlab.slot_parts.camera import Camera
from tests.test_slot_parts_cli import _body

CAM = Camera(fov=0.8, dist=2.8, size=400)


def test_project_matches_the_fit_unprojection():
    from image_to_3dlab.slot_parts import fit

    point = np.array([[0.1, 0.0, 0.3]])  # Blender: x, depth y, height z
    (u, v), = segment.project(CAM, point)
    assert fit.unproject(CAM, u, v, 0.0) == pytest.approx((0.1, 0.3))


def test_visible_faces_read_the_map_and_hidden_ones_stay_unlabelled():
    slot_map = np.zeros((400, 400), np.uint8)
    slot_map[:200] = slots.label("helmet")
    slot_map[200:] = slots.label("boots")
    centres = np.array([[0, 0, 0.3], [0, 0, -0.3], [0, 0, 0.3], [5.0, 0, 0]])
    labels = segment.visible_labels(centres, [True, True, False, True], CAM, slot_map)
    assert labels.tolist() == [slots.label("helmet"), slots.label("boots"), -1, -1]


def test_hidden_faces_take_the_nearest_bone():
    from image_to_3dlab.slot_parts.skeleton import Segment

    segs = [Segment("h", "head", "", np.array([0, 0, 1.0]), np.array([0, 0, 1.2])),
            Segment("f", "foot", "R", np.array([0, 0, 0.0]), np.array([0, 0.1, 0.0]))]
    labels = segment.hidden_labels(np.array([[0, 0.05, 1.1], [0.02, 0.0, 0.05]]), segs)
    assert [slots.LABELS[i] for i in labels] == ["helmet", "boots"]


def test_majority_cleans_a_lone_face():
    labels = np.array([2, 2, 5, 2, 2])
    neighbours = [np.array([1]), np.array([0, 2]), np.array([1, 3]), np.array([2, 4]), np.array([3])]
    assert segment.smooth(labels, neighbours).tolist() == [2, 2, 2, 2, 2]


def test_bones_carried_through_matching_cameras_land_where_they_were(tmp_path):
    path, cam, heads, names = _body(tmp_path)
    from image_to_3dlab.slot_parts import body

    record = body.load(path)
    carried = {s.bone: s for s in segment.bones_in_model(record, cam, record["image_size"], y_centre=0.0)}
    hips = carried[names["hips"]].head
    gltf = heads[names["hips"]]
    assert hips[0] == pytest.approx(gltf[0], abs=1e-6) and hips[2] == pytest.approx(gltf[1], abs=1e-6)
    assert hips[1] == pytest.approx(0.0, abs=1e-6)  # the root sits at the model's centre depth


def test_label_faces_keeps_only_the_chosen_slots(tmp_path):
    path, cam, _, _ = _body(tmp_path)
    from image_to_3dlab.slot_parts import body

    record = body.load(path)
    slot_map = np.full((400, 400), slots.label("torso"), np.uint8)
    centres = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])  # one seen, one hidden behind
    neighbours = [np.array([], int), np.array([], int)]
    labels = segment.label_faces(centres, [True, False], cam, slot_map, record, record["image_size"], 0.0,
                                 neighbours, chosen={"helmet"})
    assert set(labels.tolist()) == {slots.label("uncovered")}



def _load_segment_script():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "blender_segment_parts.py"
    spec = importlib.util.spec_from_file_location("blender_segment_parts", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # bpy is only imported inside main()
    return module


def test_the_segment_job_cuts_only_drawn_slots(tmp_path):
    import json

    bs = _load_segment_script()
    job = {"model": "m.glb", "camera": {}, "slot_map": "s.npy", "body": "b.json", "concept_size": [8, 8],
           "chosen": ["torso", "gloves"], "out": "o"}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert bs.load_job(path)["chosen"] == ["torso", "gloves"]
    path.write_text(json.dumps(dict(job, chosen=["weapon"])))
    with pytest.raises(ValueError, match="drawn slots"):
        bs.load_job(path)
    path.write_text(json.dumps({k: v for k, v in job.items() if k != "slot_map"}))
    with pytest.raises(ValueError, match="slot_map"):
        bs.load_job(path)
