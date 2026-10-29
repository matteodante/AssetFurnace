"""Tests for `scripts/twocam_mocap.py` (the command line) and `image_to_3dlab/twocam/records.py`.

Everything here runs without a video, a reader or Blender: the clips are made-up motions
projected into both views, and Blender is replaced by a stand-in that answers the way
`scripts/blender_twocam_apply.py` does.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import twocam_mocap as cli
from twocam_toy import toy_rig, write_rig

from image_to_3dlab.twocam import exam, records
from image_to_3dlab.twocam.layout import Layout


def _det(tmp_path, info, names=None, noise=1.0, frames=24):
    truth = exam.make_motion(info, seed=0, frames=frames)
    joints = exam.motion_joints(info, truth)
    d = exam.synthetic_detections(joints, Layout.from_rig_info(info), noise, seed=1)
    path = tmp_path / "det.npz"
    np.savez(path, **d, fps=24.0, names=np.array(names or [k["name"] for k in info["keypoints"]]),
             calib=np.eye(2, 3))
    return path, joints


def _run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, (json.loads(out.out) if code == 0 else None), out.err


def test_exam_scores_a_rig_in_centimetres(tmp_path, capsys):
    rig = write_rig(tmp_path)
    code, out, _ = _run(capsys, "exam", rig, "--frames", 24, "--iters", 300, "--quiet")
    assert code == 0
    assert out["score"]["mean_cm"] < 2.0
    assert {"leg_front_cm", "leg_hind_cm", "body_cm"} <= set(out["score"])


def test_exam_shows_fix_swaps_putting_a_mixed_up_leg_back(tmp_path, capsys):
    rig = write_rig(tmp_path)
    code, out, _ = _run(capsys, "exam", rig, "--frames", 30, "--iters", 200, "--noise", 0,
                        "--swap-frames", 8, 16, "--fix-swaps", "--quiet")
    assert code == 0
    assert out["swaps"]["leg_front"] == 8


def test_check_then_fit_writes_a_motion_for_this_rig(tmp_path, capsys):
    info = toy_rig()
    rig = write_rig(tmp_path, info)
    det, joints = _det(tmp_path, info)
    code, report, _ = _run(capsys, "check", det, rig)
    assert code == 0 and report["verdict"]["level"] == "good"
    code, _, _ = _run(capsys, "fit", det, rig, tmp_path / "motion.json", "--iters", 300, "--quiet")
    assert code == 0
    motion = json.loads((tmp_path / "motion.json").read_text(encoding="utf-8"))
    assert motion["bones"] == [b["name"] for b in info["bones"]]
    assert exam.score(info, joints, np.array(motion["joints"]))["mean_cm"] < 2.0


def test_fit_carries_the_reading_s_provenance_into_the_motion(tmp_path, capsys):
    info = toy_rig()
    rig = write_rig(tmp_path, info)
    det, _ = _det(tmp_path, info)
    records.write_sidecar(det, {"schema_version": 1, "video": {"made_with": "test"}})
    code, _, _ = _run(capsys, "fit", det, rig, tmp_path / "motion.json", "--iters", 50, "--quiet")
    assert code == 0
    motion = json.loads((tmp_path / "motion.json").read_text(encoding="utf-8"))
    assert motion["provenance"]["video"]["made_with"] == "test"


def test_detections_read_with_another_rig_are_refused(tmp_path, capsys):
    info = toy_rig()
    rig = write_rig(tmp_path, info)
    names = [k["name"] for k in info["keypoints"]]
    det, _ = _det(tmp_path, info, names=names[1:] + names[:1])
    code, _, err = _run(capsys, "fit", det, rig, tmp_path / "m.json", "--iters", 10)
    assert code == 1
    assert "another rig" in err
    assert not (tmp_path / "m.json").exists()


def test_a_clip_whose_views_disagree_is_not_fitted_unless_forced(tmp_path, capsys):
    info = toy_rig()
    rig = write_rig(tmp_path, info)
    det, _ = _det(tmp_path, info)
    with np.load(det) as raw:
        d = dict(raw)
    d["front_uv"][..., 1] += np.random.default_rng(0).normal(0, 40, d["front_uv"][..., 1].shape)
    np.savez(det, **d)
    code, _, err = _run(capsys, "fit", det, rig, tmp_path / "m.json", "--iters", 10)
    assert code == 1 and "not worth fitting" in err
    code, out, _ = _run(capsys, "fit", det, rig, tmp_path / "m.json", "--iters", 10, "--force", "--quiet")
    assert code == 0 and out["verdict"]["level"] == "bad"


def test_read_needs_a_reference_frame_or_says_it_is_already_a_layout_frame(tmp_path, capsys):
    rig = write_rig(tmp_path)
    code, _, err = _run(capsys, "read", tmp_path / "v.mp4", tmp_path / "r.pt", rig, tmp_path / "d.npz")
    assert code == 1 and "--ref" in err


def test_the_blender_command_is_headless_and_safe():
    cmd = cli.apply_command(Path("blender"), Path("rig.blend"), Path("rig_info.json"), Path("m.json"),
                            Path("out"), armature="rig", name="roar")
    assert cmd[:4] == ["blender", "--background", "--factory-startup", "--disable-autoexec"]
    assert cmd[cmd.index("--python") + 1].endswith("blender_twocam_apply.py")
    assert cmd[cmd.index("--") + 1:] == ["rig_info.json", "m.json", "out", "--name", "roar", "--armature", "rig"]


def _motion_for(info, tmp_path):
    motion = exam.make_motion(info, frames=4)
    motion["provenance"] = {"video": {"made_with": "MiniMax H3"}}
    path = tmp_path / "motion.json"
    path.write_text(json.dumps(motion), encoding="utf-8")
    return path


def test_apply_runs_blender_and_writes_the_glbs_licence_record(tmp_path, capsys, monkeypatch):
    info = toy_rig()
    rig = write_rig(tmp_path, info)
    motion = _motion_for(info, tmp_path)
    blend = tmp_path / "rigged.blend"
    blend.write_bytes(b"BLENDER")
    blend.with_suffix(".provenance.json").write_text(json.dumps({"license": "CC-BY-4.0"}), encoding="utf-8")
    fake_blender = tmp_path / "blender.exe"
    fake_blender.write_bytes(b"")
    out = tmp_path / "out"
    seen = {}

    def fake_run(cmd, **_kwargs):
        seen["cmd"] = cmd
        glb = out / "animation.glb"
        glb.write_bytes(b"glTF")
        report = {"glb": str(glb), "frames": 4, "max_joint_error_mm": 0.05}
        return subprocess.CompletedProcess(cmd, 0, "noise\nI2L_TWOCAM_APPLY::" + json.dumps(report), "")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    code, report, _ = _run(capsys, "apply", blend, rig, motion, out, "--blender", fake_blender)
    assert code == 0 and "warning" not in report
    record = json.loads((out / "animation.provenance.json").read_text(encoding="utf-8"))
    assert record["license"] == "CC-BY-4.0"
    assert record["motion"]["capture"]["video"]["made_with"] == "MiniMax H3"
    assert record["motion"]["method"]["code_license"] == "Apache-2.0"


def test_apply_warns_when_blenders_joints_do_not_match_the_solvers(tmp_path, capsys, monkeypatch):
    info = toy_rig()
    out = tmp_path / "out"
    blend = tmp_path / "rigged.blend"
    blend.write_bytes(b"BLENDER")
    (tmp_path / "blender.exe").write_bytes(b"")

    def fake_run(cmd, **_kwargs):
        (out / "animation.glb").write_bytes(b"glTF")
        report = {"glb": str(out / "animation.glb"), "max_joint_error_mm": 40.0}
        return subprocess.CompletedProcess(cmd, 0, "I2L_TWOCAM_APPLY::" + json.dumps(report), "")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    code, report, _ = _run(capsys, "apply", blend, write_rig(tmp_path, info), _motion_for(info, tmp_path), out,
                           "--blender", tmp_path / "blender.exe")
    assert code == 0 and "not the one rig_info.json describes" in report["warning"]


def test_apply_reports_blenders_own_words_when_it_fails(tmp_path, capsys, monkeypatch):
    info = toy_rig()
    blend = tmp_path / "rigged.blend"
    blend.write_bytes(b"BLENDER")
    (tmp_path / "blender.exe").write_bytes(b"")
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **_k: subprocess.CompletedProcess(
        cmd, 1, "", "SystemExit: 2 armatures in this file: name one with --armature"))
    code, _, err = _run(capsys, "apply", blend, write_rig(tmp_path, info), _motion_for(info, tmp_path),
                        tmp_path / "out", "--blender", tmp_path / "blender.exe")
    assert code == 1 and "name one with --armature" in err


def test_apply_refuses_a_motion_fitted_to_another_rig(tmp_path, capsys):
    info = toy_rig()
    other = toy_rig()
    other["bones"][1]["name"] = "neck_renamed"
    motion = _motion_for(other, tmp_path)
    blend = tmp_path / "rigged.blend"
    blend.write_bytes(b"BLENDER")
    code, _, err = _run(capsys, "apply", blend, write_rig(tmp_path, info), motion, tmp_path / "out")
    assert code == 1 and "another rig" in err


def test_the_capture_record_names_every_input_and_its_terms(tmp_path):
    files = {}
    for name in ("clip.mp4", "reader.pt", "rig_info.json"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())
    record = records.capture_record(files["clip.mp4"], files["reader.pt"], files["rig_info.json"])
    assert record["video"]["license"].startswith("not declared")
    assert record["reader"]["backbone_license"] == "Apache-2.0"
    assert len(record["rig_info"]["sha256"]) == 64


@pytest.mark.parametrize("sidecar", ["model.provenance.json", "model.json"])
def test_the_models_licence_is_found_beside_it(tmp_path, sidecar):
    (tmp_path / sidecar).write_text(json.dumps({"license": "MIT"}), encoding="utf-8")
    assert records.source_record_for(tmp_path / "model.blend")["license"] == "MIT"
