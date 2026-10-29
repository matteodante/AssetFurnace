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
    assert code == 0 and "warnings" not in report
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
    assert code == 0
    assert any("not the one rig_info.json describes" in w for w in report["warnings"])
    assert any("no licence record" in w for w in report["warnings"])
    record = json.loads((out / "animation.provenance.json").read_text(encoding="utf-8"))
    assert record["license"].startswith("not recorded")


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


def test_read_refuses_an_output_numpy_would_rename(tmp_path, capsys):
    rig = write_rig(tmp_path)
    code, _, err = _run(capsys, "read", tmp_path / "v.mp4", tmp_path / "r.pt", rig, tmp_path / "det.out",
                        "--no-calib")
    assert code == 1 and ".npz" in err


def test_exam_swaps_on_a_rig_without_limbs_say_why_in_one_line(tmp_path, capsys):
    info = toy_rig()
    for b in info["bones"]:
        b["name"] = b["name"].replace("_L", "_l").replace("_R", "_r")
    code, _, err = _run(capsys, "exam", write_rig(tmp_path, info), "--swap-frames", 1, 3, "--iters", 5)
    assert code == 1 and "left/right limbs" in err


def test_check_judges_the_clip_as_it_would_be_fitted(tmp_path, capsys):
    info = toy_rig()
    rig = write_rig(tmp_path, info)
    det, _ = _det(tmp_path, info, frames=60)
    code, report, _ = _run(capsys, "check", det, rig)
    assert code == 0
    assert {"verdict", "verdict_as_generated", "feet_agreement", "sync"} <= set(report)


# --- prepare, dataset, train, film ----------------------------------------------------------


def test_the_prepare_command_runs_blender_headless_with_the_creatures_size():
    cmd = cli.prepare_command(Path("blender"), Path("wolf_rigged.glb"), Path("wolf"), 1.6, "y", 60.0, 32, False)
    assert cmd[:4] == ["blender", "--background", "--factory-startup", "--disable-autoexec"]
    assert cmd[cmd.index("--python") + 1].endswith("blender_twocam_prepare.py")
    after = cmd[cmd.index("--") + 1:]
    assert after[:2] == ["wolf_rigged.glb", "wolf"]
    assert after[after.index("--length-m") + 1] == "1.6" and after[after.index("--yaw-deg") + 1] == "60.0"
    assert "--no-render" in after


def test_the_dataset_is_split_across_blender_processes_without_gaps():
    cmds = cli.dataset_commands(Path("blender"), Path("rig.blend"), Path("rig_info.json"), Path("ds"),
                                count=1001, procs=3, seed=500, scale=0.6)
    spans = [(int(c[-4]), int(c[-3])) for c in cmds]
    assert spans == [(0, 334), (334, 334), (668, 333)]
    assert all(c[4] == "rig.blend" for c in cmds)


@pytest.mark.parametrize("labels, sentence", [
    (["front-left leg", "front-right leg", "hind-left leg", "hind-right leg"], "Exactly four legs."),
    (["left arm", "right arm", "left leg", "right leg"], "Exactly two arms, two legs."),
    ([], ""),
])
def test_the_count_sentence_comes_from_the_limbs(labels, sentence):
    assert cli.count_sentence({str(i): {"label": label} for i, label in enumerate(labels)}) == sentence


def test_film_needs_a_prepared_creature(tmp_path, capsys):
    code, _, err = _run(capsys, "film", tmp_path, "it howls", tmp_path / "clips")
    assert code == 1 and "creature.json" in err


def test_film_never_downloads_the_video_model_unasked(tmp_path, capsys, monkeypatch):
    from image_to_3dlab.twocam import video

    (tmp_path / "creature.json").write_text(json.dumps({"noun": "wolf", "colour_code": "x"}), encoding="utf-8")
    (tmp_path / "ref.png").write_bytes(b"png")
    monkeypatch.setattr(video, "hub_root", lambda env=None: tmp_path / "empty-cache")
    monkeypatch.setattr(video, "download", lambda **_k: (_ for _ in ()).throw(AssertionError("downloaded")))
    code, _, err = _run(capsys, "film", tmp_path, "it howls", tmp_path / "clips")
    assert code == 1 and "--download" in err and "GB" in err and "Community License" in err
    code, _, err = _run(capsys, "film", tmp_path, "it howls", tmp_path / "clips", "--download")
    assert code == 1 and "--yes" in err


def test_train_asks_before_fetching_dinov2(tmp_path, capsys, monkeypatch):
    from image_to_3dlab.twocam import train

    info = toy_rig()
    rig = write_rig(tmp_path, info)
    monkeypatch.setattr(train, "check_dataset", lambda *_a: 2200)
    monkeypatch.setattr(train, "backbone_cached", lambda: False)
    monkeypatch.setattr(train, "train", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("trained")))
    monkeypatch.setattr(cli.sys, "stdin", None)
    code, _, err = _run(capsys, "train", tmp_path, rig, tmp_path / "reader.pt")
    assert code == 1 and "346 MB" in err and "--yes" in err


def test_a_failed_take_keeps_the_others(tmp_path, capsys, monkeypatch):
    from image_to_3dlab.twocam import video

    (tmp_path / "creature.json").write_text(json.dumps({"noun": "wolf", "colour_code": "x"}), encoding="utf-8")
    (tmp_path / "ref.png").write_bytes(b"png")
    monkeypatch.setattr(video, "weights_plan", lambda root=None: {
        "ready": True, "engine": "H3", "license": "L", "license_url": "u", "license_note": "n", "files": {}})
    filmed = []

    def fake_film(binary, prompt, start, out, **settings):
        filmed.append(settings["seed"])
        if settings["seed"] == 2:
            raise video.VideoError("sd-cli failed (exit 1): out of memory")
        Path(out).write_bytes(b"webm")
        return Path(out)

    monkeypatch.setattr(video, "film", fake_film)
    code, out, _ = _run(capsys, "film", tmp_path, "it howls", tmp_path / "clips", "--takes", 3, "--quiet")
    assert code == 0 and filmed == [1, 2, 3]
    assert [("error" in t) for t in out["takes"]] == [False, True, False]
    assert (tmp_path / "clips" / "take3_s3.provenance.json").is_file()
