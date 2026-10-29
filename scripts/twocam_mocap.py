#!/usr/bin/env python3
"""Animate a rigged creature from a two-camera AI video: read the joints, fit the rig, export a GLB.

The method is @CanItCode's; the code is ported from Patch's AnyMove (Apache-2.0). A video
model films the creature front-on (left) and side-on (right) in one frame; a joint reader
trained on that creature finds every joint in both halves; the rig is fitted to them.

    python scripts/twocam_mocap.py read VIDEO.mp4 READER.pt RIG_INFO.json DET.npz --ref REF.png
    python scripts/twocam_mocap.py check DET.npz RIG_INFO.json
    python scripts/twocam_mocap.py fit DET.npz RIG_INFO.json MOTION.json
    python scripts/twocam_mocap.py apply RIGGED.blend RIG_INFO.json MOTION.json OUT_DIR
    python scripts/twocam_mocap.py capture VIDEO.mp4 READER.pt RIG_INFO.json RIGGED.blend OUT_DIR --ref REF.png
    python scripts/twocam_mocap.py exam RIG_INFO.json

`capture` is read + fit + apply. `exam` fits a made-up motion with known joints and
scores it in centimetres: no video, Blender or weights needed, so it is the quick check
that a rig can be fitted at all. A trained reader file holds its whole network, so
reading downloads nothing. The rig files (`rigged.blend`, `rig_info.json`) and readers
come from AnyMove's rig builder and trainer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from image_to_3dlab.twocam import exam as exam_mod
from image_to_3dlab.twocam import solve
from image_to_3dlab.twocam.layout import Layout
from image_to_3dlab.twocam.motion import load_motion, write_motion

APPLY_SCRIPT = REPO / "scripts" / "blender_twocam_apply.py"
DET_KEYS = (*solve.DETECTION_KEYS, "fps", "names", "calib")


class CaptureError(RuntimeError):
    pass


def load_rig_info(path: Path) -> dict:
    try:
        info = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read rig info {path}: {exc}") from exc
    solve.validate_rig_info(info)
    Layout.from_rig_info(info)
    return info


def load_detections(path: Path, info: dict) -> tuple[dict, float]:
    """A det.npz and its frame rate, refusing one read with another rig's joints."""
    with np.load(path, allow_pickle=False) as raw:
        missing = [k for k in (*solve.DETECTION_KEYS, "fps") if k not in raw.files]
        if missing:
            raise CaptureError(f"{path} is missing {', '.join(missing)}")
        d = {k: raw[k].astype(np.float64) for k in solve.DETECTION_KEYS}
        fps = float(raw["fps"])
        names = [str(n) for n in raw["names"]] if "names" in raw.files else None
    rig_names = [k["name"] for k in info["keypoints"]]
    if names is not None and names != rig_names:
        raise CaptureError(f"{path} was read with another rig's joints (or another order): every "
                           "joint would land on the wrong bone")
    solve.validate_detections(d, len(rig_names))
    return d, fps


def save_detections(path: Path, det: dict) -> None:
    np.savez(path, **{k: det[k] for k in DET_KEYS})


def detections_record(det_path: Path) -> dict | None:
    from image_to_3dlab.twocam.records import sidecar_path

    try:
        return json.loads(sidecar_path(det_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --- commands --------------------------------------------------------------------------


def cmd_read(args) -> dict:
    from image_to_3dlab.twocam import reader, records

    info = load_rig_info(args.rig_info)
    if args.ref is None and not args.no_calib:
        raise CaptureError("pass --ref REF.png (the frame the clip started from), or --no-calib for a "
                           "clip that is already a layout frame")
    device = solve.pick_device(args.device)
    det = reader.read_video(args.video, args.reader, None if args.no_calib else args.ref,
                            Layout.from_rig_info(info), [k["name"] for k in info["keypoints"]],
                            device=device, fps=args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_detections(args.out, det)
    records.write_sidecar(args.out, records.capture_record(
        args.video, args.reader, args.rig_info, args.made_with, args.video_license))
    if args.overlay:
        reader.overlay(det["frames"], det, Layout.from_rig_info(info), args.overlay, det["fps"])
    summary = {"detections": str(args.out), "frames": int(len(det["front_uv"])), "fps": det["fps"],
               "mean_confidence": {"front": float(det["front_conf"].mean()), "side": float(det["side_conf"].mean())},
               "calibration_converged": bool(det["calib_converged"])}
    if not det["calib_converged"]:
        summary["warning"] = "the first frame did not align to the reference: a plain stretch was used"
    return summary


def clip_report(d: dict, info: dict, feet=None) -> dict:
    layout = Layout.from_rig_info(info)
    names = [k["name"] for k in info["keypoints"]]
    feet = feet or solve.ground_joints(info)
    sync = solve.sync_report(d)
    paws = solve.paw_agreement(d, names, feet)
    sync.pop("per_frame_gap_px")
    return {"sync": sync, "feet_agreement": paws,
            "verdict": solve.verdict(sync, paws, solve.creature_height_px(info, layout))}


def cmd_check(args) -> dict:
    info = load_rig_info(args.rig_info)
    d, _ = load_detections(args.det, info)
    return clip_report(d, info, args.feet)


def cmd_fit(args) -> dict:
    info = load_rig_info(args.rig_info)
    d, fps = load_detections(args.det, info)
    report = clip_report(d, info, args.feet)
    if report["verdict"]["level"] == "bad" and not args.force:
        raise CaptureError("this clip is not worth fitting: " + "; ".join(report["verdict"]["reasons"])
                           + ". Generate it again, or pass --force")
    log = None if args.quiet else (lambda msg: print(msg, file=sys.stderr, flush=True))
    motion = solve.capture(d, info, fps, iters=args.iters, device=args.device, align=not args.no_align,
                           swaps=args.fix_swaps, feet=args.feet, log=log)
    motion["provenance"] = detections_record(args.det)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_motion(motion, args.out)
    return {"motion": str(args.out), "frames": len(motion["euler"]), "reproj_px": motion["reproj_px"],
            "verdict": motion["sync"]["verdict"], "swaps": motion["swaps"]}


def apply_command(blender: Path, blend: Path, rig_info: Path, motion: Path, out: Path,
                  armature: str | None = None, name: str = "animation", keep_paint: bool = False) -> list[str]:
    cmd = [str(blender), "--background", "--factory-startup", "--disable-autoexec", str(blend),
           "--python-exit-code", "1", "--python", str(APPLY_SCRIPT), "--",
           str(rig_info), str(motion), str(out), "--name", name]
    if armature:
        cmd += ["--armature", armature]
    if keep_paint:
        cmd.append("--keep-paint")
    return cmd


def find_blender_or_fail(explicit: Path | None) -> Path:
    from image_to_3dlab.blender import find_blender, missing_help

    path = explicit or find_blender()
    if path is None or not Path(path).is_file():
        raise CaptureError(missing_help())
    return Path(path)


def cmd_apply(args) -> dict:
    from image_to_3dlab.twocam import records

    info = load_rig_info(args.rig_info)
    motion = load_motion(args.motion)
    rig_bones = [b["name"] for b in info["bones"]]
    if motion["bones"] != rig_bones:
        raise CaptureError(f"{args.motion} was fitted to another rig: its bones are not {args.rig_info}'s")
    if not args.blend.is_file():
        raise CaptureError(f"no rigged scene at {args.blend}")
    blender = find_blender_or_fail(args.blender)
    args.out.mkdir(parents=True, exist_ok=True)
    cmd = apply_command(blender, args.blend, args.rig_info, args.motion, args.out,
                        args.armature, args.name, args.keep_paint)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            check=False)
    report_line = next((line for line in result.stdout.splitlines()
                        if line.startswith("I2L_TWOCAM_APPLY::")), None)
    if result.returncode != 0 or report_line is None:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        raise CaptureError(f"Blender failed (exit {result.returncode}):\n{tail}")
    report = json.loads(report_line.split("::", 1)[1])
    glb = Path(report["glb"])
    source = records.source_record_for(args.source_model or args.blend)
    records.write_sidecar(glb, records.animation_record(args.blend, motion.get("provenance"), source))
    if report.get("max_joint_error_mm") is not None and report["max_joint_error_mm"] > 1.0:
        report["warning"] = ("Blender's joints land more than 1 mm from the solver's: the rig on disk "
                             "is not the one rig_info.json describes")
    return report


def cmd_capture(args) -> dict:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    det = out / "det.npz"
    motion = out / "motion.json"
    steps = {"read": cmd_read(argparse.Namespace(**{**vars(args), "out": det}))}
    steps["fit"] = cmd_fit(argparse.Namespace(**{**vars(args), "det": det, "out": motion, "device": "cpu"}))
    steps["apply"] = cmd_apply(argparse.Namespace(**{**vars(args), "motion": motion, "out": out}))
    return steps


def cmd_exam(args) -> dict:
    info = load_rig_info(args.rig_info)
    layout = Layout.from_rig_info(info)
    truth = exam_mod.make_motion(info, seed=args.seed, frames=args.frames)
    joints = exam_mod.motion_joints(info, truth)
    d = exam_mod.synthetic_detections(joints, layout, noise_px=args.noise, seed=args.seed + 1)
    if args.swap_frames:
        pairs = next(iter(solve.limb_pairs(info).values()))
        L, R = (np.array(x) for x in zip(*pairs))
        a, b = args.swap_frames
        d["side_uv"][a:b, L], d["side_uv"][a:b, R] = d["side_uv"][a:b, R].copy(), d["side_uv"][a:b, L].copy()
    log = None if args.quiet else (lambda msg: print(msg, file=sys.stderr, flush=True))
    motion = solve.capture(d, info, truth["fps"], iters=args.iters, device=args.device,
                           swaps=args.fix_swaps, log=log)
    if args.out:
        write_motion(motion, args.out)
    return {"score": exam_mod.score(info, joints, np.array(motion["joints"])),
            "reproj_px": motion["reproj_px"], "swaps": motion["swaps"]}


# --- arguments -----------------------------------------------------------------------------


def _feet(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def reading(s):
        s.add_argument("--ref", type=Path, help="the reference frame the clip started from")
        s.add_argument("--no-calib", action="store_true", help="the clip is already a layout frame")
        s.add_argument("--overlay", type=Path, help="also write the clip with the read joints drawn on")
        s.add_argument("--fps", type=float, help="override the clip's frame rate")
        s.add_argument("--made-with", help="the video model that made the clip, for the provenance record")
        s.add_argument("--video-license", help="the terms the clip is under, for the provenance record")

    def fitting(s):
        s.add_argument("--iters", type=int, default=1500)
        s.add_argument("--fix-swaps", action="store_true",
                       help="swap near/far limbs the side view mixed up (unpainted rigs)")
        s.add_argument("--no-align", action="store_true", help="never re-time the front view")
        s.add_argument("--feet", type=_feet, help="comma-separated joints for the feet agreement score")
        s.add_argument("--force", action="store_true", help="fit even a clip the check calls bad")
        s.add_argument("--quiet", action="store_true")

    def applying(s):
        s.add_argument("--armature", help="the armature to animate (default: the only one)")
        s.add_argument("--name", default="animation", help="the GLB's file name, without .glb")
        s.add_argument("--keep-paint", action="store_true", help="keep the limb paint in the GLB")
        s.add_argument("--blender", type=Path, help="the Blender to run (default: found like Finish)")
        s.add_argument("--source-model", type=Path,
                       help="the model whose licence sidecar to carry over (default: the .blend's)")

    s = sub.add_parser("read", help="read every joint in both views of a clip")
    s.add_argument("video", type=Path)
    s.add_argument("reader", type=Path)
    s.add_argument("rig_info", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--device", default="auto", help="auto, cuda or cpu")
    reading(s)

    s = sub.add_parser("check", help="say whether a read clip's two views agree enough to fit")
    s.add_argument("det", type=Path)
    s.add_argument("rig_info", type=Path)
    s.add_argument("--feet", type=_feet)

    s = sub.add_parser("fit", help="fit the rig to a read clip")
    s.add_argument("det", type=Path)
    s.add_argument("rig_info", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--device", default="cpu", help="cpu (default: faster, same result anywhere), cuda or auto")
    fitting(s)

    s = sub.add_parser("apply", help="key a fitted motion onto the rig in Blender and export a GLB")
    s.add_argument("blend", type=Path)
    s.add_argument("rig_info", type=Path)
    s.add_argument("motion", type=Path)
    s.add_argument("out", type=Path)
    applying(s)

    s = sub.add_parser("capture", help="read + fit + apply")
    s.add_argument("video", type=Path)
    s.add_argument("reader", type=Path)
    s.add_argument("rig_info", type=Path)
    s.add_argument("blend", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--device", default="auto", help="for reading; the fit always runs on the CPU")
    reading(s)
    fitting(s)
    applying(s)

    s = sub.add_parser("exam", help="fit a made-up motion with known joints and score it in cm")
    s.add_argument("rig_info", type=Path)
    s.add_argument("--frames", type=int, default=48)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--noise", type=float, default=3.0, help="reader noise to simulate, in pixels")
    s.add_argument("--iters", type=int, default=800)
    s.add_argument("--device", default="cpu")
    s.add_argument("--fix-swaps", action="store_true")
    s.add_argument("--swap-frames", type=int, nargs=2, metavar=("FROM", "TO"),
                   help="mix up the first limb's side-view labels in these frames, to test --fix-swaps")
    s.add_argument("--out", type=Path, help="also write the solved motion")
    s.add_argument("--quiet", action="store_true")
    return p


COMMANDS = {"read": cmd_read, "check": cmd_check, "fit": cmd_fit, "apply": cmd_apply,
            "capture": cmd_capture, "exam": cmd_exam}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = COMMANDS[args.command](args)
    except (RuntimeError, ValueError, ArithmeticError, OSError) as exc:
        # CaptureError, ReaderError, the rig/motion/layout errors, a missing file: one line,
        # no traceback, exit 1
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
