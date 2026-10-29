#!/usr/bin/env python3
"""Animate a rigged creature from a two-camera AI video: read the joints, fit the rig, export a GLB.

The method is @CanItCode's; the code is ported from Patch's AnyMove (Apache-2.0). A video
model films the creature front-on (left) and side-on (right) in one frame; a joint reader
trained on that creature finds every joint in both halves; the rig is fitted to them.

Once per creature (any rigged model: the Animate tab's SkinTokens rig, or one by hand):

    python scripts/twocam_mocap.py prepare RIGGED.glb CREATURE/ --length-m 1.6 --noun "grey wolf"
    python scripts/twocam_mocap.py dataset CREATURE/rigged.blend CREATURE/rig_info.json CREATURE/dataset
    python scripts/twocam_mocap.py train CREATURE/dataset CREATURE/rig_info.json CREATURE/reader.pt

Then per move, all on this machine (MiniMax H3 through stable-diffusion.cpp films it):

    python scripts/twocam_mocap.py film CREATURE/ "it rears up and howls" CLIPS/ --takes 3 --reader CREATURE/reader.pt
    python scripts/twocam_mocap.py capture CLIPS/take1_s1.webm CREATURE/reader.pt CREATURE/rig_info.json         CREATURE/rigged.blend OUT/ --ref CREATURE/ref.png

`capture` is read + fit + apply; each also runs alone (`check` judges a read clip before the
fit). `exam` fits a made-up motion with known joints and scores it in centimetres: no
video, Blender or weights needed. Nothing is downloaded without --yes (DINOv2 for `train`,
346 MB; the video model for `film --download`, about 30 GB).
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
PREPARE_SCRIPT = REPO / "scripts" / "blender_twocam_prepare.py"
DATASET_SCRIPT = REPO / "scripts" / "blender_twocam_dataset.py"
DINO_DOWNLOAD = ("the joint reader starts from Meta's DINOv2 with registers (facebook/dinov2-with-registers-base, "
                 "Apache-2.0, 346 MB download)")
NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
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
    if args.out.suffix != ".npz":
        raise CaptureError(f"the detections file must end in .npz (numpy would write {args.out}.npz)")
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
        args.video, args.reader, args.rig_info, args.made_with, args.video_license, args.reader_license))
    if args.overlay:
        reader.overlay(det["frames"], det, Layout.from_rig_info(info), args.overlay, det["fps"])
    summary = {"detections": str(args.out), "frames": int(len(det["front_uv"])), "fps": det["fps"],
               "mean_confidence": {"front": float(det["front_conf"].mean()), "side": float(det["side_conf"].mean())},
               "calibration_converged": bool(det["calib_converged"])}
    if not det["calib_converged"]:
        summary["warning"] = "the first frame did not align to the reference: a plain stretch was used"
    return summary


def clip_report(d: dict, info: dict, feet=None, align: bool = True, swaps: bool = False) -> dict:
    """The clip judged as it would be fitted (re-timed when that helps), and as generated."""
    _, sync = solve.assess(d, info, feet=feet, align=align, swaps=swaps)
    sync.pop("per_frame_gap_px")
    return {"verdict": sync.pop("verdict"), "verdict_as_generated": sync.pop("verdict_as_generated"),
            "feet_agreement": sync.pop("paw_agreement"), "sync": sync}


def cmd_check(args) -> dict:
    info = load_rig_info(args.rig_info)
    d, _ = load_detections(args.det, info)
    return clip_report(d, info, args.feet, align=not args.no_align, swaps=args.fix_swaps)


def cmd_fit(args) -> dict:
    info = load_rig_info(args.rig_info)
    d, fps = load_detections(args.det, info)
    report = clip_report(d, info, args.feet, align=not args.no_align, swaps=args.fix_swaps)
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
    warnings = []
    if source is None:
        warnings.append("no licence record beside the model: the GLB's record says so. Pass --source-model "
                        "with the model whose .provenance.json names its licence")
    if report.get("max_joint_error_mm") is not None and report["max_joint_error_mm"] > 1.0:
        warnings.append("Blender's joints land more than 1 mm from the solver's: the rig on disk is not the "
                        "one rig_info.json describes")
    if warnings:
        report["warnings"] = warnings
    return report


def prepare_command(blender: Path, glb: Path, out: Path, length_m: float, axis: str = "y",
                    yaw_deg: float | None = None, samples: int = 64, render: bool = True) -> list[str]:
    cmd = [str(blender), "--background", "--factory-startup", "--disable-autoexec",
           "--python-exit-code", "1", "--python", str(PREPARE_SCRIPT), "--",
           str(glb), str(out), "--length-m", str(length_m), "--axis", axis, "--samples", str(samples)]
    if yaw_deg is not None:
        cmd += ["--yaw-deg", str(yaw_deg)]
    if not render:
        cmd.append("--no-render")
    return cmd


def count_sentence(paint: dict) -> str:
    """"Exactly four legs." or "Exactly two arms, two legs." from the limbs' labels."""
    kinds: dict[str, int] = {}
    for group in paint.values():
        word = group["label"].split()[-1]
        kinds[word] = kinds.get(word, 0) + 1
    parts = [f"{NUMBERS[n] if n < len(NUMBERS) else n} {word}{'s' if n != 1 else ''}" for word, n in kinds.items()]
    return "Exactly " + ", ".join(parts) + "." if parts else ""


def read_rgba(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise CaptureError(f"{path} is not an RGBA render")
    return img[..., [2, 1, 0, 3]]


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    import cv2

    ok, data = cv2.imencode(Path(path).suffix or ".png", np.ascontiguousarray(rgb[..., ::-1]))
    if not ok:
        raise CaptureError(f"cannot encode {path}")
    data.tofile(str(path))


def joints_preview(ref: np.ndarray, info: dict) -> np.ndarray:
    """The reference frame with every rest joint drawn on, both views (left pink, right blue)."""
    import cv2

    layout = Layout.from_rig_info(info)
    img = np.ascontiguousarray(ref.copy())
    rest = np.array([k["rest"] for k in info["keypoints"]])
    front, side = layout.project(rest)
    for k, f, s in zip(info["keypoints"], front, side):
        n = k["name"]
        colour = (245, 60, 209) if n.endswith("_L") else (47, 107, 255) if n.endswith("_R") else (255, 255, 255)
        for (u, v), x0 in ((f, 0), (s, layout.w_front)):
            cv2.circle(img, (int(u + x0), int(v)), 4, colour, -1, cv2.LINE_AA)
    return img


def cmd_prepare(args) -> dict:
    from image_to_3dlab.twocam import records, rigging

    if not args.glb.is_file():
        raise CaptureError(f"no rigged model at {args.glb}")
    blender = find_blender_or_fail(args.blender)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cmd = prepare_command(blender, args.glb, out, args.length_m, args.axis, args.yaw_deg, args.samples,
                          not args.no_render)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    line = next((x for x in result.stdout.splitlines() if x.startswith("I2L_TWOCAM_PREPARE::")), None)
    if result.returncode != 0 or line is None:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        raise CaptureError(f"Blender failed (exit {result.returncode}):\n{tail}")
    plan = json.loads(line.split("::", 1)[1])
    info = load_rig_info(out / "rig_info.json")
    layout = Layout.from_rig_info(info)
    creature = {
        "name": args.name or out.name, "noun": args.noun, "describe": args.describe or "",
        "colour_code": rigging.colour_code(plan["paint"]), "count": count_sentence(plan["paint"]),
        "layout": {k: v for k, v in layout.to_dict().items() if k != "Z_CENTER"},
        "paint": {k: {"color": g["color"], "bones": g["bones"]} for k, g in plan["paint"].items()},
        "feet": solve.ground_joints(info),
        "prep": {"length_m": args.length_m, "axis": args.axis, "yaw_deg": plan["yaw_deg"]},
    }
    (out / "creature.json").write_text(json.dumps(creature, indent=1), encoding="utf-8")
    source = records.source_record_for(args.source_model or args.glb)
    if source is not None:  # the prepared rig keeps the model's licence, so apply finds it
        records.write_sidecar(out / "rigged.blend", source)
    summary = {"creature": str(out), "limbs": plan["limbs"], "bones": plan["bones"], "keypoints": plan["keypoints"],
               "turned_deg": plan["yaw_deg"], "size_m": plan["size_m"], "colour_code": creature["colour_code"],
               "feet": creature["feet"], "licence": (source or {}).get("license") or records.NO_LICENCE}
    if not args.no_render:
        ref = layout.compose(read_rgba(out / "ref_front.png"), read_rgba(out / "ref_side.png"))
        write_rgb(out / "ref.png", ref)
        write_rgb(out / "preview.png", joints_preview(ref, info))
        summary.update(ref=str(out / "ref.png"), preview=str(out / "preview.png"),
                       render_device=plan.get("render_device"))
    if not plan["limbs"]:
        summary["warning"] = ("no left/right limbs were found: check the turn in preview.png, or pass --yaw-deg; "
                              "without painted limbs a video model easily swaps left and right")
    return summary


def ask_yes(what: str, yes: bool) -> None:
    """A download needs an affirmative answer: --yes, or y typed at a terminal."""
    if yes:
        return
    if sys.stdin is not None and sys.stdin.isatty():
        answer = input(f"Download {what}? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            return
    raise CaptureError(f"not downloaded: {what}. Pass --yes to download it")


def dataset_commands(blender: Path, blend: Path, rig_info: Path, out: Path, count: int, procs: int,
                     seed: int, scale: float) -> list[list[str]]:
    per = -(-count // procs)
    return [[str(blender), "--background", "--factory-startup", "--disable-autoexec", str(blend),
             "--python-exit-code", "1", "--python", str(DATASET_SCRIPT), "--", str(rig_info), str(out),
             str(k * per), str(min(per, count - k * per)), str(seed), str(scale)]
            for k in range(procs) if k * per < count]


def cmd_dataset(args) -> dict:
    import time

    load_rig_info(args.rig_info)
    if not args.blend.is_file():
        raise CaptureError(f"no prepared rig at {args.blend}: run prepare first")
    blender = find_blender_or_fail(args.blender)
    args.out.mkdir(parents=True, exist_ok=True)
    logs = []
    procs = []
    for k, cmd in enumerate(dataset_commands(blender, args.blend, args.rig_info, args.out, args.count,
                                             args.procs, args.seed, args.scale)):
        log = open(args.out / f"render_{k}.log", "w", encoding="utf-8")  # noqa: SIM115 - closed below
        logs.append(log)
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    start = time.time()
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(10)
            done = len(list(args.out.glob("*_meta.json")))
            if not args.quiet:
                print(f"{done}/{args.count} poses rendered, {time.time() - start:.0f} s", file=sys.stderr, flush=True)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
    failed = [k for k, p in enumerate(procs) if p.returncode != 0]
    done = len(list(args.out.glob("*_meta.json")))
    if failed:
        raise CaptureError(f"Blender stopped in render_{failed[0]}.log ({done} poses kept; run again to resume)")
    return {"dataset": str(args.out), "poses": done, "minutes": round((time.time() - start) / 60, 1)}


def cmd_train(args) -> dict:
    from image_to_3dlab.twocam import train

    info = load_rig_info(args.rig_info)
    samples = train.check_dataset(args.dataset, info)
    if not train.backbone_cached():
        ask_yes(DINO_DOWNLOAD, args.yes)
    device = solve.pick_device(args.device)
    log = None if args.quiet else (lambda msg: print(msg, file=sys.stderr, flush=True))
    result = train.train(args.dataset, info, args.out, epochs=args.epochs, batch=args.batch, device=device,
                         workers=args.workers, log=log)
    return {"reader": str(args.out), "samples": samples, "device": device,
            "held_out_error_px": {k: round(v, 2) for k, v in result.items()}}


def cmd_film(args) -> dict:
    from image_to_3dlab import host
    from image_to_3dlab.twocam import prompt as prompts
    from image_to_3dlab.twocam import records, video

    creature_dir = args.creature
    try:
        creature = json.loads((creature_dir / "creature.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"{creature_dir} is not a prepared creature (no creature.json): {exc}") from exc
    start = creature_dir / "ref.png"
    if not start.is_file():
        raise CaptureError(f"no reference frame at {start}: prepare the creature without --no-render")
    plan = video.weights_plan()
    if not plan["ready"]:
        if not args.download:
            raise CaptureError(f"the video model is not here: {', '.join(plan['missing'])}. Pass --download "
                               f"(and --yes) to fetch {plan['download_gb']} GB. {plan['license']}: "
                               f"{plan['license_note']}")
        ask_yes(f"{plan['engine']}, {plan['download_gb']} GB ({plan['license']}; {plan['license_note']})", args.yes)
        video.download(log=lambda msg: print(msg, file=sys.stderr, flush=True))
    text = prompts.make_prompt(creature, args.action, args.end, seconds=round(args.seconds))
    binary = host.executable(REPO / "vendor" / "sdcpp", "sd-cli")
    args.out.mkdir(parents=True, exist_ok=True)
    log = None if args.quiet else (lambda msg: print(msg, file=sys.stderr, flush=True))
    rig_info = creature_dir / "rig_info.json"
    reader_path = args.reader
    takes = []
    for k in range(args.takes):
        seed = args.seed + k
        clip = args.out / f"take{k + 1}_s{seed}.webm"
        video.film(binary, text, start, clip, width=args.width, height=args.height, seconds=args.seconds,
                   seed=seed, steps=args.steps, log=log)
        record = {"schema_version": 1, "created_at": records.now(), "tool": "scripts/twocam_mocap.py film",
                  "made_with": plan["engine"], "license": plan["license"], "license_url": plan["license_url"],
                  "license_note": plan["license_note"], "prompt": text, "seed": seed,
                  "start_frame": records.sha256_file(start), "settings": {
                      "width": args.width, "height": args.height, "seconds": args.seconds, "steps": args.steps}}
        records.write_sidecar(clip, record)
        take = {"clip": str(clip), "seed": seed}
        if reader_path:
            det = clip.with_suffix(".npz")
            cmd_read(argparse.Namespace(video=clip, reader=reader_path, rig_info=rig_info, out=det, ref=start,
                                        no_calib=False, overlay=None, fps=None, made_with=plan["engine"],
                                        video_license=plan["license"], reader_license=None, device=args.device))
            info = load_rig_info(rig_info)
            d, _ = load_detections(det, info)
            take["check"] = clip_report(d, info)["verdict"]
        takes.append(take)
    rank = {"good": 0, "doubtful": 1, "bad": 2}
    checked = [t for t in takes if "check" in t]
    best = min(checked, key=lambda t: (rank[t["check"]["level"]], t["check"]["height_gap_fraction"])) if checked else None
    return {"prompt": text, "takes": takes, "best": best["clip"] if best else None}


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
        limbs = solve.limb_pairs(info)
        if not limbs:
            raise CaptureError("--swap-frames needs a rig with left/right limbs (bones named ..._L and ..._R)")
        pairs = next(iter(limbs.values()))
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
        s.add_argument("--reader-license", help="the terms the reader file is under, for the provenance record")

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

    s = sub.add_parser("prepare", help="make a rigged GLB ready: turned, sized, limbs named and painted")
    s.add_argument("glb", type=Path, help="the rigged model (e.g. the Animate tab's *_rigged.glb)")
    s.add_argument("out", type=Path, help="the creature's folder")
    s.add_argument("--length-m", type=float, required=True, help="its real size along --axis, in metres")
    s.add_argument("--axis", choices=("x", "y", "z"), default="y",
                   help="y: nose to tail (default), x: side to side, z: height")
    s.add_argument("--noun", required=True, help='what it is, for the video prompt: "grey wolf"')
    s.add_argument("--describe", help='how it looks, for the video prompt: "grey fur, cream chest"')
    s.add_argument("--name", help="a short name (default: the folder's)")
    s.add_argument("--yaw-deg", type=float, help="turn it by this much instead of finding the turn")
    s.add_argument("--samples", type=int, default=64, help="render samples for the reference frame")
    s.add_argument("--no-render", action="store_true", help="skip the reference frame")
    s.add_argument("--blender", type=Path)
    s.add_argument("--source-model", type=Path, help="the model whose licence sidecar to carry over")

    s = sub.add_parser("film", help="film the creature from both cameras with MiniMax H3, on this machine")
    s.add_argument("creature", type=Path, help="a prepared creature's folder (creature.json, ref.png)")
    s.add_argument("action", help='one sentence: "the wolf rears up and howls, then drops back onto all fours"')
    s.add_argument("out", type=Path, help="folder for the takes")
    s.add_argument("--end", help='how it ends, in both views: "standing on all four legs"')
    s.add_argument("--seconds", type=float, default=3.0)
    s.add_argument("--width", type=int, default=1024)
    s.add_argument("--height", type=int, default=576)
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--takes", type=int, default=1, help="film this many takes, seeds counting up from --seed")
    s.add_argument("--reader", type=Path, help="read and judge every take with this reader; the best is named")
    s.add_argument("--device", default="auto", help="for reading the takes")
    s.add_argument("--download", action="store_true", help="fetch the video model if it is missing")
    s.add_argument("--yes", action="store_true", help="say yes to the download without asking")
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("dataset", help="render the poses a reader learns the creature from")
    s.add_argument("blend", type=Path, help="the prepared rigged.blend")
    s.add_argument("rig_info", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--count", type=int, default=2200)
    s.add_argument("--procs", type=int, default=2, help="Blender processes side by side")
    s.add_argument("--seed", type=int, default=500)
    s.add_argument("--scale", type=float, default=0.6, help="render size as a fraction of the layout")
    s.add_argument("--blender", type=Path)
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("train", help="teach the joint reader the creature, from its rendered poses")
    s.add_argument("dataset", type=Path)
    s.add_argument("rig_info", type=Path)
    s.add_argument("out", type=Path, help="the reader file to write (.pt)")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--batch", type=int, default=20)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--device", default="auto")
    s.add_argument("--yes", action="store_true", help="say yes to DINOv2's download without asking")
    s.add_argument("--quiet", action="store_true")

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
    s.add_argument("--no-align", action="store_true", help="judge the clip without re-timing it")
    s.add_argument("--fix-swaps", action="store_true", help="judge it after fixing side-view swaps")

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


COMMANDS = {"prepare": cmd_prepare, "dataset": cmd_dataset, "train": cmd_train, "film": cmd_film,
            "read": cmd_read, "check": cmd_check, "fit": cmd_fit, "apply": cmd_apply,
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
