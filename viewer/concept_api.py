"""From a reference to a concept, in the viewer and for other steps (function 2).

A concept is an image job of `image_api` with more arguments: the vision half of the
encoder and the reference pictures (`image_to_3dlab.concept.edit`), the words of one of
the four ways (`image_to_3dlab.concept.prompts`). So it shares the image queue, progress
stream and provenance; this module adds the inputs, the request, and the
`<image>.concept.json` record with the silhouette IoU.

    GET  /api/concept                    the ways, the limits, what is installed
    GET  /api/concept/record?picture=..  a concept's record, and whether its IoU is still being measured
    POST /api/concept/upload?name=..     a reference picture (the raw bytes), kept as PNG
    POST /api/concept/prepare            {source, kind, view}: a map from a model or a picture
    POST /api/concept/jobs               {mode, ..., shape: [...], style_refs: [...], seed}
    POST /api/concept/judge              {picture, verdict: keep | discard, note}: the user's judgement

Every path is relative to the output folder, and nothing outside it is read. The job's
progress, status and picture come from `/api/image/<job id>/...`, like any picture.

`generate(prompt, references, seed)` is the same edit as one call, for scripts and other
steps (the style bible sends its style pictures through it).
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
from PIL import Image, UnidentifiedImageError

import image_api
from blender_edit_api import check_request

from image_to_3dlab.blender import find_blender
from image_to_3dlab.checks.silhouette import fit_mask, iou
from image_to_3dlab.concept import edit, maps, prompts, sidecar

HEADER = "reference-concept"
FOLDER = "concept"
UPLOAD_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD = 25 * 1024 * 1024
MAX_BODY = 64 * 1024
MODEL_KINDS = ("depth", "colour", "pose")
PICTURE_KINDS = ("lineart", "picture")
# Other views take 2-4 attempts, the user picking the best in the studio (16/10/2026).
ATTEMPTS = {"default": 1, "most": 4}
MAP_SIZE = 1024
MEASURING: set[str] = set()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(output: Path, relative: str) -> Path:
    """`relative` under the output folder, as an existing file; anything else is refused."""
    root = Path(output).resolve()
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root) or Path(str(relative)).is_absolute():
        raise ValueError(f"{relative} is not in the output folder")
    if not path.is_file():
        raise FileNotFoundError(f"{relative} was not found in the output folder")
    return path


def _relative(output: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(Path(output).resolve()).as_posix()


def info(output: Path) -> dict[str, Any]:
    try:
        image_api.resolve_weights()
        weights = True
    except image_api.MissingWeights:
        weights = False
    return {
        "modes": list(prompts.MODES),
        "structures": list(prompts.STRUCTURES),
        "guides": list(prompts.GUIDES),
        "views": list(prompts.VIEWS),
        "map_views": list(maps.VIEWS),
        "max_references": edit.MAX_REFERENCES,
        "attempts": ATTEMPTS,
        "ready": {"binary": image_api.BINARY.exists(), "weights": weights,
                  "vision": edit.find_vision(image_api.HF_HUB) is not None, "blender": find_blender() is not None},
        "license": {"name": image_api.LICENSE_NAME, "url": image_api.LICENSE_URL},
    }


def save_upload(output: Path, data: bytes, filename: str) -> dict[str, str]:
    """A reference picture, re-encoded as PNG under concept/uploads/ and named by its content."""
    if Path(filename).suffix.lower() not in UPLOAD_TYPES:
        raise ValueError("send a PNG, JPG or WebP picture")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            picture = image.convert("RGBA")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"{Path(filename).name} is not a picture that can be read") from error
    target = Path(output) / FOLDER / "uploads" / f"{hashlib.sha256(data).hexdigest()[:16]}.png"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        picture.save(target)
    return {"path": _relative(output, target)}


def prepare(output: Path, source: str, kind: str, view: str = "front") -> dict[str, Any]:
    """A map to draw from: depth, colour render or pose map of a model; lineart of a picture.

    A model's views are all rendered on its first request and kept beside each other under
    concept/maps/<its hash>/, so asking for another view or kind costs nothing.
    """
    path = inside(output, source)
    folder = Path(output) / FOLDER / "maps" / _sha256(path)[:16]
    if path.suffix.lower() == ".glb":
        if kind not in MODEL_KINDS:
            raise ValueError(f"a model gives one of {', '.join(MODEL_KINDS)}, not {kind!r}")
        if view not in maps.VIEWS:
            raise ValueError(f"the view is one of {', '.join(maps.VIEWS)}, not {view!r}")
        if not all((folder / f"{v}_colour.png").is_file() for v in maps.VIEWS):
            maps.render(path, folder, maps.VIEWS, MAP_SIZE)
        if kind == "depth":
            made = maps.depth_on_black(folder / f"{view}_depth.png", folder / f"{view}_depth_black.png")
            mask = made
        elif kind == "colour":
            made = maps.flatten(folder / f"{view}_colour.png", folder / f"{view}_colour_grey.png")
            mask = folder / f"{view}_colour.png"
        else:
            bones = folder / "bones.json"
            if not bones.is_file():
                raise ValueError(f"{path.name} has no skeleton to draw a pose from")
            points = maps.pose_points(json.loads(bones.read_text(encoding="utf-8"))[view])
            made, mask = maps.draw_pose(points, MAP_SIZE, folder / f"{view}_pose.png"), None
    elif path.suffix.lower() in UPLOAD_TYPES:
        if kind not in PICTURE_KINDS:
            raise ValueError(f"a picture gives one of {', '.join(PICTURE_KINDS)}, not {kind!r}")
        if kind == "picture":
            return {"path": _relative(output, path), "kind": kind, "view": None, "mask": None}
        folder.mkdir(parents=True, exist_ok=True)
        made, mask, view = maps.lineart(path, folder / "lineart.png", MAP_SIZE), None, None
    else:
        raise ValueError("the source is a GLB model or a PNG, JPG or WebP picture")
    return {"path": _relative(output, made), "kind": kind, "view": view,
            "mask": _relative(output, mask) if mask else None}


def _text(request: dict, key: str, label: str | None = None) -> str:
    value = str(request.get(key) or "").strip()
    if not value:
        raise ValueError(f"the {label or key} is needed")
    if len(value) > 400:
        raise ValueError(f"the {label or key} is longer than 400 characters")
    return value


def build(output: Path, request: dict[str, Any]) -> dict[str, Any]:
    """A request from the studio as the prompt, the pictures in order, and what the IoU is measured on."""
    mode = request.get("mode")
    if mode not in prompts.MODES:
        raise ValueError(f"the way is one of {', '.join(prompts.MODES)}")
    depth = bool(request.get("depth"))
    shape = [inside(output, p) for p in request.get("shape") or []]
    style_refs = [inside(output, p) for p in request.get("style_refs") or []]
    needed = prompts.shape_pictures(mode, depth=depth)
    if len(shape) != needed:
        raise ValueError(f"this way takes {needed} shape picture{'s' if needed > 1 else ''}, got {len(shape)}")
    if len(shape) + len(style_refs) > edit.MAX_REFERENCES:
        raise ValueError(f"at most {edit.MAX_REFERENCES} pictures in all")
    style = str(request.get("style") or "").strip()[:400]
    if mode == "redraw":
        subject = _text(request, "subject")
        prompt = prompts.redraw(subject, _text(request, "colours"), structure=request.get("structure") or "depth",
                                parts=str(request.get("parts") or "")[:400], style=style)
        name, against, fit = subject, shape[0], True
    elif mode == "dress":
        guide = request.get("guide") or "body"
        outfit = _text(request, "outfit")
        prompt = prompts.dress(outfit, guide=guide, style=style)
        name, against, fit = outfit, shape[0] if guide == "body" else None, False
    elif mode == "add_piece":
        piece = _text(request, "piece")
        prompt, name, against, fit = prompts.add_piece(piece, style=style), piece, shape[0], False
    else:
        view = request.get("view")
        prompt = prompts.other_view(view, depth=depth, style=style)
        name, against, fit = f"{view} view", shape[1] if depth else None, True
    prompt += prompts.style_sentence(len(shape) + 1, len(style_refs))
    return {"mode": mode, "prompt": prompt, "name": f"concept {mode.replace('_', ' ')} {name}",
            "references": [(p, "shape") for p in shape] + [(p, "style") for p in style_refs],
            "iou_against": against, "fit": fit,
            "inputs": {k: request[k] for k in ("structure", "guide", "view", "depth") if k in request}}


def reference_mask(path: Path) -> np.ndarray | None:
    """The subject of a shape picture, when it says so itself: a render's alpha, a depth map's non-black."""
    image = Image.open(path)
    if image.mode in ("RGBA", "LA"):
        alpha = np.asarray(image.getchannel("A"))
        if alpha.min() < 128 <= alpha.max():
            return alpha > 127
    if path.name.endswith("_depth_black.png"):
        return np.asarray(image.convert("L")) > 0
    sibling = path.with_name(path.name.replace("_colour_grey.png", "_colour.png"))
    if sibling != path and sibling.is_file():
        return reference_mask(sibling)
    return None


def result_mask(path: Path) -> np.ndarray:
    """The generated subject, cut out with the project's background remover."""
    from image_to_3dlab import matte

    cut, _ = matte.cut_out(Image.open(path))
    return np.asarray(cut)[..., 3] > 127


def measure(reference: Path, result: Path, fit: bool) -> float | None:
    """Silhouette IoU of the result against its shape picture, fitted to the frame or as framed."""
    import cv2

    wanted = reference_mask(reference)
    if wanted is None:
        return None
    got = result_mask(result)
    if got.shape != wanted.shape:
        got = cv2.resize(got.astype(np.uint8), wanted.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return round(iou(fit_mask(wanted), fit_mask(got)) if fit else iou(wanted, got), 3)


def _vision() -> Path:
    vision = edit.find_vision(image_api.HF_HUB)
    if vision is None:
        raise image_api.MissingWeights([edit.VISION_FILE])
    return vision


def _reference_names(references: list[tuple[Path, str]]) -> list[dict[str, str]]:
    return [{"name": p.name, "role": role, "sha256": _sha256(p)} for p, role in references]


def _seed(request: dict[str, Any]) -> int:
    try:
        return int(request["seed"])
    except (KeyError, TypeError, ValueError):
        return random.randrange(2 ** 31 - 1)


def start(output: Path, request: dict[str, Any]) -> image_api.ImageJob:
    """Queue the edit as an image job; its record and IoU follow when the picture is done."""
    plan = build(output, request)
    vision = _vision()
    weights = image_api.resolve_weights()
    paths = [p for p, _ in plan["references"]]
    job = image_api.MANAGER.create(
        plan["prompt"], image_api.clean_settings({"seed": _seed(request)}),
        extra_args=edit.edit_args(paths, vision),
        extra_provenance={"concept": {"mode": plan["mode"], "references": _reference_names(plan["references"])}},
        name=plan["name"])
    thread = threading.Thread(target=_run, args=(output, job, plan, weights, vision), daemon=True)
    job.thread = thread
    thread.start()
    return job


def wait(job: image_api.ImageJob, timeout: float | None = None) -> None:
    """Until the job and its record are done (for scripts and tests)."""
    thread = getattr(job, "thread", None)
    if thread is not None:
        thread.join(timeout)


def _run(output: Path, job: image_api.ImageJob, plan: dict, weights: dict, vision: Path) -> None:
    image_api.run_job(job, image_api.MANAGER, weights)
    if job.status != "done":
        return
    command = edit.with_references(image_api.build_command(job.prompt, job.settings, job.output_path, weights),
                                   [p for p, _ in plan["references"]], vision)
    record = dict(mode=plan["mode"], prompt=job.prompt, seed=job.settings["seed"], command=command,
                  references=plan["references"], inputs=plan["inputs"])
    sidecar.write(job.output_path, **record)
    if plan["iou_against"] is None:
        return
    picture = job.picture_path()
    MEASURING.add(picture)
    try:
        value = measure(plan["iou_against"], job.output_path, plan["fit"])
    except Exception:  # noqa: BLE001 - a picture without its IoU is still a picture
        value = None
    finally:
        MEASURING.discard(picture)
    sidecar.write(job.output_path, **record, iou=value)


def record(output: Path, picture: str) -> dict[str, Any]:
    return {"record": sidecar.read(inside(output, picture)), "measuring": picture in MEASURING}


VERDICTS = ("keep", "discard")


def judge(output: Path, request: dict[str, Any]) -> dict[str, Any]:
    """The user's judgement by eye, kept in the concept's record and in concept/judgements.jsonl."""
    picture = str(request.get("picture") or "")
    verdict = request.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"the verdict is one of {', '.join(VERDICTS)}")
    path = inside(output, picture)
    data = sidecar.read(path)
    if data is None:
        raise ValueError(f"{picture} has no concept record")
    judgement = {"verdict": verdict, "note": str(request.get("note") or "")[:1000],
                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    data["judgement"] = judgement
    sidecar.path_for(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log = Path(output) / FOLDER / "judgements.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"picture": picture, "mode": data.get("mode"), "image_sha256": data["image"]["sha256"],
                                 "iou": data.get("iou"), **judgement}) + "\n")
    return {"picture": picture, "judgement": judgement}


def generate(prompt: str, references: list[Path], seed: int, name: str | None = None) -> Path:
    """One edit, run to the end: the picture's path, or RuntimeError with sd-cli's reason.

    `references` are any readable pictures (shape first, then style), in the order the
    prompt names them. The provenance beside the picture lists them by name and SHA-256.
    """
    vision = _vision()
    paths = [Path(p) for p in references]
    job = image_api.MANAGER.create(
        prompt, image_api.clean_settings({"seed": seed}), extra_args=edit.edit_args(paths, vision),
        extra_provenance={"concept": {"references": _reference_names([(p, "reference") for p in paths])}},
        name=name)
    image_api.run_job(job, image_api.MANAGER)
    if job.status != "done":
        raise RuntimeError(job.error or f"the picture was not made ({job.status})")
    return job.output_path


def _body(handler, maximum: int) -> bytes:
    length = int(handler.headers.get("Content-Length") or 0)
    if not 0 < length <= maximum:
        raise ValueError(f"send a body of at most {maximum // 1024} KB")
    return handler.rfile.read(length)


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    url = urlsplit(handler.path)
    if not (url.path == "/api/concept" or url.path.startswith("/api/concept/")):
        return False
    parts = [unquote(p) for p in url.path.strip("/").split("/")]
    query = parse_qs(url.query)
    output = Path(output)
    try:
        check_request(handler)
        if mutation and (not handler.headers.get("Origin") or handler.headers.get("X-AssetFurnace-Request") != HEADER):
            raise PermissionError("Use the Studio page to draw from a reference")
        if not mutation and parts == ["api", "concept"]:
            handler._send_json(200, info(output))
        elif not mutation and parts == ["api", "concept", "record"]:
            handler._send_json(200, record(output, (query.get("picture") or [""])[0]))
        elif mutation and parts == ["api", "concept", "upload"]:
            handler._send_json(200, save_upload(output, _body(handler, MAX_UPLOAD), (query.get("name") or [""])[0]))
        elif mutation and parts[2:] in (["prepare"], ["jobs"], ["judge"]):
            body = json.loads(_body(handler, MAX_BODY) or b"null")
            if not isinstance(body, dict):
                raise ValueError("send a JSON object")
            if parts[2] == "judge":
                handler._send_json(200, judge(output, body))
            elif parts[2] == "prepare":
                handler._send_json(200, prepare(output, body.get("source", ""), body.get("kind", ""),
                                                body.get("view") or "front"))
            else:
                job = start(output, body)
                handler._send_json(202, {"job_id": job.id, "settings": job.settings, "prompt": job.prompt,
                                         "events_url": f"/api/image/{job.id}/events",
                                         "status_url": f"/api/image/{job.id}/status"})
        else:
            handler._send_json(404, {"error": "No such concept request"})
    except PermissionError as exc:
        handler._send_json(403, {"error": str(exc)})
    except image_api.MissingWeights as exc:
        handler._send_json(503, {"error": str(exc), "needs_setup": True, "missing": exc.missing})
    except FileNotFoundError as exc:
        handler._send_json(404, {"error": str(exc)})
    except RuntimeError as exc:
        handler._send_json(409, {"error": str(exc)})
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        handler._send_json(400, {"error": str(exc)})
    return True
