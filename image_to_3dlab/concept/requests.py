"""A concept request, from words and pictures to a drawn picture with its record (feature 2).

The studio's "Draw from a reference" panel (`viewer/concept_api.py`), the command line
(`scripts/reference_concept.py make`) and the batch step `concept` all go through here:

    plan(mode, shape, style_refs, fields)     the prompt, the pictures in order, what to measure
    draw(plan, seed, output_path)             one picture, its `<picture>.concept.json`, its IoU

`plan` takes paths already found (the studio checks first that they are in its output
folder). `draw` runs Qwen-Image 2.1 in edit mode through `image_to_3dlab.text_image`, so
it holds the shared GPU lock while it draws. The silhouette IoU against the shape picture
is measured on cut-out masks, a geometric measure, and recorded, not used to decide.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_to_3dlab import text_image
from image_to_3dlab.checks.silhouette import fit_mask, iou
from image_to_3dlab.concept import edit, prompts, sidecar

MAX_TEXT = 400


def _text(fields: dict, key: str, label: str | None = None) -> str:
    value = str(fields.get(key) or "").strip()
    if not value:
        raise ValueError(f"the {label or key} is needed")
    if len(value) > MAX_TEXT:
        raise ValueError(f"the {label or key} is longer than {MAX_TEXT} characters")
    return value


def plan(mode: str, shape: list[Path], style_refs: list[Path], fields: dict[str, Any],
         bible=None) -> dict[str, Any]:
    """The prompt, the pictures in order (shape first, then style), the picture the IoU
    is measured against and how. `fields`: subject, colours, parts, structure, outfit,
    guide, piece, view, style, depth, as the mode needs them. `bible`: a style bible
    (`image_to_3dlab.style.apply.resolve`), whose words join the style words and whose
    pictures join the style pictures, as many as there is room for."""
    if mode not in prompts.MODES:
        raise ValueError(f"the way is one of {', '.join(prompts.MODES)}")
    if bible is not None:
        fields = {**fields, "style": ", ".join(w for w in (str(fields.get("style") or "").strip(), bible.words) if w)}
        room = max(0, edit.MAX_REFERENCES - len(shape) - len(style_refs))
        style_refs = [*style_refs, *bible.references[:room]]
    depth = bool(fields.get("depth"))
    needed = prompts.shape_pictures(mode, depth=depth)
    if len(shape) != needed:
        raise ValueError(f"this way takes {needed} shape picture{'s' if needed > 1 else ''}, got {len(shape)}")
    if len(shape) + len(style_refs) > edit.MAX_REFERENCES:
        raise ValueError(f"at most {edit.MAX_REFERENCES} pictures in all")
    style = str(fields.get("style") or "").strip()[:MAX_TEXT]
    if mode == "redraw":
        subject = _text(fields, "subject")
        prompt = prompts.redraw(subject, _text(fields, "colours"), structure=fields.get("structure") or "depth",
                                parts=str(fields.get("parts") or "")[:MAX_TEXT], style=style)
        name, against, fit = subject, shape[0], True
    elif mode == "dress":
        guide = fields.get("guide") or "body"
        outfit = _text(fields, "outfit")
        prompt = prompts.dress(outfit, guide=guide, style=style)
        name, against, fit = outfit, shape[0] if guide == "body" else None, False
    elif mode == "add_piece":
        piece = _text(fields, "piece")
        prompt, name, against, fit = prompts.add_piece(piece, style=style), piece, shape[0], False
    elif mode == "complete_piece":
        # A cut-out piece on grey (function 4): no outline of its own to measure against.
        piece = _text(fields, "piece")
        prompt, name, against, fit = prompts.complete_piece(piece, style=style), piece, None, False
    else:
        view = fields.get("view")
        prompt = prompts.other_view(view, depth=depth, style=style)
        name, against, fit = f"{view} view", shape[1] if depth else None, True
    prompt += prompts.style_sentence(len(shape) + 1, len(style_refs))
    from image_to_3dlab.style import apply

    return {"mode": mode, "prompt": prompt, "name": f"concept {mode.replace('_', ' ')} {name}",
            "references": [(p, "shape") for p in shape] + [(p, "style") for p in style_refs],
            "iou_against": against, "fit": fit,
            "inputs": {k: fields[k] for k in ("structure", "guide", "view", "depth") if k in fields},
            **({"style_record": apply.record(bible)} if bible is not None else {})}


# --------------------------------------------------------------------- the IoU

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


def measure(reference: Path, result: Path, fit: bool,
            cut: Callable[[Path], np.ndarray] = result_mask) -> float | None:
    """Silhouette IoU of the result against its shape picture, fitted to the frame or as framed."""
    import cv2

    wanted = reference_mask(reference)
    if wanted is None:
        return None
    got = cut(result)
    if got.shape != wanted.shape:
        got = cv2.resize(got.astype(np.uint8), wanted.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return round(iou(fit_mask(wanted), fit_mask(got)) if fit else iou(wanted, got), 3)


# --------------------------------------------------------------------- drawing

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_names(references: list[tuple[Path, str]]) -> list[dict[str, str]]:
    return [{"name": p.name, "role": role, "sha256": sha256(p)} for p, role in references]


def vision() -> Path:
    """The vision half of the encoder the edit needs; MissingWeights names it when absent."""
    found = edit.find_vision(text_image.HF_HUB)
    if found is None:
        raise text_image.MissingWeights([edit.VISION_FILE])
    return found


def draw(plan_: dict[str, Any], seed: int, output_path: Path, *, settings: dict[str, Any] | None = None,
         log: Callable[[str], None] = lambda line: None, session: str = "concept",
         measure_iou: Callable[[Path, Path, bool], float | None] = measure) -> Path:
    """One concept picture to the end: the picture, its provenance, and its
    `.concept.json` record with the silhouette IoU when there is an outline to measure.
    RuntimeError with sd-cli's reason, MissingWeights when the model is not installed."""
    eye = vision()
    weights = text_image.resolve_weights()
    paths = [p for p, _ in plan_["references"]]
    text_image.generate(plan_["prompt"], {**(settings or {}), "seed": seed}, output_path, weights=weights,
                        extra_args=edit.edit_args(paths, eye),
                        extra_provenance={"concept": {"mode": plan_["mode"],
                                                      "references": reference_names(plan_["references"])},
                                          **({"style": plan_["style_record"]} if plan_.get("style_record") else {})},
                        log=log, session=session)
    command = edit.with_references(text_image.build_command(plan_["prompt"], text_image.clean_settings(
        {**(settings or {}), "seed": seed}), output_path, weights), paths, eye)
    sidecar.write(output_path, mode=plan_["mode"], prompt=plan_["prompt"], seed=seed, command=command,
                  references=plan_["references"], inputs=plan_["inputs"])
    if plan_["iou_against"] is not None:
        try:
            value = measure_iou(plan_["iou_against"], output_path, plan_["fit"])
        except Exception:  # noqa: BLE001 - a picture without its IoU is still a picture
            value = None
        sidecar.update(output_path, iou=value)
    return output_path
