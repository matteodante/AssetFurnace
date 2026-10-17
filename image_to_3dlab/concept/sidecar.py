"""`<image>.concept.json`: which pictures shaped a concept, and how.

It sits beside the picture's own provenance (licence, generator) and adds what that one
does not know: each reference by file name and SHA-256 with its role (shape or style),
the inputs it was made from, the weights by file name, and the silhouette IoU against
the reference when one was measured. Only file names: no folder of the machine goes in.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_to_3dlab.concept.prompts import MODES

ROLES = ("shape", "style")
SUFFIX = ".concept.json"
WEIGHT_FLAGS = {"--diffusion-model": "diffusion_model", "--llm": "llm", "--vae": "vae",
                "--llm_vision": "llm_vision"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _models(command: list[str]) -> dict[str, str]:
    found = {}
    for flag, key in WEIGHT_FLAGS.items():
        if flag in command[:-1]:
            found[key] = Path(command[command.index(flag) + 1].replace("\\", "/")).name
    return found


def path_for(image: Path) -> Path:
    image = Path(image)
    return image.with_name(image.name + SUFFIX)


def write(image: Path, *, mode: str, prompt: str, seed: int, command: list[str],
          references: list[tuple[Path, str]], inputs: dict[str, Any] | None = None,
          iou: float | None = None) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode is one of {', '.join(MODES)}, got {mode!r}")
    for _, role in references:
        if role not in ROLES:
            raise ValueError(f"a reference's role is one of {', '.join(ROLES)}, got {role!r}")
    record = {
        "image": {"name": Path(image).name, "sha256": sha256(image)},
        "mode": mode,
        "prompt": prompt,
        "seed": seed,
        "references": [{"name": Path(p).name, "role": role, "sha256": sha256(p)} for p, role in references],
        "inputs": inputs or {},
        "models": _models(command),
        "iou": iou,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = path_for(image)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def update(image: Path, **fields: Any) -> dict[str, Any]:
    """Change some fields of an existing record and keep the rest (a judgement, an IoU)."""
    data = read(image)
    if data is None:
        raise ValueError(f"{Path(image).name} has no concept record")
    data.update(fields)
    path_for(image).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def read(image: Path) -> dict[str, Any] | None:
    path = path_for(image)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
