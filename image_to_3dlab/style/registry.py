"""The bibles AssetFurnace knows about, one per project, and which one the studio uses.

The list lives in `<output>/styles/bibles.json`. A bible made here sits next to it in
`<output>/styles/<name>/` and is stored by a path relative to `output/`, so moving the
whole checkout keeps it working; a bible kept in a game's own repository is stored by
its absolute path. Unregistering forgets a bible; it never deletes the folder.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from image_to_3dlab.style.bible import BibleError, load

_LOCK = threading.Lock()


def _file(output_root: Path) -> Path:
    return Path(output_root) / "styles" / "bibles.json"


def _read(output_root: Path) -> dict[str, Any]:
    path = _file(output_root)
    if not path.is_file():
        return {"active": None, "bibles": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BibleError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("bibles"), list):
        raise BibleError(f"{path} has no 'bibles' list")
    return data


def _write(output_root: Path, data: dict) -> None:
    path = _file(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".bibles-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
            out.write("\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _absolute(output_root: Path, stored: str) -> Path:
    path = Path(stored)
    return (path if path.is_absolute() else Path(output_root) / path).resolve()


def bibles(output_root: Path) -> list[dict[str, Any]]:
    """Every registered bible as `{"name", "path"}`, in the order they were added."""
    return [{"name": b["name"], "path": _absolute(output_root, b["path"])} for b in _read(output_root)["bibles"]]


def register(output_root: Path, root: Path) -> dict[str, Any]:
    """Add an existing bible folder to the list. Names must be unique among registered bibles."""
    output_root, root = Path(output_root), Path(root).resolve()
    name = load(root)["name"]
    with _LOCK:
        data = _read(output_root)
        for known in data["bibles"]:
            if _absolute(output_root, known["path"]) == root:
                raise BibleError(f"{root} is already registered as {known['name']!r}")
            if known["name"] == name:
                raise BibleError(f"another bible is already registered as {name!r}; rename one first")
        inside = root.is_relative_to(output_root.resolve())
        stored = root.relative_to(output_root.resolve()).as_posix() if inside else str(root)
        data["bibles"].append({"name": name, "path": stored})
        _write(output_root, data)
    return {"name": name, "path": root}


def unregister(output_root: Path, name: str) -> None:
    with _LOCK:
        data = _read(output_root)
        kept = [b for b in data["bibles"] if b["name"] != name]
        if len(kept) == len(data["bibles"]):
            raise BibleError(f"no bible is registered as {name!r}")
        data["bibles"] = kept
        if data.get("active") == name:
            data["active"] = None
        _write(output_root, data)


def set_active(output_root: Path, name: str | None) -> None:
    """Make `name` the bible the studio and `-` on the command line mean; None clears it."""
    with _LOCK:
        data = _read(output_root)
        if name is not None and not any(b["name"] == name for b in data["bibles"]):
            raise BibleError(f"no bible is registered as {name!r}")
        data["active"] = name
        _write(output_root, data)


def active(output_root: Path) -> dict[str, Any] | None:
    name = _read(output_root).get("active")
    return next((b for b in bibles(output_root) if b["name"] == name), None)


def find(output_root: Path, name_or_path: str) -> Path:
    """A bible folder from a registered name, `-` for the active one, or a folder path."""
    if name_or_path == "-":
        current = active(output_root)
        if current is None:
            raise BibleError("no bible is active; pick one with 'use'")
        return current["path"]
    for known in bibles(output_root):
        if known["name"] == name_or_path:
            return known["path"]
    path = Path(name_or_path)
    if (path / "bible.json").is_file():
        return path.resolve()
    raise BibleError(f"no bible called {name_or_path!r} is registered, and it is not a bible folder")
