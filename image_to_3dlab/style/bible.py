"""A style bible: one folder holding a project's approved and rejected pictures, with notes.

    my-game-style/
      bible.json
      images/<sha8>-<name>.png       approved, each with its .provenance.json when it had one
                                     (its hash is recorded, so a lost licence record is reported)
      rejected/<sha8>-<name>.png     sent back: what the style is NOT
      references/<sha8>-<name>.png   style references that are not ours (an original game's
                                     art): they define the style, are marked "internal" and
                                     must never be distributed or shipped
      .cache/                        regenerable (embeddings, judgements); ignore it in git

The folder is the unit: copy it, keep it in the game's repository, open it on another
machine. Pictures are copied in, never referenced, so nothing breaks when `output/` is
cleaned. Every function takes the folder and re-reads `bible.json`, so the studio and a
terminal can both change one bible without overwriting each other's last edit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

SCHEMA_VERSION = 1
FILE = "bible.json"
KINDS = ("character", "creature", "prop", "equipment_part", "map_object")
LISTS = {"images": "images", "rejected": "rejected", "references": "references"}  # list -> folder
OPTIONAL_LISTS = {"references"}  # added after schema 1 shipped; absent means empty
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
SHA256 = re.compile(r"[0-9a-f]{64}")
# Where the vision-language judge starts; each project rewrites them in its own words.
DEFAULT_CRITERIA = [
    "Is it painted in the same way as the reference pictures, not photographic?",
    "Are the colours as saturated and as bright as in the reference pictures?",
    "Are the shapes and proportions stylised the way they are in the reference pictures?",
]

_LOCK = threading.Lock()
# Across processes (the studio and a terminal), a lock file beside bible.json. A lock older
# than STALE_LOCK_SECONDS was left by a crash: no edit takes that long.
LOCK_FILE = ".bible.lock"
LOCK_WAIT_SECONDS = 10.0
STALE_LOCK_SECONDS = 120.0


class BibleError(ValueError):
    """Something a person can fix: a bad name, a missing note, an unknown picture."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 - 3.10


def _write(root: Path, data: dict) -> None:
    """Replace bible.json in one step: a crash leaves the old file or the new one, never half."""
    handle, temp = tempfile.mkstemp(prefix=".bible-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
            out.write("\n")
        os.replace(temp, root / FILE)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


@contextmanager
def _locked(root: Path):
    """Hold the bible for one read-modify-write, in this process and against others."""
    if not (Path(root) / FILE).is_file():
        load(root)  # says there is no bible here, rather than failing on the lock file
    with file_lock(Path(root) / LOCK_FILE, f"the bible in {root}"):
        yield


@contextmanager
def file_lock(lock: Path, what: str):
    """An exclusive lock file shared by every process; one older than STALE_LOCK_SECONDS is taken over."""
    with _LOCK:
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(handle, str(os.getpid()).encode())
                os.close(handle)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > STALE_LOCK_SECONDS:
                        lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise BibleError(f"{what} is busy (another edit holds {lock.name}); try again") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)


def create(root: Path, name: str, description: str = "", wording: str = "") -> Path:
    """Make a new, empty bible in `root` (created if needed); return the folder."""
    root = Path(root)
    if not NAME.fullmatch(name or ""):
        raise BibleError(f"a bible name is lower-case letters, digits, '-' or '_', got {name!r}")
    if (root / FILE).exists():
        raise BibleError(f"{root} already holds a bible")
    for folder in LISTS.values():
        (root / folder).mkdir(parents=True, exist_ok=True)
    _write(root, {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": description,
        "style": {"wording": wording, "criteria": list(DEFAULT_CRITERIA)},
        "kinds": {},
        "images": [],
        "rejected": [],
        "references": [],
        "score": {"method": None, "warn": None, "fail": None},
    })
    return root


def _inside(folder: Path, relative: str) -> bool:
    """Whether `relative` (from the bible root) lands inside `folder` once `..` is resolved."""
    path = (folder.parent / relative).resolve()
    return path.is_relative_to(folder.resolve()) and path != folder.resolve()


def load(root: Path) -> dict[str, Any]:
    """Read and check `bible.json`. Files on disk are checked separately, by `problems`."""
    root = Path(root)
    path = root / FILE
    if not path.is_file():
        raise BibleError(f"no bible in {root} (no {FILE})")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BibleError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise BibleError(f"{path}: unsupported schema_version {data.get('schema_version') if isinstance(data, dict) else None!r}")
    if not NAME.fullmatch(str(data.get("name", ""))):
        raise BibleError(f"{path}: missing or invalid name")
    style = data.get("style")
    if not isinstance(style, dict) or not isinstance(style.get("wording", ""), str):
        raise BibleError(f"{path}: 'style' must be an object with a 'wording' text")
    criteria = style.get("criteria", [])
    if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
        raise BibleError(f"{path}: 'style.criteria' must be a list of questions")
    kinds = data.get("kinds", {})
    if not isinstance(kinds, dict):
        raise BibleError(f"{path}: 'kinds' must be an object")
    for kind, value in kinds.items():
        if kind not in KINDS:
            raise BibleError(f"{path}: unknown kind {kind!r} in 'kinds'; choose from {', '.join(KINDS)}")
        if not isinstance(value, dict) or not isinstance(value.get("wording", ""), str):
            raise BibleError(f"{path}: 'kinds.{kind}' must be an object with a 'wording' text")
    for key in OPTIONAL_LISTS:
        data.setdefault(key, [])
    for key, folder in LISTS.items():
        entries = data.get(key)
        if not isinstance(entries, list):
            raise BibleError(f"{path}: '{key}' must be a list")
        for entry in entries:
            _check_entry(root, path, folder, entry)
    return data


def _check_entry(root: Path, path: Path, folder: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise BibleError(f"{path}: every picture must be an object")
    file = entry.get("file")
    if not isinstance(file, str) or not file.startswith(f"{folder}/") or not _inside(root / folder, file):
        raise BibleError(f"{path}: {file!r} must be a file inside the bible's {folder}/ folder")
    if not SHA256.fullmatch(str(entry.get("sha256", ""))):
        raise BibleError(f"{path}: {file} has no valid sha256")
    if not str(entry.get("note", "")).strip():
        raise BibleError(f"{path}: {file} has no note")
    if "provenance_sha256" in entry and not SHA256.fullmatch(str(entry["provenance_sha256"])):
        raise BibleError(f"{path}: {file} has an invalid provenance_sha256")
    if entry.get("kind") is not None and entry["kind"] not in KINDS:
        raise BibleError(f"{path}: {file} has unknown kind {entry['kind']!r}")


def problems(root: Path) -> list[str]:
    """Pictures listed in bible.json whose file is missing or no longer matches its hash."""
    root = Path(root)
    found = []
    for key in LISTS:
        for entry in load(root)[key]:
            target = root / entry["file"]
            if not target.is_file():
                found.append(f"{entry['file']}: missing")
            elif sha256_file(target) != entry["sha256"]:
                found.append(f"{entry['file']}: changed since it was added")
            if "provenance_sha256" in entry:
                sidecar = target.with_suffix(".provenance.json")
                if not sidecar.is_file():
                    found.append(f"{entry['file']}: its provenance record is missing")
                elif sha256_file(sidecar) != entry["provenance_sha256"]:
                    found.append(f"{entry['file']}: its provenance record changed since it was added")
    return found


def _stem(name: str) -> str:
    stem = re.sub(r"[^a-z0-9_-]+", "-", Path(name).stem.lower()).strip("-")
    return stem[:48] or "picture"


def _check_picture(image: Path) -> None:
    if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
        raise BibleError(f"{image} is not a picture (PNG, JPG or WebP)")
    try:
        with Image.open(image) as opened:
            opened.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise BibleError(f"{image} is not a readable picture: {exc}") from exc


def add(root: Path, image: Path, note: str, kind: str | None = None, rejected: bool = False,
        source_label: str | None = None, reference: bool = False) -> dict[str, Any]:
    """Copy `image` (and its `.provenance.json`) into the bible with a note; return the entry.

    `rejected=True` files it under what the style is not; `reference=True` under style
    references that are not ours, marked `"distribution": "internal"`. The same picture
    cannot be in the bible twice, in any list.
    """
    if rejected and reference:
        raise BibleError("a picture is either rejected or a style reference, not both")
    root, image = Path(root), Path(image)
    note = (note or "").strip()
    if not note:
        raise BibleError("every picture needs a note saying why it is (or is not) the style")
    if kind is not None and kind not in KINDS:
        raise BibleError(f"unknown kind {kind!r}; choose one of {', '.join(KINDS)}")
    _check_picture(image)
    digest = sha256_file(image)
    key = "rejected" if rejected else "references" if reference else "images"
    with _locked(root):
        data = load(root)
        for other in LISTS:
            if any(e["sha256"] == digest for e in data[other]):
                raise BibleError(f"this picture is already in the bible ({other})")
        relative = f"{LISTS[key]}/{digest[:8]}-{_stem(image.name)}{image.suffix.lower()}"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, target)
        entry: dict[str, Any] = {"file": relative, "sha256": digest, "kind": kind, "note": note,
                                 "added": _now()}
        sidecar = image.with_suffix(".provenance.json")
        if sidecar.is_file():
            copied = target.with_suffix(".provenance.json")
            shutil.copyfile(sidecar, copied)
            # Recorded, so a licence record that goes missing later is reported, not silent.
            entry["provenance_sha256"] = sha256_file(copied)
        if source_label:
            entry["source"] = source_label
        if reference:
            entry["distribution"] = "internal"
        data[key].append(entry)
        _write(root, data)
    return entry


def _find(data: dict, selector: str) -> tuple[str, int]:
    """The list and index of the one picture whose hash starts with `selector`."""
    selector = (selector or "").strip().lower()
    if len(selector) < 4:
        raise BibleError("name a picture by at least the first 4 characters of its hash")
    hits = [(key, i) for key in LISTS for i, e in enumerate(data[key]) if e["sha256"].startswith(selector)]
    if not hits:
        raise BibleError(f"no picture in the bible has a hash starting {selector!r}")
    if len(hits) > 1:
        raise BibleError(f"{selector!r} matches {len(hits)} pictures; give more of the hash")
    return hits[0]


def remove(root: Path, selector: str) -> dict[str, Any]:
    """Take a picture out: its entry, its file and its provenance copy. Returns the entry."""
    root = Path(root)
    with _locked(root):
        data = load(root)
        key, index = _find(data, selector)
        entry = data[key].pop(index)
        _write(root, data)
        # Inside the lock: a re-add of the same picture must not have its new copy deleted.
        target = root / entry["file"]
        target.unlink(missing_ok=True)
        target.with_suffix(".provenance.json").unlink(missing_ok=True)
    return entry


def set_note(root: Path, selector: str, note: str) -> dict[str, Any]:
    """Change one picture's note. Returns the updated entry."""
    note = (note or "").strip()
    if not note:
        raise BibleError("a note cannot be empty")
    root = Path(root)
    with _locked(root):
        data = load(root)
        key, index = _find(data, selector)
        data[key][index]["note"] = note
        _write(root, data)
        return dict(data[key][index])
