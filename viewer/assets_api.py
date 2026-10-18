"""The studio Library: every run under output/ grouped into assets.

An asset is one thing the user is making, followed through its steps:
picture -> 3D model -> finished model -> rigged model -> clips (or, for a prop sheet,
the props cut from it). The steps were separate tabs with separate run folders, so the
links have to be rebuilt from what each run left behind:

* picture -> 3D model: the 3D run's folder name starts with the picture's
  (`<picture>__pixal3d__...` or `<picture>-cutout__...`).
* 3D model -> finished, and 3D model -> prop sheet: the finish or props run keeps a copy
  of its input (`input/source.glb`, `source.glb`); same bytes, same sha256.
* picture -> finished: a finish run also keeps the source picture (`input/source.png`),
  which links it even when its 3D model was made outside the Generate tab.
* finished (or raw) model -> rig: the rig's provenance records its input's sha256.
* any of these -> a Blender version (`blender-edits/versions/<id>/`): the version records
  its parent's sha256; a finish or rig made from the version keeps its sha256 the same
  way, so what is made from a version stays in the asset it came from.

Hashes are cached in `output/.assets-index.json`, keyed by path and invalidated by size
and modification time, so a Library refresh hashes only new or changed files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))  # rig_check, for whether a rig fits the moves

import creature_family_api  # noqa: E402 - the family a rig was given

FINISH_RUN = re.compile(r"^[A-Za-z0-9_-]{1,80}__finish__\d{8}-\d{6}(?:-\d+)?$")
RIG_RUN = re.compile(r"^[A-Za-z0-9_-]{1,80}__rig__\d{8}-\d{6}(?:-\d+)?$")
PROPS_RUN = re.compile(r"^[A-Za-z0-9_-]{1,80}__props__\d{8}-\d{6}(?:-\d+)?$")
# Folders under output/ that hold other steps' runs, not 3D generations.
NOT_GENERATED = {"finish", "props", "images", "rig-rebind", "animate", "blender-edits", "editor-feasibility",
                 "batches"}
# A second take of a preset (`_b`) is a choice between versions, not another move.
TAKE_SUFFIX = re.compile(r"_[b-z]$")
STAGES = ("picture", "model", "finished", "rigged", "animated")
INDEX_NAME = ".assets-index.json"
META_NAME = ".assets-meta.json"   # the user's own choices about assets, e.g. hidden ones
ASSET_ID = re.compile(r"^[0-9a-f]{12}$")


def friendly_name(stem: str) -> str:
    """'a-chunky-knight__20261005-205524-cutout__pixal3d__...' -> 'a chunky knight'."""
    words = stem.split("__")[0].removesuffix("-cutout").replace("-", " ").replace("_", " ").strip()
    words = re.sub(r"\s+\d+k$", "", words)  # '... 40k' is a face count, not a name
    if len(words) > 40:
        words = words[:39].rstrip() + "…"
    return words or stem


class _Hashes:
    """sha256 per file, cached by (size, mtime)."""

    def __init__(self, output: Path):
        self.output = output
        self.path = output / INDEX_NAME
        try:
            self.index = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.index = {}
        self.changed = False

    def of(self, file: Path) -> str:
        rel = file.relative_to(self.output).as_posix()
        stat = file.stat()
        cached = self.index.get(rel)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime:
            return cached["sha256"]
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        self.index[rel] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest.hexdigest()}
        self.changed = True
        return self.index[rel]["sha256"]

    def rig_facts(self, file: Path) -> dict[str, Any]:
        """{"fits_moves": bool | None, "facing": [x, y, z] | None} for a rig (scripts/rig_check.py),
        cached like the hashes. None when it cannot be told (the move list stays, as before)."""
        key = "fits:" + file.relative_to(self.output).as_posix()
        stat = file.stat()
        cached = self.index.get(key)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime and "facing" in cached:
            return {"fits_moves": cached["humanoid"], "facing": cached["facing"]}
        try:
            import rig_check  # scripts/ is on the path (see the top of this module)
            result = rig_check.check(file)
        except Exception:  # noqa: BLE001 - an unreadable rig must not take the Library down
            return {"fits_moves": None, "facing": None}
        humanoid, facing = bool(result["humanoid"]), result.get("forward")
        self.index[key] = {"size": stat.st_size, "mtime": stat.st_mtime, "humanoid": humanoid, "facing": facing}
        self.changed = True
        return {"fits_moves": humanoid, "facing": facing}

    def save(self) -> None:
        if not self.changed:
            return
        try:
            self.path.write_text(json.dumps(self.index, indent=0, sort_keys=True), encoding="utf-8")
        except OSError:
            pass  # a read-only output folder still gets a Library, just a slower one


def _dirs(path: Path):
    return sorted(d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")) if path.is_dir() else []


def _pictures(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "images"):
        png = run / f"{run.name}.png"
        if png.is_file():
            found.append({"stem": run.name, "file": png})
    return found


def _models(output: Path) -> list[dict[str, Any]]:
    """GLBs the Generate tab made: `<run>/<run>.glb`, maybe one licence-class folder down."""
    found = []
    for top in _dirs(output):
        if top.name in NOT_GENERATED:
            continue
        for run in (top, *_dirs(top)):
            glb = run / f"{run.name}.glb"
            if glb.is_file():
                found.append({"stem": run.name, "file": glb, "record": run / f"{run.name}.json"})
    return found


def _finishes(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "finish"):
        if not FINISH_RUN.fullmatch(run.name):
            continue
        results = sorted(run.glob("*.glb"))
        if results:
            source, image = run / "input" / "source.glb", run / "input" / "source.png"
            found.append({"stem": run.name, "file": results[0], "source": source if source.is_file() else None,
                          "image": image if image.is_file() else None, "settings": run / "input" / "settings.json"})
    return found


def _rigs(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "animate"):
        if not RIG_RUN.fullmatch(run.name):
            continue
        for rigged in sorted(run.glob("*_rigged.glb")):
            base = rigged.name.removesuffix("_rigged.glb")
            try:
                record = json.loads(rigged.with_suffix(".provenance.json").read_text(encoding="utf-8"))
                input_sha = (record.get("input") or {}).get("sha256")
            except (OSError, ValueError, AttributeError):
                input_sha = None
            clips = []
            for clip in sorted(run.glob(f"{base}_*.glb")):
                name = clip.stem.removeprefix(f"{base}_")
                if clip == rigged or name == "rigged" or TAKE_SUFFIX.search(name):
                    continue
                clips.append({"name": name, "file": clip})
            found.append({"stem": base, "file": rigged, "input_sha": input_sha, "clips": clips})
    return found


def _prop_sets(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "props"):
        if not PROPS_RUN.fullmatch(run.name):
            continue
        props = []
        for prop in _dirs(run / "finished"):
            lod0 = prop / f"{prop.name}_LOD0.glb"
            if lod0.is_file():
                props.append({"name": prop.name, "file": lod0})
        source = run / "source.glb"
        found.append({"stem": run.name, "file": source if source.is_file() else None, "props": props,
                      "dir": run})
    return found


def _versions(output: Path) -> list[dict[str, Any]]:
    """Blender versions, for the lineage only: the Library lists them under their asset."""
    from blender_versions import list_versions

    return [{"stem": Path(v["file"]).parent.name, "file": output / v["file"], "sha": v["sha256"],
             "parent_sha": v.get("parent_sha256")} for v in list_versions(output)]


def list_assets(output: Path) -> list[dict[str, Any]]:
    """Every asset on disk, the one touched most recently first."""
    if not output.is_dir():
        return []
    hashes = _Hashes(output)
    runs = _runs(output)
    nodes = _nodes(runs)
    groups = _groups(runs, nodes, hashes)
    hashes.save()
    assets = [asset for members in groups if (asset := _asset(output, hashes, nodes, members)) is not None]
    hashes.save()
    _with_versions_names_and_hidden(output, assets)
    assets.sort(key=lambda a: a["updated"], reverse=True)
    return assets


def _runs(output: Path) -> dict[str, list[dict[str, Any]]]:
    """Every run on disk by what it made, a batch's finished rows among them."""
    pictures, models, finishes = _pictures(output), _models(output), _finishes(output)
    rigs, sets, versions = _rigs(output), _prop_sets(output), _versions(output)
    from batch_api import library_nodes  # a batch's finished rows are assets too (docs/batches.md)

    batched = library_nodes(output)
    # a batch that only checked or exported a GLB the Library already shows adds no
    # second node for it: that one would take over the original's rigs and props
    known = {model["file"].resolve(): model for model in models}
    for model in batched["model"]:
        kept = known.get(model["file"].resolve())
        if kept is not None:   # its picture and finish join the kept node instead
            kept.setdefault("links", []).append(model["link"])
    batched["model"] = [model for model in batched["model"] if model["file"].resolve() not in known]
    pictures, models = pictures + batched["picture"], models + batched["model"]
    finishes = finishes + batched["finished"]
    return {"picture": pictures, "model": models, "finished": finishes, "rigged": rigs, "set": sets,
            "version": versions}


def _nodes(runs: dict[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Every run as (what it made, the run), in the order the assets are built from."""
    return [(kind, node) for kind in ("picture", "model", "finished", "rigged", "set", "version")
            for node in runs[kind]]


class _Joined:
    """Union-find over the runs: each connected group is one asset."""

    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, i: int) -> int:
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _groups(runs: dict[str, list[dict[str, Any]]], nodes: list[tuple[str, dict[str, Any]]],
            hashes: _Hashes) -> list[list[int]]:
    """The runs joined into assets, as lists of indices into `nodes`, by the links the module
    docstring names; each asset's runs in the order of `nodes`, the assets in the order of
    their first run."""
    joined = _Joined(len(nodes))
    index_of = {id(n): i for i, (_, n) in enumerate(nodes)}
    found = _by_sha(runs, hashes, index_of, joined)
    _join_rigs_versions_and_sets(runs, hashes, index_of, joined, found)
    linked: dict[str, int] = {}   # nodes that name their asset outright (batch rows)
    for i, (_, node) in enumerate(nodes):
        for link in [node.get("link"), *node.get("links", [])]:
            if link is not None:
                joined.join(i, linked.setdefault(link, i))
    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(joined.find(i), []).append(i)
    return list(groups.values())


def _by_sha(runs: dict[str, list[dict[str, Any]]], hashes: _Hashes, index_of: dict[int, int],
            joined: _Joined) -> dict[str, dict[str, int]]:
    """Pictures with the models named after them, and finishes with the picture and model they
    kept a copy of; returns the models, pictures, finishes and versions by sha256."""
    pictures, models, finishes, versions = runs["picture"], runs["model"], runs["finished"], runs["version"]
    model_by_sha: dict[str, int] = {}
    for model in models:
        sha = hashes.of(model["file"])
        if model.get("link") is None or sha not in model_by_sha:   # a batch copy never
            model_by_sha[sha] = index_of[id(model)]                # takes over a hash
    for picture in pictures:
        for model in models:
            if model["stem"].startswith(picture["stem"] + "__") or model["stem"].startswith(picture["stem"] + "-cutout__"):
                joined.join(index_of[id(picture)], index_of[id(model)])
    picture_by_sha: dict[str, int] = {}
    for picture in pictures:   # as for models: a batch copy never takes over a hash
        sha = hashes.of(picture["file"])
        if picture.get("link") is None or sha not in picture_by_sha:
            picture_by_sha[sha] = index_of[id(picture)]
    finished_by_sha: dict[str, int] = {}
    for fin in finishes:
        if fin["image"] is not None:
            match = picture_by_sha.get(hashes.of(fin["image"]))
            if match is not None:
                joined.join(index_of[id(fin)], match)
        sha = hashes.of(fin["file"])
        if fin.get("link") is None or sha not in finished_by_sha:
            finished_by_sha[sha] = index_of[id(fin)]
        if fin["source"] is not None:
            match = model_by_sha.get(hashes.of(fin["source"]))
            if match is not None:
                joined.join(index_of[id(fin)], match)
    version_by_sha: dict[str, int] = {}
    for version in versions:   # the same bytes exported twice are one file to follow
        joined.join(index_of[id(version)], version_by_sha.setdefault(version["sha"], index_of[id(version)]))
    for fin in finishes:   # a finish made from a Blender version
        if fin["source"] is not None and (match := version_by_sha.get(hashes.of(fin["source"]))) is not None:
            joined.join(index_of[id(fin)], match)
    return {"model": model_by_sha, "picture": picture_by_sha, "finished": finished_by_sha, "version": version_by_sha}


def _join_rigs_versions_and_sets(runs: dict[str, list[dict[str, Any]]], hashes: _Hashes, index_of: dict[int, int],
                                 joined: _Joined, found: dict[str, dict[str, int]]) -> None:
    """Rigs with the model they were made from, versions with their parent file, prop sets
    with their source model."""
    model_by_sha, finished_by_sha, version_by_sha = found["model"], found["finished"], found["version"]
    rigs = runs["rigged"]
    for rig in rigs:
        sha = rig["input_sha"]
        match = finished_by_sha.get(sha) if sha else None
        if match is None and sha:
            match = model_by_sha.get(sha)
        if match is None and sha:
            match = version_by_sha.get(sha)
        if match is not None:
            joined.join(index_of[id(rig)], match)
    rig_by_sha = {hashes.of(rig["file"]): index_of[id(rig)] for rig in rigs}
    for rig in rigs:   # a version of one of the rig's moves comes from that rig too
        for clip in rig["clips"]:
            rig_by_sha.setdefault(hashes.of(clip["file"]), index_of[id(rig)])
    for version in runs["version"]:   # a version joins the file it was made from, a version too
        parent_sha = version["parent_sha"]
        match = next((known[parent_sha] for known in (model_by_sha, finished_by_sha, rig_by_sha, version_by_sha)
                      if parent_sha in known), None)
        if match is not None:
            joined.join(index_of[id(version)], match)
    for prop_set in runs["set"]:
        if prop_set["file"] is not None:
            match = model_by_sha.get(hashes.of(prop_set["file"]))
            if match is not None:
                joined.join(index_of[id(prop_set)], match)


def _rel(output: Path, path: Path | None) -> str | None:
    return path.relative_to(output).as_posix() if path is not None else None


def _latest(by_kind: dict[str, list[dict[str, Any]]], kind: str) -> dict[str, Any] | None:
    items = [n for n in by_kind.get(kind, []) if n.get("file") is not None]
    return max(items, key=lambda n: n["file"].stat().st_mtime) if items else None


def _oldest(by_kind: dict[str, list[dict[str, Any]]], *kinds: str) -> dict[str, Any] | None:
    items = [n for kind in kinds for n in by_kind.get(kind, []) if n.get("file") is not None]
    return min(items, key=lambda n: n["file"].stat().st_mtime) if items else None


def _asset(output: Path, hashes: _Hashes, nodes: list[tuple[str, dict[str, Any]]],
           members: list[int]) -> dict[str, Any] | None:
    """One asset from its runs: the newest of each step, its kind, stage, id and name. None
    for versions alone: attach_versions lists them under their asset."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for i in members:
        by_kind.setdefault(nodes[i][0], []).append(nodes[i][1])
    if set(by_kind) == {"version"}:
        return None

    picture, model, fin, rig = (_latest(by_kind, "picture"), _latest(by_kind, "model"), _latest(by_kind, "finished"),
                                _latest(by_kind, "rigged"))
    prop_set = max(by_kind.get("set", []), key=lambda n: n["dir"].stat().st_mtime, default=None)
    files = [n["file"] for i in members for n in [nodes[i][1]] if n.get("file") is not None]
    files += [c["file"] for r in by_kind.get("rigged", []) for c in r["clips"]]
    files += [p["file"] for s in by_kind.get("set", []) for p in s["props"]]
    clips = rig["clips"] if rig else []

    if prop_set is not None:
        kind, stage = "prop set", "finished" if prop_set["props"] else "model"
    elif rig is not None:
        kind, stage = "character", "animated" if clips else "rigged"
    else:
        kind = "picture" if model is None and fin is None else "model"
        stage = "finished" if fin else "model" if model else "picture"

    # the id is the first node of the chain, which a later finish or rig (from a Blender
    # version, say) never replaces: hidden flags, names and versions keep their asset
    first = (picture or model or (prop_set if prop_set and prop_set.get("file") else None)
             or _oldest(by_kind, "finished", "rigged"))
    # A person-given name ("rune_knight" at the finish or rig step) beats the prompt the
    # picture was named after; it is the shortest name in the chain.
    stems = [n["stem"] for n in (rig, fin, prop_set, model, picture) if n is not None]
    name_from = min(stems, key=lambda stem: len(friendly_name(stem)))
    return {
        "id": (hashlib.sha1(_rel(output, first["file"]).encode()).hexdigest()[:12] if first and first.get("file")
               else name_from),
        "name": friendly_name(name_from),
        "kind": kind,
        "stage": stage,
        "picture": _rel(output, picture["file"]) if picture else None,
        "model": _rel(output, model["file"]) if model else None,
        "finished": _rel(output, fin["file"]) if fin else None,
        "rigged": _rel(output, rig["file"]) if rig else None,
        **(hashes.rig_facts(rig["file"]) if rig else {"fits_moves": None, "facing": None}),
        **creature_family_api.facts(rig["file"] if rig else None),
        "clips": [{"name": c["name"], "file": _rel(output, c["file"])} for c in clips],
        "props": [{"name": p["name"], "file": _rel(output, p["file"])} for p in (prop_set["props"] if prop_set else [])],
        "updated": max((f.stat().st_mtime for f in files), default=0),
        "made": made_with(model, fin),
        "stale": stale_steps(hashes, model, fin, rig),
    }


def _with_versions_names_and_hidden(output: Path, assets: list[dict[str, Any]]) -> None:
    """Each asset's Blender versions, and the person's own name and hidden flag for it."""
    from blender_versions import attach_versions

    attach_versions(output, assets)
    meta = _read_meta(output)
    hidden = set(meta.get("hidden", []))
    names = meta.get("names") if isinstance(meta.get("names"), dict) else {}
    for asset in assets:
        asset["hidden"] = asset["id"] in hidden
        if isinstance(names.get(asset["id"]), str):
            asset["name"] = names[asset["id"]]


def _json(path: Path | None) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path is not None else {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def made_with(model: dict[str, Any] | None, fin: dict[str, Any] | None) -> dict[str, Any]:
    """What the current 3D model and finish were made with, for the Steps panel's Redo to
    start from: the engine and seed in the 3D run's record, the finish's own settings."""
    made: dict[str, Any] = {}
    record = _json(model.get("record")) if model else {}
    if record:
        seed = (record.get("parameters") or (record.get("model") or {}).get("parameters") or {}).get("seed")
        engine = record.get("backend") or (record.get("model") or {}).get("backend")
        made["model"] = {"seed": seed if isinstance(seed, int) else None, "engine": engine}
    settings = _json(fin.get("settings")) if fin else {}
    if settings:
        made["finished"] = {"faces": settings.get("faces"), "texture_size": settings.get("texture_size"),
                            "photo": not settings.get("skip_photo", False)}
    return made


def stale_steps(hashes: _Hashes, model: dict[str, Any] | None, fin: dict[str, Any] | None,
                rig: dict[str, Any] | None) -> dict[str, bool]:
    """Which later steps were made from an earlier version of the step before them: a
    finish of another 3D model than the current one, a rig of another finish."""
    finished = bool(fin and model and fin.get("source") is not None
                    and hashes.of(fin["source"]) != hashes.of(model["file"]))
    base = fin or model
    rigged = bool(rig and base and rig.get("input_sha") and rig["input_sha"] != hashes.of(base["file"]))
    return {"finished": finished, "rigged": rigged}


def _read_meta(output: Path) -> dict[str, Any]:
    try:
        meta = json.loads((output / META_NAME).read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


# One change at a time: the threaded server can take a rename and a hide together, and
# each would otherwise write over the other's read.
_META_LOCK = threading.Lock()


def _write_meta(output: Path, meta: dict[str, Any]) -> None:
    """Replace the file whole (written aside, then os.replace), so a crash mid-write leaves
    the old file, not a truncated one. A file that does not parse is moved aside as
    `.bad-<time>` first: it may hold names a person chose, and starting over must not
    erase them for good."""
    path = output / META_NAME
    if path.is_file():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            path.replace(path.with_name(f"{META_NAME}.bad-{time.strftime('%Y%m%d-%H%M%S')}"))
    aside = path.with_name(f"{META_NAME}.{os.getpid()}.tmp")
    aside.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(aside, path)


def set_hidden(output: Path, asset_id: str, hidden: bool) -> None:
    """Hide an asset from the Library, or bring it back. Never touches its files."""
    if not ASSET_ID.fullmatch(asset_id or ""):
        raise ValueError(f"not an asset id: {asset_id!r}")
    if asset_id not in {a["id"] for a in list_assets(output)}:
        raise ValueError(f"no such asset: {asset_id}")
    with _META_LOCK:
        meta = _read_meta(output)
        ids = set(meta.get("hidden", []))
        ids.add(asset_id) if hidden else ids.discard(asset_id)
        meta["hidden"] = sorted(ids)
        _write_meta(output, meta)


NAME_LIMIT = 80


def set_name(output: Path, asset_id: str, name: str) -> str:
    """Give an asset the name a person chose; an empty name goes back to the one made from its
    prompt. Only the Library's label changes: no file is renamed or moved. Returns the name."""
    if not ASSET_ID.fullmatch(asset_id or ""):
        raise ValueError(f"not an asset id: {asset_id!r}")
    found = {a["id"]: a for a in list_assets(output)}
    if asset_id not in found:
        raise ValueError(f"no such asset: {asset_id}")
    clean = " ".join(str(name or "").split())[:NAME_LIMIT]
    with _META_LOCK:
        meta = _read_meta(output)
        names = meta.get("names") if isinstance(meta.get("names"), dict) else {}
        if clean:
            names[asset_id] = clean
        else:
            names.pop(asset_id, None)
        meta["names"] = names
        _write_meta(output, meta)
    return clean or next(a["name"] for a in list_assets(output) if a["id"] == asset_id)


def assets_payload(output: Path) -> dict[str, Any]:
    """What `GET /api/assets` sends: the list, plus where its relative paths are served."""
    return {"base": "/output/", "stages": list(STAGES), "assets": list_assets(output)}


THUMBS_DIR = ".thumbs"


def thumbnail(output: Path, relative: str, size: int = 128) -> Path:
    """A small copy of a picture in output/, for the Library list, made once and kept in
    output/.thumbs (a new one when the picture changes). Raises ValueError for paths outside output/."""
    from PIL import Image

    source = (output / relative).resolve()
    if output.resolve() not in source.parents or not source.is_file():
        raise ValueError(f"not a picture in output/: {relative}")
    stat = source.stat()
    key = hashlib.sha1(f"{relative}|{stat.st_size}|{stat.st_mtime}|{size}".encode()).hexdigest()[:20]
    target = output / THUMBS_DIR / f"{key}.png"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.thumbnail((size, size))
            partial = target.with_suffix(".part")
            image.save(partial, format="PNG", optimize=True)
            partial.replace(target)  # never leave half a thumbnail behind for the next request
    return target
