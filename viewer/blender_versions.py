"""Immutable snapshots and explicit Library lineage for manual Blender edits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath

from image_to_3dlab.blender_exchange import MAX_BYTES, inspect_glb

ID = re.compile(r"^[0-9a-f]{32}$")
LOCK = threading.RLock()
DIRECTORY = "blender-edits"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(output, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or relative.startswith("/") or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("Use a relative Library file path")
    path = output / PurePosixPath(relative)
    # Reject links even inside output: work folders must not alias source folders.
    current = output
    for piece in PurePosixPath(relative).parts:
        current /= piece
        if current.is_symlink() or (current.exists() and current.resolve() != current.absolute()):
            raise ValueError("Linked paths are not supported for Blender edits")
    if not path.resolve().is_relative_to(output.resolve()):
        raise ValueError("Path is outside output")
    return path


def session_path(output, session_id):
    if not isinstance(session_id, str) or not ID.fullmatch(session_id):
        raise ValueError("Invalid edit id")
    return safe_path(output, f"{DIRECTORY}/sessions/{session_id}")


def read_json(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Metadata exceeds 16 MiB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except RecursionError as exc:
        raise ValueError("Metadata is nested too deeply") from exc
    if not isinstance(value, dict):
        raise ValueError("Invalid metadata")  # noqa: TRY004 - invalid serialized data, not a caller type error
    return value


def write_json(path, data):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_glb(path):
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Choose a GLB file up to 512 MiB")
    with path.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("GLB exceeds 512 MiB")
    return data


def asset_files(asset):
    return {path for path in [asset.get(key) for key in ("model", "finished", "rigged")]
            + [v.get("file") for key in ("clips", "props", "blender_versions") for v in asset.get(key, [])]
            if path}


def read_session(output, session_id):
    path = session_path(output, session_id)
    session = read_json(path / "session.json")
    if session.get("id") != session_id or session.get("schema_version") != 1:
        raise ValueError("Invalid edit metadata")
    datetime.fromisoformat(session["created_at"])
    if (not re.fullmatch(r"[0-9a-f]{12}", str(session.get("asset_id", "")))
            or not isinstance(session.get("asset_name"), str)
            or not isinstance(session.get("facts"), dict)
            or not isinstance(session.get("sidecars"), list)
            or not re.fullmatch(r"[0-9a-f]{64}", str(session.get("parent_sha256", "")))):
        raise ValueError("Invalid edit metadata")
    safe_path(output, session["parent_file"])
    return session


def list_sessions(output, asset_id):
    root = safe_path(output, f"{DIRECTORY}/sessions")
    results = []
    if root.is_dir():
        for directory in root.iterdir():
            if not ID.fullmatch(directory.name):
                continue
            try:
                session = read_session(output, directory.name)
                if session["asset_id"] == asset_id and not (output / DIRECTORY / "versions" / session["id"] / "version.json").exists():
                    results.append(session)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(results, key=lambda s: s["created_at"], reverse=True)


def create_session(output, asset_id, relative, request_id):
    from assets_api import list_assets

    with LOCK:
        source = safe_path(output, relative)
        target = session_path(output, request_id)
        if (target / "session.json").is_file():
            old = read_session(output, request_id)
            if old["asset_id"] != asset_id or old["parent_file"] != relative:
                raise ValueError("Edit id belongs to a different source")
            return old
        asset = next((a for a in list_assets(output) if a["id"] == asset_id), None)
        if asset is None or relative not in asset_files(asset):
            raise ValueError("This model does not belong to the selected Library asset")
        if source.suffix.lower() != ".glb":
            raise ValueError("Choose a GLB model")
        data = read_glb(source)
        facts = inspect_glb(data)
        parent = next((v for v in asset.get("blender_versions", []) if v["file"] == relative), None)
        if parent and sha(data) != parent["sha256"]:
            raise ValueError("This saved version changed on disk; restore it before starting another edit")
        provenance = []
        for suffix in (".provenance.json", ".json", ".retopo-repaint.json"):
            sidecar = safe_path(output, source.with_suffix(suffix).relative_to(output).as_posix())
            if sidecar.is_file():
                record = read_json(sidecar)  # bad provenance must not silently disappear
                provenance.append((suffix, sidecar.read_bytes(), record))
        source_record = next((r for _, _, r in provenance if r.get("license")), provenance[0][2] if provenance else None)
        session = {"schema_version": 1, "id": request_id, "asset_id": asset_id, "asset_name": asset["name"],
                   "parent_file": relative, "parent_sha256": sha(data), "parent_version": parent["id"] if parent else None,
                   "created_at": now(), "facts": facts, "source_record": source_record,
                   "sidecars": [{"suffix": s, "sha256": sha(b)} for s, b, _ in provenance]}
        staging = safe_path(output, f"{DIRECTORY}/sessions/.pending-{uuid.uuid4().hex}")
        (staging / "input").mkdir(parents=True)
        (staging / "work").mkdir()
        (staging / "input/source.glb").write_bytes(data)
        for suffix, raw, _ in provenance:
            (staging / "input" / f"source{suffix}").write_bytes(raw)
        write_json(staging / "session.json", session)
        staging.rename(target)
        return session


def checked_snapshot(output, session):
    root = session_path(output, session["id"])
    path = safe_path(output, (root / "input/source.glb").relative_to(output).as_posix())
    if sha(read_glb(path)) != session["parent_sha256"]:
        raise ValueError("The source snapshot has changed; start a new Blender edit")
    for sidecar in session.get("sidecars", []):
        if sidecar["suffix"] not in (".provenance.json", ".json", ".retopo-repaint.json"):
            raise ValueError("Invalid snapshot sidecar")
        item = safe_path(output, (root / "input" / f"source{sidecar['suffix']}").relative_to(output).as_posix())
        if sha(item.read_bytes()) != sidecar["sha256"]:
            raise ValueError("Source provenance snapshot has changed")
    return path


def list_versions(output):
    root = safe_path(output, f"{DIRECTORY}/versions")
    results = []
    if root.is_dir():
        for directory in root.iterdir():
            if not ID.fullmatch(directory.name):
                continue
            try:
                record = read_json(safe_path(output, f"{DIRECTORY}/versions/{directory.name}/version.json"))
                if (not re.fullmatch(r"[0-9a-f]{12}", str(record.get("asset_id", "")))
                        or not isinstance(record.get("asset_name"), str) or not isinstance(record.get("label"), str)
                        or not isinstance(record.get("warning_codes"), list) or not isinstance(record.get("facts"), dict)):
                    continue
                if any(type(record["facts"].get(key)) is not int for key in ("skins", "joints", "meshes")):
                    continue
                if not isinstance(record["facts"].get("animations"), list):
                    continue
                datetime.fromisoformat(record["created_at"]).timestamp()
                safe_path(output, record["parent_file"])
                expected = f"{DIRECTORY}/versions/{directory.name}/model.glb"
                if (record.get("metadata") != f"{DIRECTORY}/versions/{directory.name}/version.json"
                        or record.get("provenance") != f"{DIRECTORY}/versions/{directory.name}/model.provenance.json"
                        or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
                    continue
                if (record.get("id") == directory.name and record.get("file") == expected
                        and record.get("schema_version") == 1 and safe_path(output, expected).is_file()):
                    results.append(record)
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                continue
    return sorted(results, key=lambda r: (r["created_at"], r["id"]))


def attach_versions(output, assets):
    by_id = {a["id"]: a for a in assets}
    for version in list_versions(output):
        asset = by_id.get(version["asset_id"])
        if asset is None:
            # Imported versions remain findable if the original is later removed.
            asset = {"id": version["asset_id"], "name": version["asset_name"], "kind": "model", "stage": "model",
                     "picture": None, "model": None, "finished": None, "rigged": None, "fits_moves": None,
                     "facing": None, "clips": [], "props": [], "updated": 0}
            assets.append(asset)
            by_id[asset["id"]] = asset
        asset.setdefault("blender_versions", []).append(version)
        asset["updated"] = max(asset["updated"], datetime.fromisoformat(version["created_at"]).timestamp())


def import_version(output, session_id, data, filename):
    with LOCK:
        session = read_session(output, session_id)
        target = safe_path(output, f"{DIRECTORY}/versions/{session_id}")
        if (target / "version.json").is_file():
            previous = read_json(target / "version.json")
            if previous["sha256"] != sha(data):
                raise ValueError("This edit already imported a different file; start a new edit")
            if sha(read_glb(safe_path(output, f"{DIRECTORY}/versions/{session_id}/model.glb"))) != previous["sha256"]:
                raise ValueError("This saved version changed on disk; restore it before retrying")
            return previous
        checked_snapshot(output, session)
        if not isinstance(filename, str) or len(filename) > 240 or not filename.lower().endswith(".glb"):
            raise ValueError("Export glTF Binary (.glb), then choose that file")
        facts = inspect_glb(data)
        codes = []
        old = session["facts"]
        if old["skins"] and not facts["skins"]:
            codes.append("rig_removed")
        elif old["skins"] and old["rig_signature"] != facts["rig_signature"]:
            codes.append("rig_changed")
        if len(facts["animations"]) < len(old["animations"]):
            codes.append("animations_removed")
        if facts["images"] < old["images"]:
            codes.append("textures_removed")
        try:
            current = read_glb(safe_path(output, session["parent_file"]))
            if sha(current) != session["parent_sha256"]:
                codes.append("source_changed")
        except (OSError, ValueError):
            codes.append("source_changed")
        source = session["source_record"] or {}
        source_sidecars = []
        for item in session["sidecars"]:
            record = read_json(session_path(output, session_id) / "input" / f"source{item['suffix']}")
            # Do not embed the previous provenance twice on every iteration.
            payload = {"record_ref": "source_record"} if record == source else {"record": record}
            source_sidecars.append({**item, **payload})
        source_output = source.get("output")
        classification = source.get("classification") or (source_output.get("classification") if isinstance(source_output, dict) else None)
        if not source.get("license"):
            codes.append("provenance_unknown")
        relative = f"{DIRECTORY}/versions/{session_id}"
        version = {"schema_version": 1, "id": session_id, "asset_id": session["asset_id"], "asset_name": session["asset_name"],
                   "created_at": now(), "parent_file": session["parent_file"], "parent_sha256": session["parent_sha256"],
                   "parent_version": session["parent_version"], "file": f"{relative}/model.glb",
                   "provenance": f"{relative}/model.provenance.json", "metadata": f"{relative}/version.json",
                   "label": filename.replace("\\", "/").split("/")[-1], "sha256": sha(data),
                   "size_bytes": len(data), "facts": facts, "warning_codes": codes,
                   "derived_artifacts": "not_inherited"}
        provenance = {"schema_version": 1, "created_at": version["created_at"], "tool": "AssetFurnace manual Blender exchange",
                      "input": {"source": session["parent_file"], "sha256": session["parent_sha256"]},
                      "output": {"sha256": version["sha256"], "classification": classification},
                      "license": source.get("license"), "classification": classification,
                      "source_record": source or None, "source_sidecars": source_sidecars,
                      "manual_edit": {"session_id": session_id, "generator": facts["generator"],
                      "additional_content_licenses": "not_verified", "derived_artifacts": "not_inherited"}}
        staging = safe_path(output, f"{DIRECTORY}/versions/.pending-{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
        with (staging / "model.glb").open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        write_json(staging / "model.provenance.json", provenance)
        write_json(staging / "version.json", version)
        staging.rename(target)  # publication is atomic; failed writes stay invisible
        return version
