"""Same-origin manual Blender exchange endpoints and local desktop launch."""

from __future__ import annotations

import ipaddress
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import blender_versions as versions

from image_to_3dlab.blender import find_blender
from image_to_3dlab.blender_exchange import MAX_BYTES

REPO = Path(__file__).resolve().parents[1]
PROCESSES = {}


def local_address(value):
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def desktop_reason(handler):
    host = urlsplit("//" + handler.headers.get("Host", "")).hostname or ""
    if not local_address(handler.client_address[0]) or not local_address(host):
        return "Open in Blender needs Studio on this computer. Download the GLB, edit locally, then import it here."
    if any(os.environ.get(key) for key in ("SSH_CONNECTION", "SSH_CLIENT", "RUNPOD_POD_ID")):
        return "This server is remote. Download the GLB, edit in local Blender, then import it here."
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "This server has no desktop. Download the GLB, edit in local Blender, then import it here."
    if find_blender() is None:
        return "Blender was not found. Install Blender or set I2L_BLENDER to its executable, then retry. Import is still available."
    return None


def check_host(handler):
    """The Host a request names. A loopback server only answers loopback names, so a page
    that rebinds its own domain to 127.0.0.1 cannot read or drive it (DNS rebinding)."""
    host = handler.headers.get("Host", "")
    parsed = urlsplit("//" + host)
    if not parsed.hostname or parsed.username or parsed.path or parsed.query or parsed.fragment:
        raise PermissionError("Invalid request host")
    if local_address(handler.server.server_address[0]) and not local_address(parsed.hostname):
        raise PermissionError("Use the local Studio address")
    return host


def check_request(handler, *, mutation=False, header="blender-edit"):
    """The Studio page only: a loopback Host on a loopback server, the same origin, never a
    cross-site fetch, and on mutations a custom header that a cross-site form cannot send.
    ``header=None`` accepts any value of it: the older Studio endpoints are shared by pages
    that each name their own."""
    host = check_host(handler)
    # behind a proxy (RunPod) the page's origin is the public name the proxy forwards
    allowed = {host.lower()}
    forwarded = handler.headers.get("X-Forwarded-Host")
    if forwarded and not local_address(handler.server.server_address[0]):
        allowed.add(forwarded.split(",")[0].strip().lower())
    origin = handler.headers.get("Origin")
    if origin:
        source = urlsplit(origin)
        if source.scheme not in ("http", "https") or source.netloc.lower() not in allowed or source.path or source.query or source.fragment:
            raise PermissionError("Use this Studio page to edit models" if header == "blender-edit" else "Use the Studio page")
    if handler.headers.get("Sec-Fetch-Site") == "cross-site":
        raise PermissionError("Cross-site edit requests are not allowed" if header == "blender-edit" else "Cross-site requests are not allowed")
    sent = handler.headers.get("X-AssetFurnace-Request")
    if mutation and (not origin or not sent or (header is not None and sent != header)):
        raise PermissionError("Use this Studio page to edit models" if header == "blender-edit" else "Use the Studio page")


def body(handler, maximum):
    values = handler.headers.get_all("Content-Length", [])
    if len(values) != 1 or handler.headers.get("Transfer-Encoding"):
        raise ValueError("A single Content-Length is required")
    length = int(values[0])
    if not 0 < length <= maximum:
        raise ValueError("Request is empty or exceeds the upload limit (GLB: 512 MiB)")
    before = handler.connection.gettimeout()
    try:
        handler.connection.settimeout(60)
        data = handler.rfile.read(length)
    finally:
        handler.connection.settimeout(before)
    if len(data) != length:
        raise ValueError("Upload was interrupted; choose the file again")
    return data


def launch(output, session_id, blender=None, popen=None):
    with versions.LOCK:
        session = versions.read_session(output, session_id)
        root = versions.session_path(output, session_id)
        source = versions.checked_snapshot(output, session)
        process = PROCESSES.get((str(output), session_id))
        if process is not None and process.poll() is None:
            return {"state": "opening", "already_open": True}
        binary = blender or find_blender()
        if binary is None:
            raise ValueError("Blender was not found; install it or download the GLB to edit elsewhere")
        workspace = versions.safe_path(output, (root / "work/model.blend").relative_to(output).as_posix())
        status_file = versions.safe_path(output, (root / "open-status.json").relative_to(output).as_posix())
        versions.write_json(status_file, {"state": "opening"})
        args = [str(binary), "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                "--python", str(REPO / "scripts/blender_open_asset.py"), "--", "--source", str(source),
                "--workspace", str(workspace), "--status", str(status_file)]
        try:
            with (root / "blender.log").open("ab") as log:
                process = (popen or subprocess.Popen)(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                      cwd=REPO, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            versions.write_json(status_file, {"state": "error", "error": "Blender could not start. Check its installation."})
            raise ValueError("Blender could not start. Check its installation.") from exc
        PROCESSES[(str(output), session_id)] = process
        return {"state": "opening", "already_open": False}


def status(output, session_id):
    root = versions.session_path(output, session_id)
    session = versions.read_session(output, session_id)
    result = {"state": "not_opened"}
    try:
        result = versions.read_json(root / "open-status.json")
    except FileNotFoundError:
        pass
    process = PROCESSES.get((str(output), session_id))
    if result.get("state") == "opening" and process is not None and process.poll() is not None:
        result = {"state": "error", "error": "Blender closed before importing the model. Retry opening it."}
    return {"session": session, "launch": result}


def private_file(output, path):
    root, original = (output / versions.DIRECTORY).absolute(), Path(path).absolute()
    resolved = original.resolve()
    if original.is_relative_to(root):
        parts = original.relative_to(root).parts
        if not resolved.is_relative_to(root.resolve()) or resolved != root.resolve().joinpath(*parts):
            return True
    elif resolved.is_relative_to(root.resolve()):
        parts = resolved.relative_to(root.resolve()).parts
    else:
        return False
    return not (len(parts) == 3 and parts[0] == "versions" and versions.ID.fullmatch(parts[1])
                and parts[2] in ("model.glb", "model.provenance.json", "version.json"))


def handle(handler, output, *, mutation=False):
    url = urlsplit(handler.path)
    if not url.path.startswith("/api/blender-edit"):
        return False
    parts = url.path.strip("/").split("/")
    try:
        check_request(handler, mutation=mutation)
        if not mutation and parts == ["api", "blender-edit"]:
            asset_id = parse_qs(url.query).get("asset_id", [""])[0]
            reason = desktop_reason(handler)
            handler._send_json(200, {"can_open": reason is None, "reason": reason,
                                     "max_bytes": MAX_BYTES, "sessions": versions.list_sessions(output, asset_id)})
        elif not mutation and len(parts) == 4 and parts[2] == "sessions":
            handler._send_json(200, status(output, parts[3]))
        elif not mutation and len(parts) == 5 and parts[2] == "sessions" and parts[4] == "source.glb":
            session = versions.read_session(output, parts[3])
            path = versions.checked_snapshot(output, session)
            data = versions.read_glb(path)
            handler.send_response(200)
            handler.send_header("Content-Type", "model/gltf-binary")
            handler.send_header("Content-Disposition", 'attachment; filename="blender-edit-source.glb"')
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        elif mutation and parts == ["api", "blender-edit", "sessions"]:
            if handler.headers.get_content_type() != "application/json":
                raise ValueError("Expected JSON")
            data = json.loads(body(handler, 8192))
            handler._send_json(201, versions.create_session(output, data["asset_id"], data["file"], data["request_id"]))
        elif mutation and len(parts) == 5 and parts[2] == "sessions" and parts[4] == "open":
            if handler.headers.get("Content-Length", "0") != "0" or handler.headers.get("Transfer-Encoding"):
                raise ValueError("Open does not accept an upload")
            reason = desktop_reason(handler)
            if reason:
                raise ValueError(reason)
            handler._send_json(202, launch(output, parts[3]))
        elif mutation and len(parts) == 5 and parts[2] == "sessions" and parts[4] == "import":
            if handler.headers.get_content_type() != "model/gltf-binary":
                raise ValueError("Choose a binary GLB file")
            data = body(handler, MAX_BYTES)
            filename = unquote(handler.headers.get("X-Filename", ""))
            handler._send_json(201, versions.import_version(output, parts[3], data, filename))
        else:
            handler._send_json(404, {"error": "Unknown Blender edit endpoint"})
    except PermissionError as exc:
        handler.close_connection = True
        handler._send_json(403, {"error": str(exc)})
    except (ValueError, KeyError, TypeError, OSError) as exc:
        handler.close_connection = True
        handler._send_json(422, {"error": str(exc) if not isinstance(exc, KeyError) else "Missing edit information"})
    return True
