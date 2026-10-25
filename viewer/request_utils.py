"""What every Studio API does with a request: who may send it, its body, the files it names.

One module, so the pages that came one by one (Blender edits, review, the
reference concept, Unreal export, batches, characters in pieces) cannot drift apart:

- `check_host` / `check_request`: a loopback Host on a loopback server (no DNS rebinding),
  the same origin, never a cross-site fetch, and on a mutation the page's own
  `X-AssetFurnace-Request` header; a script that sent the access token as a Bearer header
  passes (`Handler._admitted`).
- `read_body` / `read_json` / `drain`: the body read before anything is judged. Answering a
  refused request without reading what it sent makes Windows reset the connection, and the
  page sees a network error instead of the reason.
- `inside_output`: a file the page names, relative to `output/`, that exists there.
"""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit


def local_address(value):
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


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


def check_request(handler, *, mutation=False, header="blender-edit", refusal=None):
    """The Studio page only: a loopback Host on a loopback server, the same origin, never a
    cross-site fetch, and on mutations a custom header that a cross-site form cannot send.
    ``header=None`` accepts any value of it: the older Studio endpoints are shared by pages
    that each name their own. A script that sent the access token as a Bearer header
    (`Handler._admitted`) is not a browser, so it cannot be a forged page: it passes, on
    every endpoint, as serve.py promises. `refusal` is what the page's own refusal says."""
    if getattr(handler, "_bearer", False):
        return
    refusal = refusal or ("Use this Studio page to edit models" if header == "blender-edit" else "Use the Studio page")
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
            raise PermissionError(refusal)
    if handler.headers.get("Sec-Fetch-Site") == "cross-site":
        raise PermissionError("Cross-site edit requests are not allowed" if header == "blender-edit" else "Cross-site requests are not allowed")
    sent = handler.headers.get("X-AssetFurnace-Request")
    if mutation and (not origin or not sent or (header is not None and sent != header)):
        raise PermissionError(refusal)


def content_length(handler) -> int | None:
    """The one Content-Length a request gives, or None (none, several, or not a number)."""
    values = handler.headers.get_all("Content-Length", []) or []
    if len(values) != 1 or not str(values[0]).strip().isdigit():
        return None
    return int(values[0])


def read_body(handler, maximum: int) -> bytes:
    """The body, read at once; empty when there is none, it is over `maximum`, or the request
    gives no single Content-Length (the caller says what it wanted)."""
    length = content_length(handler)
    return handler.rfile.read(length) if length and 0 < length <= maximum else b""


def read_json(handler, maximum: int) -> dict[str, Any]:
    """The body as JSON; {} when there is none (or it is over `maximum`). Bad JSON raises
    ValueError, as `json.loads` does."""
    raw = read_body(handler, maximum)
    return json.loads(raw) if raw else {}


def drain(handler, limit=1024 * 1024):
    """Read the body of a request refused before it was read, so the answer arrives: closing
    on unread bytes makes Windows reset the connection, and the caller saw
    ConnectionAbortedError instead of the reason. A body over `limit` is left unread."""
    length = content_length(handler) or 0
    if 0 < length <= limit:
        handler.rfile.read(length)
    handler.close_connection = True


class OutsideOutput(ValueError):
    """A path that is not a file of the output folder."""


class MissingInOutput(OutsideOutput, FileNotFoundError):
    """A path inside the output folder where no file is."""


def inside_output(output: Path, relative, *, suffixes: tuple[str, ...] = (), refusal: str | None = None,
                  missing: str | None = None) -> Path:
    """`relative`, as the Library spells it, resolved inside `output/` as an existing file.

    Refused (`OutsideOutput`): anything but a non-empty string, a backslash (the Library spells
    paths with /), an absolute path (a leading / or a drive), one that climbs out (`..`, or a
    link resolving outside), or another suffix than `suffixes` when given. No file there:
    `MissingInOutput`, a FileNotFoundError too. `refusal` and `missing` are the page's own
    words, `{relative}` filled in."""
    refusal = refusal or "{relative} is not a file in the output folder"
    if (not isinstance(relative, str) or not relative or "\\" in relative or relative.startswith("/")
            or PureWindowsPath(relative).drive or Path(relative).is_absolute()):
        raise OutsideOutput(refusal.format(relative=relative))
    root = Path(output).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root or (suffixes and path.suffix.lower() not in suffixes):
        raise OutsideOutput(refusal.format(relative=relative))
    if not path.is_file():
        raise MissingInOutput((missing or refusal).format(relative=relative))
    return path
