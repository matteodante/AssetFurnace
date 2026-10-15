"""The studio's review queue: check an asset, look at it, approve it or send it back.

    GET  /api/review/queue                 what waits, plus the latest decisions
    GET  /api/review/item/<key>            one asset: state, notes, history, its checks report
    POST /api/review/checks                {"asset": <path under output/>, "key", "reference", "profile", "height"}
    POST /api/review/item/<key>/decision   {"decision": "approved"|"sent_back", "note": "..."}

The record is `output/review-log.jsonl` (`image_to_3dlab/checks/review.py`); a decision is
also written as a verdict in `output/verdicts.jsonl`, as `scripts/review_queue.py` does.
Changes need the studio's own page: same origin and an `X-AssetFurnace-Request: review` header.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from blender_edit_api import check_request

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):  # the checks package, and review_queue's decide()
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

MAX_BODY = 16 * 1024
RECENT = 10


def url_for(path: str | Path | None, repo: Path | None = None) -> str | None:
    """Where the studio server serves a file (it serves the repository), or None."""
    if not path:
        return None
    try:
        return "/" + Path(path).resolve().relative_to((repo or REPO).resolve()).as_posix()
    except ValueError:
        return None


def _log(output: Path) -> Path:
    return output / "review-log.jsonl"


def _inside(output: Path, relative: str, what: str) -> Path:
    if not isinstance(relative, str) or not relative or relative.startswith(("/", "\\")):
        raise ValueError(f"{what} must be a path under output/")
    path = (output / relative).resolve()
    try:
        path.relative_to(output.resolve())
    except ValueError as exc:
        raise ValueError(f"{what} must be a path under output/") from exc
    if not path.is_file():
        raise ValueError(f"{what} {relative} does not exist")
    return path


def _brief(item: dict) -> dict:
    return {k: item.get(k) for k in ("key", "state", "status", "summary", "notes", "updated_at", "asset_sha256",
                                     "source", "kind")} | {
        "asset_url": url_for(item.get("asset")), "name": Path(item.get("asset") or "").name}


def queue_payload(output: Path) -> dict:
    from image_to_3dlab.checks import review

    events = review.load(_log(output))
    decided = [i for i in review.fold(events).values() if i["state"] == "approved"]
    decided.sort(key=lambda i: i.get("updated_at") or "", reverse=True)
    return {"items": [_brief(i) for i in review.queue(events)], "approved": [_brief(i) for i in decided[:RECENT]]}


def item_payload(output: Path, key: str) -> dict:
    from image_to_3dlab.checks import review

    item = review.fold(review.load(_log(output))).get(review.safe_key(key))
    if item is None:
        raise KeyError(key)
    out = _brief(item) | {"history": item["history"], "report_url": url_for(item.get("report"))}
    report = None
    if item.get("report") and Path(item["report"]).is_file():
        report = json.loads(Path(item["report"]).read_text(encoding="utf-8"))
        folder = Path(item["report"]).parent
        for found in report.get("results", []):
            overlays = found.get("details", {}).get("overlays")
            if overlays:
                found["details"]["overlay_urls"] = {view: url_for(folder / rel) for view, rel in overlays.items()}
        reference = report.get("reference") or {}
        if reference.get("path"):
            reference["url"] = url_for(reference["path"])
    out["report"] = report
    return out


def run_checks_request(output: Path, body: dict) -> dict:
    from image_to_3dlab.checks import Asset, profile_names, review

    glb = _inside(output, body.get("asset"), "asset")
    if glb.suffix.lower() != ".glb":
        raise ValueError("asset must be a .glb")
    reference = _inside(output, body["reference"], "reference") if body.get("reference") else None
    profile = body.get("profile") or "prop"
    if profile not in profile_names():
        raise ValueError(f"profile must be one of {profile_names()}")
    numbers = {k: float(body[k]) for k in ("height", "width", "depth") if body.get(k)}
    asset = Asset(glb=glb, key=body.get("key"), reference=reference, reference_numbers=numbers or None,
                  output_root=output)
    report, _ = review.check_and_enqueue(asset, log=_log(output), profile=profile, key=body.get("key"))
    return item_payload(output, report["asset_key"])


def decide_request(output: Path, key: str, body: dict) -> dict:
    import review_queue

    _, problem = review_queue.decide(key, body.get("decision"), body.get("note") or "",
                                     log=_log(output), register=output / "verdicts.jsonl",
                                     asset_sha256=body.get("asset_sha256"))
    return item_payload(output, key) | {"verdict_problem": problem}


def review_conflict() -> type:
    from image_to_3dlab.checks.review import ReviewConflict

    return ReviewConflict


def _read_body(handler) -> bytes:
    """The request body, read before anything is judged.

    Answering a refused request (403, 400) without reading what it sent makes Windows reset
    the connection, and the browser sees a network error instead of the reason.
    """
    values = handler.headers.get_all("Content-Length", [])
    if len(values) != 1:
        return b""
    length = int(values[0]) if values[0].isdigit() else 0
    return handler.rfile.read(length) if 0 < length <= MAX_BODY else b""


def _body(raw: bytes) -> dict:
    if not raw:
        raise ValueError("Send a JSON object of at most 16 KB, with a single Content-Length")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object")
    return payload


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    """Serve `/api/review...`; return False for any other path."""
    url = urlsplit(handler.path)
    if not url.path.startswith("/api/review"):
        return False
    parts = [unquote(p) for p in url.path.strip("/").split("/")]
    output = Path(output)
    raw = _read_body(handler) if mutation else b""
    try:
        check_request(handler)
        if mutation and (not handler.headers.get("Origin") or handler.headers.get("X-AssetFurnace-Request") != "review"):
            raise PermissionError("Use this Studio page to review assets")
        if not mutation and parts == ["api", "review", "queue"]:
            handler._send_json(200, queue_payload(output))
        elif not mutation and len(parts) == 4 and parts[2] == "item":
            handler._send_json(200, item_payload(output, parts[3]))
        elif mutation and parts == ["api", "review", "checks"]:
            handler._send_json(200, run_checks_request(output, _body(raw)))
        elif mutation and len(parts) == 5 and parts[2] == "item" and parts[4] == "decision":
            handler._send_json(200, decide_request(output, parts[3], _body(raw)))
        else:
            handler._send_json(404, {"error": "No such review request"})
    except PermissionError as exc:
        handler._send_json(403, {"error": str(exc)})
    except review_conflict() as exc:
        handler._send_json(409, {"error": str(exc)})
    except KeyError as exc:
        handler._send_json(404, {"error": f"{exc.args[0]} is not in the review queue"})
    except (ValueError, TypeError) as exc:
        handler._send_json(400, {"error": str(exc)})
    return True
