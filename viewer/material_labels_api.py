"""The studio's material labels: the user's judgements on an asset in the review queue.

    GET  /api/material-labels                 vocabulary, and progress over the label set
    GET  /api/material-labels/item/<key>      the labels of the version under review
    POST /api/material-labels/item/<key>      {"asset_sha256", "baked_light", "materials_look", "points", "note"}

The label set is every review item queued with source ``material-labels``
(``scripts/material_label_set.py``). Labels are kept in ``output/material-labels.jsonl``
(``image_to_3dlab/material_labels.py``) and bound to the version the review queue shows:
a save for any other version is refused. Saving needs the studio's own page: same origin
and an ``X-AssetFurnace-Request: material-labels`` header.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from blender_edit_api import check_request

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import material_labels as ml
from image_to_3dlab.checks import review
from image_to_3dlab.provenance import sha256_file

HEADER = "material-labels"
SOURCE = "material-labels"
MAX_BODY = 64 * 1024


class VersionConflict(ValueError):
    """The labels name a version the review queue no longer shows."""


def _labels_log(output: Path) -> Path:
    return output / "material-labels.jsonl"


def _review_items(output: Path) -> dict:
    return review.fold(review.load(output / "review-log.jsonl"))


def label_set(output: Path) -> dict[str, str]:
    """The review items queued for labelling: key -> the version in the queue."""
    return {key: item.get("asset_sha256") for key, item in _review_items(output).items()
            if item.get("source") == SOURCE}


def _labelled_item(output: Path, key: str) -> dict:
    """The review item for `key`, if it belongs to the label set; labels elsewhere would be lost."""
    item = _review_items(output).get(key)
    if item is None or item.get("source") != SOURCE:
        raise KeyError(key)
    return item


def overview(output: Path) -> dict:
    return {"materials": list(ml.MATERIALS), "looks": list(ml.LOOKS),
            "progress": ml.progress(ml.load(_labels_log(output)), label_set(output))}


def item_labels(output: Path, key: str) -> dict:
    item = _labelled_item(output, key)
    labels = ml.fold(ml.load(_labels_log(output))).get(key)
    if labels is None or labels["asset_sha256"] != item.get("asset_sha256"):
        labels = {"key": key, "asset_sha256": item.get("asset_sha256"), "baked_light": None,
                  "materials_look": None, "points": [], "note": ""}
    return {"labels": labels, "complete": ml.is_complete(labels)}


def save_labels(output: Path, key: str, body: dict) -> dict:
    item = _labelled_item(output, key)
    if body.get("asset_sha256") != item.get("asset_sha256"):
        raise VersionConflict("these labels are for another version than the one under review; reload it")
    on_disk = Path(item.get("asset") or "")
    if not on_disk.is_file() or sha256_file(on_disk) != item["asset_sha256"]:
        # Studio shows the file on disk: labels clicked on a replacement must not be filed
        # under the hash of the version that was queued.
        raise VersionConflict("the file on disk is not the version in the queue; queue it again")
    fields = {k: body[k] for k in ml.FIELDS if k in body}
    labels = ml.record(_labels_log(output), key, item.get("asset") or "", item["asset_sha256"], **fields)
    return {"labels": labels, "complete": ml.is_complete(labels)}


def _read_body(handler) -> bytes:
    # Read before judging: answering a refused request without reading what it sent makes
    # Windows reset the connection, and the page sees a network error, not the reason.
    values = handler.headers.get_all("Content-Length", [])
    if len(values) != 1 or not values[0].isdigit():
        return b""
    length = int(values[0])
    return handler.rfile.read(length) if 0 < length <= MAX_BODY else b""


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    url = urlsplit(handler.path)
    if not url.path.startswith("/api/material-labels"):
        return False
    parts = [unquote(p) for p in url.path.strip("/").split("/")]
    output = Path(output)
    raw = _read_body(handler) if mutation else b""
    try:
        check_request(handler)
        if mutation and (not handler.headers.get("Origin") or handler.headers.get("X-AssetFurnace-Request") != HEADER):
            raise PermissionError("Use this Studio page to label materials")
        if not mutation and parts == ["api", "material-labels"]:
            handler._send_json(200, overview(output))
        elif len(parts) == 4 and parts[2] == "item":
            if mutation:
                body = json.loads(raw) if raw else None
                if not isinstance(body, dict):
                    raise ValueError("Send a JSON object of at most 64 KB")
                handler._send_json(200, save_labels(output, parts[3], body))
            else:
                handler._send_json(200, item_labels(output, parts[3]))
        else:
            handler._send_json(404, {"error": "No such labels request"})
    except PermissionError as exc:
        handler._send_json(403, {"error": str(exc)})
    except VersionConflict as exc:
        handler._send_json(409, {"error": str(exc)})
    except KeyError as exc:
        handler._send_json(404, {"error": f"{exc.args[0]} is not in the material label set"})
    except (ValueError, TypeError) as exc:
        handler._send_json(400, {"error": str(exc)})
    return True
