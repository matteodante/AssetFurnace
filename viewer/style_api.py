"""The style bibles the studio can draw with.

    GET /api/style-bible     {"active": name or null, "bibles": [names]}

A bible is made and filled with `scripts/style_bible.py`; the studio only offers to draw
with the active one ("Use the style bible" in Create, the reference panel and the Pieces
page), which is opt-in: nothing changes unless it is ticked.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from image_to_3dlab.style import registry


def info(output: Path) -> dict:
    current = registry.active(output)
    return {"active": current["name"] if current else None,
            "bibles": [b["name"] for b in registry.bibles(output)]}


def handle(handler, output: Path) -> bool:
    """Answer GET /api/style-bible; False for any other path."""
    if urlsplit(handler.path).path != "/api/style-bible":
        return False
    try:
        handler._send_json(200, info(Path(output)))
    except (OSError, ValueError) as exc:
        handler._send_json(200, {"active": None, "bibles": [], "error": str(exc)})
    return True
