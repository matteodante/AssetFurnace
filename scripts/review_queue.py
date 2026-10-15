#!/usr/bin/env python3
"""See the review queue, and approve an asset or send it back with a note.

    python scripts/review_queue.py --list
    python scripts/review_queue.py --approve mob_0412 --sha <version from --list>
    python scripts/review_queue.py --send-back mob_0412 --sha <version> --note "horns too short"

The same queue the studio's Review sheet shows (`output/review-log.jsonl`), for agents and
the terminal. A send-back needs a note: it stays in the queue, note in view, until a new
version is checked. Every decision is also recorded as a verdict in `output/verdicts.jsonl`
(`mark_asset.py`), so approvals and send-backs grow the same dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

VERDICT = {"approved": "good", "sent_back": "bad"}


def decide(key: str, decision: str, note: str = "", log: Path | None = None,
           register: Path | None = None, asset_sha256: str | None = None) -> tuple[dict, str | None]:
    """Record a decision in the review log and as a `mark_asset` verdict.

    Returns the asset's new state, and why the verdict was not written (None if it was).
    The review decision is the one that counts; a verdict that cannot be written (the file
    has moved) is reported, never allowed to lose the decision.
    """
    import mark_asset

    from image_to_3dlab.checks import review

    item = review.decide(log or review.LOG, key, decision, note, asset_sha256=asset_sha256)
    asset = Path(item.get("asset") or "")
    if not asset.is_file():
        return item, f"no verdict written: {asset} is not on disk"
    reason = note.strip() or f"approved in the review queue (checks: {item.get('status')})"
    entry = mark_asset.build_entry(asset, VERDICT[decision], reason, tags=["review-queue"])
    if entry["asset_sha256"] != item.get("asset_sha256"):
        # Overwritten between the decision and now: a verdict on these bytes would be about a
        # file nobody judged. The decision stands; the verdict is left out and said so.
        return item, f"no verdict written: {asset.name} changed on disk after the decision"
    mark_asset.append_entry(register or mark_asset.REGISTER, entry)
    return item, None


def _format(item: dict) -> str:
    fails = ", ".join(item.get("summary", {}).get("fail", [])) or "-"
    line = (f"{item['state']:9}  {item['key']:30} checks {item.get('status')} (fail: {fails})"
            f"\n           version {item.get('asset_sha256')}")
    if item["notes"]:
        line += f"\n           last note: {item['notes'][-1]['note']}"
    return line


def main(argv: list[str] | None = None) -> int:
    from image_to_3dlab.checks import review

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="what waits for review")
    action.add_argument("--approve", metavar="KEY")
    action.add_argument("--send-back", metavar="KEY")
    parser.add_argument("--note", default="", help="why (required for --send-back)")
    parser.add_argument("--sha", help="the version you looked at, as --list prints it (required to decide)")
    parser.add_argument("--log", type=Path, help="review log (default output/review-log.jsonl)")
    args = parser.parse_args(argv)
    log = args.log or review.LOG
    if args.list:
        items = review.queue(review.load(log))
        print("\n".join(_format(i) for i in items) if items else "Nothing waits for review.")
        return 0
    key, decision = (args.approve, "approved") if args.approve else (args.send_back, "sent_back")
    if not args.sha:
        print("error: add --sha with the version you looked at (see --list), so a newer one is not "
              "decided by mistake", file=sys.stderr)
        return 2
    try:
        item, problem = decide(key, decision, args.note, log, asset_sha256=args.sha)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format(item))
    if problem:
        print(problem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
