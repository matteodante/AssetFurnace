#!/usr/bin/env python3
"""Keep a project's style bible: approved and rejected pictures with notes, prompts and style references.

    python scripts/style_bible.py new my-game --wording "bold saturated colours"
    python scripts/style_bible.py new my-game --at ../my-game/art/style     # kept in the game's repo
    python scripts/style_bible.py use my-game                                 # "-" now means my-game
    python scripts/style_bible.py add - output/images/x/x.png --note "Colours are right." --kind creature
    python scripts/style_bible.py reject - wolf.png --note "Grey and photographic."
    python scripts/style_bible.py list -
    python scripts/style_bible.py prompt - creature "a forest troll" --references 2
    python scripts/style_bible.py refs - creature --k 2

A bible is named by its registered name, by a folder, or by `-` for the active one.
Pictures are named by the first characters of their hash (`list` shows them). Exit code
0 means done, 2 means refused, with the reason on one line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab.style import bible as sb  # noqa: E402
from image_to_3dlab.style import prompts, refs, registry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=REPO / "output",
                        help="AssetFurnace's output folder, where styles/bibles.json lives")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="make a new bible and register it")
    new.add_argument("name")
    new.add_argument("--at", type=Path, help="folder for the bible (default: <output>/styles/<name>)")
    new.add_argument("--description", default="")
    new.add_argument("--wording", default="", help="the style in words, added to every prompt")

    commands.add_parser("register", help="register an existing bible folder").add_argument("folder", type=Path)
    commands.add_parser("unregister", help="forget a bible (the folder stays)").add_argument("name")
    commands.add_parser("use", help="make a bible the active one").add_argument("name")
    commands.add_parser("bibles", help="list the registered bibles")

    for name, text in (("add", "add an approved picture"), ("reject", "add a picture that is not the style")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("bible")
        sub.add_argument("image", type=Path)
        sub.add_argument("--note", required=True, help="why it is (or is not) the style")
        sub.add_argument("--kind", choices=sb.KINDS)
        sub.add_argument("--source", help="where the picture came from, for the record")

    remove = commands.add_parser("remove", help="take a picture out of the bible")
    remove.add_argument("bible")
    remove.add_argument("picture", help="the first characters of its hash")

    note = commands.add_parser("note", help="change a picture's note")
    note.add_argument("bible")
    note.add_argument("picture")
    note.add_argument("text")

    show = commands.add_parser("list", help="show a bible's pictures and any problems")
    show.add_argument("bible")
    show.add_argument("--json", action="store_true")

    prompt = commands.add_parser("prompt", help="print the prompt for one asset")
    prompt.add_argument("bible")
    prompt.add_argument("kind", choices=sb.KINDS)
    prompt.add_argument("subject")
    prompt.add_argument("--references", type=int, default=0, help="how many style pictures go with it")

    pick = commands.add_parser("refs", help="print the style references for a kind")
    pick.add_argument("bible")
    pick.add_argument("kind", choices=sb.KINDS)
    pick.add_argument("--k", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> int:
    output = args.output
    if args.command == "new":
        root = sb.create(args.at or output / "styles" / args.name, args.name, args.description, args.wording)
        registry.register(output, root)
        print(f"made {args.name} in {root}")
    elif args.command == "register":
        print(f"registered {registry.register(output, args.folder)['name']}")
    elif args.command == "unregister":
        registry.unregister(output, args.name)
    elif args.command == "use":
        registry.set_active(output, args.name)
    elif args.command == "bibles":
        current = registry.active(output)
        for known in registry.bibles(output):
            mark = "  (active)" if current and current["name"] == known["name"] else ""
            print(f"{known['name']}  {known['path']}{mark}")
    elif args.command in ("add", "reject"):
        entry = sb.add(registry.find(output, args.bible), args.image, args.note, kind=args.kind,
                       rejected=args.command == "reject", source_label=args.source)
        print(f"{entry['sha256'][:8]}  {entry['file']}")
    elif args.command == "remove":
        print(f"removed {sb.remove(registry.find(output, args.bible), args.picture)['file']}")
    elif args.command == "note":
        sb.set_note(registry.find(output, args.bible), args.picture, args.text)
    elif args.command == "list":
        root = registry.find(output, args.bible)
        data = sb.load(root)
        found = {"name": data["name"], "path": str(root), "images": data["images"],
                 "rejected": data["rejected"], "problems": sb.problems(root)}
        if args.json:
            print(json.dumps(found, indent=2))
        else:
            for key, label in (("images", "approved"), ("rejected", "rejected")):
                print(f"{label}: {len(found[key])}")
                for e in found[key]:
                    print(f"  {e['sha256'][:8]}  {e.get('kind') or 'any':<14} {e['note']}")
            for problem in found["problems"]:
                print(f"problem: {problem}")
    elif args.command == "prompt":
        root = registry.find(output, args.bible)
        print(prompts.compose(args.subject, args.kind, bible=root, references=args.references)["prompt"])
    elif args.command == "refs":
        for path in refs.pick(registry.find(output, args.bible), args.kind, k=args.k):
            print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse already printed why
        return int(exc.code or 0)
    try:
        return run(args)
    except (sb.BibleError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
