#!/usr/bin/env python3
"""Write the outfit's words into sets made before they were saved with the concept (one-off).

    python scripts/migrate_set_outfits.py [--output output] [--dry-run]

Since 18/10/2026 a set's concept keeps the outfit's words it was drawn from
(`concept.outfit` in `<set>.parts.json`, and in `concept_info` of a run waiting for picks),
and `image_to_3dlab/piece_materials.py` reads the pieces' materials from there. A set made
before kept only the edit prompt, and `piece_materials` refuses it. This command takes the
words out of that prompt once, the way they were read until then ("Add <outfit> to the
character in image 1..."; a prompt in another form is taken whole, a concept given
ready-made has none), and writes them in. Run it once per output folder; a set that has its
words already is left as it is, so a second run changes nothing. Delete this script once no
set made before 18/10/2026 is left.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def words_in(prompt: str) -> str:
    """The outfit's words inside a concept's edit prompt, as piece_materials read them until 18/10/2026."""
    found = re.match(r"\s*Add (.*?) to the character in image 1", str(prompt or ""), re.S)
    return found.group(1) if found else str(prompt or "")


def _write(path: Path, data: dict) -> None:
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    staged.replace(path)


def migrate(output: Path, dry_run: bool = False) -> list[tuple[Path, str]]:
    """Every record under `output/sets` given its outfit's words; returns (record, words) per change."""
    changed = []
    for folder in sorted(Path(output).glob("sets/*/*")):
        for path in sorted(folder.glob("*.parts.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            concept = data.get("concept")
            if isinstance(concept, dict) and "outfit" not in concept:
                concept["outfit"] = words_in(concept.get("prompt", ""))
                changed.append((path, concept["outfit"]))
                if not dry_run:
                    _write(path, data)
        state_path = folder / "run_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            info = state.get("concept_info")
            if state.get("concept") and isinstance(info, dict) and "outfit" not in info:
                info["outfit"] = words_in(info.get("prompt", ""))
                changed.append((state_path, info["outfit"]))
                if not dry_run:
                    _write(state_path, state)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("output"), help="the output folder (default output)")
    parser.add_argument("--dry-run", action="store_true", help="say what would change, write nothing")
    args = parser.parse_args(argv)
    changed = migrate(args.output, args.dry_run)
    for path, words in changed:
        print(f"{'would write' if args.dry_run else 'wrote'} {path}: outfit {words!r}")
    print(f"{len(changed)} record(s) {'to change' if args.dry_run else 'changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
