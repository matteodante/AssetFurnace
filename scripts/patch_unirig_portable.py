#!/usr/bin/env python3
"""Make UniRig's skeleton model run without flash-attn.

    python scripts/patch_unirig_portable.py [vendor/UniRig]

UniRig asks for flash-attn twice: its language model is configured for it, and its model
registry imports the skin model, which needs it, even when only the skeleton model runs.
flash-attn has no Windows wheel, and building it needs a CUDA toolkit this kind of machine
does not have. Two edits, both upstream's own options: the language model uses PyTorch's
attention (SDPA), and the skin model is imported only when it is asked for. The creature rig
uses the skeleton model alone, so nothing it runs loses anything.

Each edit checks its anchor and is skipped when already made, so running this twice is safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "UniRig"

CONFIG = "configs/model/unirig_ar_350m_1024_81920_float32.yaml"
CONFIG_OLD = "_attn_implementation: flash_attention_2"
CONFIG_NEW = "_attn_implementation: sdpa"

PARSE = "src/model/parse.py"
PARSE_IMPORT = "from .unirig_skin import UniRigSkin\n"
PARSE_MAP_OLD = "'unirig_skin': UniRigSkin,"
PARSE_MAP_NEW = "'unirig_skin': None,"
PARSE_RETURN = "    return MAP[__target__](**kwargs)"
PARSE_LAZY = ("    if __target__ == 'unirig_skin':  # flash-attn only for the skin model\n"
              "        from .unirig_skin import UniRigSkin\n"
              "        MAP['unirig_skin'] = UniRigSkin\n")


class PatchError(RuntimeError):
    pass


def patch_config(text: str) -> str:
    if CONFIG_NEW in text:
        return text
    if CONFIG_OLD not in text:
        raise PatchError(f"{CONFIG}: neither {CONFIG_OLD!r} nor {CONFIG_NEW!r} found")
    return text.replace(CONFIG_OLD, CONFIG_NEW)


def patch_parse(text: str) -> str:
    if PARSE_LAZY in text:
        return text
    if PARSE_IMPORT not in text or PARSE_MAP_OLD not in text or PARSE_RETURN not in text:
        raise PatchError(f"{PARSE}: the skin model's import or registry entry was not found")
    text = text.replace(PARSE_IMPORT, "", 1).replace(PARSE_MAP_OLD, PARSE_MAP_NEW, 1)
    return text.replace(PARSE_RETURN, PARSE_LAZY + PARSE_RETURN, 1)


def patch(root: Path = VENDOR) -> list[str]:
    """Apply both edits, or neither when either anchor is missing; returns the files changed."""
    edits = []
    for name, edit in ((CONFIG, patch_config), (PARSE, patch_parse)):
        text = (root / name).read_text(encoding="utf-8")
        edits.append((name, text, edit(text)))   # raises before anything is written
    changed = []
    for name, text, new in edits:
        if new != text:
            (root / name).write_text(new, encoding="utf-8")
            changed.append(name)
    return changed


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else VENDOR
    try:
        changed = patch(root)
    except (OSError, PatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("patched: " + (", ".join(changed) if changed else "nothing to do"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
