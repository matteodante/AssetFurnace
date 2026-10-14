"""Check a GLB for the guided editor and save reproducible S0/RIG0 evidence.

Usage: python scripts/inspect_edit_asset.py input.glb --output-dir output/edit-probe/run
The output directory must not exist. This does not enable editing in the web app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.edit_probe import run_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_probe(args.source, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Inspection failed: {exc}\n")
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "report.json"),
                      "production_editing_ready": False, "error": report.get("error")}))
    return 0 if report["status"] == "inspection_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
