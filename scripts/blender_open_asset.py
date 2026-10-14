"""Open an isolated GLB editing copy in Blender and save a resumable workspace.

Use --source source.glb --workspace model.blend --status status.json after --.
The user exports a GLB manually; this script never rewrites the source model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def prepare(source, workspace):
    import bpy

    if workspace.exists():
        bpy.ops.wm.open_mainfile(filepath=str(workspace), use_scripts=False)
    else:
        # --factory-startup already isolated this process. Resetting factory
        # settings again invalidates the interactive window context mid-script
        # (the glTF importer needs context.object when creating bone displays).
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=True)
        if not any(obj.type == "MESH" for obj in bpy.context.scene.objects):
            raise ValueError("Blender imported no mesh")
        bpy.ops.wm.save_as_mainfile(filepath=str(workspace), check_existing=False)
    return {"state": "ready", "blender_version": bpy.app.version_string}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.source, args.workspace)
    except Exception as exc:
        result = {"state": "error", "error": "Blender could not open this editing copy. Check its console or re-export an uncompressed GLB and retry.",
                  "detail": str(exc)[-4000:]}
        raise
    finally:
        temporary = args.status.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(args.status)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
