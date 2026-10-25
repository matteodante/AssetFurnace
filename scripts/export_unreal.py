#!/usr/bin/env python3
"""Turn a finished GLB into files Unreal Engine 5.8 imports with no touch-ups.

    python scripts/export_unreal.py MODEL.glb --kind prop --size-cm 60 --measure longest
    python scripts/export_unreal.py RIGGED.glb --kind creature --size-cm 180 --animations IDLE.glb WALK.glb
    python scripts/export_unreal.py PART.glb --kind equipment --body RIGGED.glb --size-cm 180
    python scripts/export_unreal.py ... --check-in-unreal      # then import it in Unreal and check

Writes one folder (default `output/unreal/<Name>__<time>/`) holding:
- `SM_<Name>.glb` or `SK_<Name>.glb`: centimetre-true once imported, on the ground,
  textures `T_<Name>_BaseColor|N|ORM`, material `MI_<Name>`, LODs and collision for
  objects, a `root` bone and Mannequin bone names for humanoids, every animation;
- `<mesh>.unreal.json`: how to import it and what Unreal should measure afterwards;
- `<mesh>.provenance.json`: the licences of everything it was made from;
- `T_<Name>_Tint.png` and `<mesh>.materials.json`, only when a tint mask is named (`--tint`) or
  lies beside the model as `T_<model>_Tint.png` with its record. The chain makes none (one
  material per piece), so such a mask is one you made yourself. `--tint none` ships none even then.

`--material` gives the piece its one material (`metal`, `leather`, `cloth`...): its roughness and
metal go into the ORM (`image_to_3dlab/piece_materials.py`).

`--size-cm` is the real height (`--measure height`, the default) or longest side: a
generated model has no real size of its own. Equipment takes its size, facing and
ground from the body it was cut from (`--body`), so it lands on that body in Unreal and
shares its Skeleton. An object's baked LODs (`<name>_LOD1.glb`... beside `_LOD0`) are
picked up by themselves.

`export_unreal(...)` is the same thing as a Python call, for the batch runner's
`export_unreal` step. The design and its measurements are in
`docs/plans/2026-10-15-unreal-profile-design.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import piece_materials  # noqa: E402
from image_to_3dlab import unreal_profile as up  # noqa: E402
from image_to_3dlab.glb_turn import parse_glb  # noqa: E402

BLENDER_SCRIPT = REPO / "scripts" / "blender_export_unreal.py"
OUTPUT_ROOT = REPO / "output" / "unreal"
BLENDER_TIMEOUT = 3600  # seconds; a reshape takes under a minute, a hung Blender would block forever


def read_grip(path: Path) -> dict:
    """A grip record (`slot_parts/grip.py`): its point and axis on the model."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a grip record: {exc}") from None
    if not isinstance(record, dict) or not record.get("point"):
        raise ValueError(f"{path} has no grip yet: place it with slot_parts.py grip")
    return record


def default_collision(kind: str) -> str:
    """Objects get a box; skinned meshes get Unreal's own physics asset instead."""
    return "box" if kind == "prop" else "none"


def default_base(source: Path, kind: str, animations: list[Path] | None = None) -> str:
    """A creature is named after its model, not after `_rigged` or the move it holds."""
    prefix = up.model_prefix([Path(source).stem, *(Path(p).stem for p in animations or [])])
    return up.base_name(prefix if kind == "creature" and prefix else Path(source).stem)


def plan(source: Path, *, kind: str, size_cm: float, measure: str = "height", name: str | None = None,
         out_dir: Path | None = None, lods: list[Path] | None = None, collision: str | None = None,
         body: Path | None = None, animations: list[Path] | None = None, tint: Path | None = None,
         forward: tuple[float, float, float] | None = None, body_name: str | None = None,
         material: str | None = None, two_sided: bool | None = None, grip: dict | None = None,
         opacity_mask: bool = False) -> dict[str, Any]:
    """Check the request and work out names, files and the Blender job. No side effects.

    `opacity_mask`: a fitted base body (`body_masked.glb` from characters in pieces), whose vertex alpha hides
    the skin under the pieces; only then is that alpha exported and its materials made MASK.

    `grip` (a weapon's or shield's, `image_to_3dlab/slot_parts/grip.py`: its `point` and `axis` on
    the model, glTF frame) puts the pivot there, the blade or head along +Z, and a `grip` socket
    on it; the size is then the real length (`measure` longest)."""
    source = Path(source)
    material = piece_materials.check(material) if material is not None else None
    kind = up.KIND_ALIASES.get(kind, kind)
    if kind not in up.KINDS:
        raise ValueError(f"--kind must be one of {', '.join(up.KINDS)}")
    if measure not in up.MEASURES:
        raise ValueError(f"--measure must be one of {', '.join(up.MEASURES)}")
    if size_cm is None or size_cm <= 0:
        raise ValueError("--size-cm is required: a generated model has no real size of its own")
    collision = collision or default_collision(kind)
    if kind == "prop" and collision not in up.COLLISIONS:
        raise ValueError(f"--collision must be one of {', '.join(up.COLLISIONS)}")
    if kind == "equipment" and body is None:
        raise ValueError("equipment needs --body, the rigged GLB it was cut from")
    if kind != "equipment" and body is not None:
        raise ValueError("--body is only for equipment")
    if kind == "prop" and animations:
        raise ValueError("objects carry no animations")
    if grip is not None:
        if kind != "prop":
            raise ValueError("a grip is a weapon's or shield's: an object (--kind prop)")
        numbers = [grip.get(key) for key in ("point", "axis")]
        if not all(isinstance(v, list | tuple) and len(v) == 3 and all(isinstance(c, int | float) for c in v)
                   for v in numbers):
            raise ValueError("the grip needs its point and axis on the model (slot_parts/grip.py)")
        grip = {"point": [float(c) for c in numbers[0]], "axis": [float(c) for c in numbers[1]]}
    for path in [source, *(lods or []), *(animations or []), *([body] if body else []), *([tint] if tint else [])]:
        if not Path(path).is_file():
            raise ValueError(f"{path} does not exist")
    if tint is False:
        tint = None   # asked for none: no mask, even if one lies beside the model
    elif tint is None:
        # A mask left beside the model as `T_<model>_Tint.png` is shipped with it;
        # only that one is taken, never a mask that belongs to another model in the same folder.
        own = source.with_name(f"T_{source.stem}_Tint.png")
        tint = own if own.is_file() else None
    record = source.with_name(source.stem + ".materials.json")
    animations = [Path(p) for p in (animations or [])]
    family = _family(source) if kind == "creature" else None
    if forward is None and family and family.get("frame"):
        # A creature that is not a humanoid faces where its family file says (Blender
        # works out a humanoid's facing itself); a drake facing half-way between two axes
        # would otherwise come into Unreal turned.
        forward = up.gltf_to_blender(family["frame"]["forward"])
    prefix = up.model_prefix([source.stem, *(p.stem for p in animations)])
    base = up.base_name(name) if name else default_base(source, kind, animations)
    names = up.asset_names(kind, base)
    if kind == "prop" and lods is None:
        lods = up.sibling_lods(source)
    lods = [Path(p) for p in (lods or [])] if kind == "prop" else []
    out_dir = Path(out_dir) if out_dir else OUTPUT_ROOT / f"{base}__{time.strftime('%Y%m%d-%H%M%S')}"
    work = out_dir / ".work"
    inputs = {Path(p).resolve() for p in [source, *lods, *animations, *([body] if body else []), *([tint] if tint else [])]}
    for written in (out_dir / f"{names['mesh']}{suffix}" for suffix in (".glb", ".unreal.json", ".provenance.json")):
        if written.resolve() in inputs:
            raise ValueError(f"the export would overwrite its own input {written.name}; choose another --out")
    job = {
        "source": str(source.resolve()), "kind": kind, "base": base, "size_cm": float(size_cm), "measure": measure,
        "lods": [str(p.resolve()) for p in lods], "collision": collision if kind == "prop" else "none",
        "body": str(Path(body).resolve()) if body else None,
        "grip": grip,
        "animations": [{"path": str(p.resolve()), "name": up.clip_name(p.stem, prefix, base)} for p in animations],
        "source_animation": up.clip_name(source.stem, prefix, base),
        "forward": list(forward) if forward else [0.0, -1.0, 0.0],
        "family": (family or {}).get("family"),
        "two_sided": up.two_sided(base, two_sided),
        "opacity_mask": bool(opacity_mask),
        "out": str((work / "blender.glb").resolve()), "report": str((work / "blender.json").resolve()),
    }
    return {"names": names, "out_dir": out_dir, "work": work, "job": job, "material": material,
            "glb": out_dir / f"{names['mesh']}.glb", "sidecar": out_dir / f"{names['mesh']}.unreal.json",
            "provenance": out_dir / f"{names['mesh']}.provenance.json",
            "tint": (out_dir / f"{names['tint']}.png") if tint else None, "tint_source": tint,
            "materials_record": record if record.is_file() else None,
            "inputs": [source, *lods, *animations, *([Path(body)] if body else [])],
            "skeleton_from": "SK_" + up.base_name(body_name or up.model_prefix([Path(body).stem])) if body else None}


def _family(source: Path) -> dict | None:
    """The creature family file the clip step wrote beside a rig, if any."""
    from image_to_3dlab import creature_family

    try:
        return creature_family.read_family(source)
    except (OSError, ValueError):
        return None


def facing_bones(family: dict | None, renamed: dict[str, str]) -> list[str] | None:
    """The two bones Unreal can tell a creature's facing by, under their exported names:
    hips to head, or a serpent's tail to its head. None when the family has neither."""
    if not family:
        return None
    if family.get("spine") and family.get("neck"):
        pair = [family["spine"][0], family["neck"][-1]]
    elif len(family.get("body") or []) >= 2:
        pair = [family["body"][-1], family["body"][0]]
    else:
        return None
    return [renamed.get(b, b) for b in pair]


def blender_command(blender: Path, job_path: Path) -> list[str]:
    return [str(blender), "-b", "--factory-startup", "--python-exit-code", "1",
            "--python", str(BLENDER_SCRIPT), "--", str(job_path)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_of(record: dict | None) -> str | None:
    record = record or {}
    output = record.get("output") if isinstance(record.get("output"), dict) else {}
    return record.get("classification") or output.get("classification")


def _license_of(record: dict | None) -> Any:
    record = record or {}
    output = record.get("output") if isinstance(record.get("output"), dict) else {}
    return record.get("license") or output.get("license")


def _distinct(values) -> list:
    seen = []
    for value in values:
        if value is not None and value not in seen:
            seen.append(value)
    return seen


def provenance(inputs: list[Path], glb: Path) -> dict[str, Any]:
    """The licence record that travels with the export: every input and its own record."""
    sources = []
    for path in inputs:
        record_path = up.find_provenance(path)
        record = None
        if record_path is not None:
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                record = {"unreadable": str(record_path)}
        sources.append({"file": Path(path).name, "sha256": _sha256(path),
                        "record_file": record_path.name if record_path else None, "record": record})
    main = sources[0]["record"] or {}
    output = main.get("output") if isinstance(main.get("output"), dict) else {}
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "scripts/export_unreal.py (Unreal export profile)",
        "input": {"name": Path(inputs[0]).name, "sha256": sources[0]["sha256"], "source": "derived"},
        "output": {"name": glb.name, "sha256": _sha256(glb)},
        # Generation records keep the class under output; derived records at the top.
        # The export is as restricted as the strictest of all its inputs.
        "license": main.get("license") or output.get("license"),
        "classification": up.strictest_class(_class_of(s["record"]) for s in sources),
        "licenses": _distinct(_license_of(s["record"]) for s in sources),
        "sources": sources,
        "note": "Reshaped and renamed for Unreal; no model was run. Licences are those of the sources."
                + ("" if sources[0]["record"] else " No record was found for the main source: its licence is unknown."),
    }


def export_unreal(source: Path, *, kind: str, size_cm: float, measure: str = "height", name: str | None = None,
                  out_dir: Path | None = None, lods: list[Path] | None = None, collision: str | None = None,
                  body: Path | None = None, animations: list[Path] | None = None, tint: Path | None = None,
                  forward: tuple[float, float, float] | None = None, body_name: str | None = None,
                  material: str | None = None, blender: Path | None = None, run=subprocess.run,
                  log=lambda message: print(message), two_sided: bool | None = None,
                  grip: dict | None = None, opacity_mask: bool = False) -> dict[str, Any]:
    """Export one asset for Unreal. Returns the paths written, the problems found and
    the sidecar. Raises ValueError for a bad request and RuntimeError when Blender fails."""
    from image_to_3dlab.blender import find_blender, missing_help

    p = plan(source, kind=kind, size_cm=size_cm, measure=measure, name=name, out_dir=out_dir, lods=lods,
             collision=collision, body=body, animations=animations, tint=tint, forward=forward,
             body_name=body_name, material=material, two_sided=two_sided, grip=grip,
             opacity_mask=opacity_mask)
    kind = p["job"]["kind"]
    blender = blender or find_blender()
    if blender is None:
        raise RuntimeError(missing_help())
    p["work"].mkdir(parents=True, exist_ok=True)
    job_path = p["work"] / "job.json"
    job_path.write_text(json.dumps(p["job"], indent=1), encoding="utf-8")
    log(f"export_unreal: {p['names']['mesh']} from {Path(source).name} in Blender")
    for stale in (Path(p["job"]["out"]), Path(p["job"]["report"])):
        stale.unlink(missing_ok=True)  # a kept .work from a failed run must not pass for this one
    with (p["work"] / "blender.log").open("wb") as handle:
        try:
            run(blender_command(blender, job_path), stdout=handle, stderr=subprocess.STDOUT, check=False, cwd=REPO,
                timeout=BLENDER_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Blender was still working after {BLENDER_TIMEOUT // 60} minutes and was stopped; "
                               f"see {p['work'] / 'blender.log'}") from None
    report_path = Path(p["job"]["report"])
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "ok": False, "error": f"Blender wrote no report; see {p['work'] / 'blender.log'}"}
    if not report.get("ok"):
        raise RuntimeError(f"Blender could not reshape the model: {report.get('error')}")
    stamp = {"kind": kind, "size_cm": size_cm, "measure": measure, "profile": up.SIDECAR_SCHEMA}
    data, textures = up.finalize_glb(Path(p["job"]["out"]).read_bytes(), stamp=stamp, material=p["material"])
    if p["job"]["opacity_mask"] and report.get("vertex_colour"):  # a fitted base body: hide the skin under the armour
        data, masked = up.mask_from_vertex_alpha(data)
        report["masked_materials"] = masked
    p["glb"].write_bytes(data)
    if p["tint"]:
        from PIL import Image

        with Image.open(p["tint_source"]) as image:
            image.convert("RGB").save(p["tint"], format="PNG")
    problems = up.check_document(parse_glb(data)[0], p["names"])
    sidecar = up.sidecar(kind=kind, names=p["names"], glb=p["glb"].name, size_cm=size_cm, measure=measure,
                         lods=report.get("lods", 1) if kind == "prop" else (3 if kind == "creature" else 1),
                         collision=p["job"]["collision"] if kind == "prop" else None,
                         humanoid=bool(report.get("humanoid")), skeleton_from=p["skeleton_from"],
                         animations=report.get("animations", []), tint=p["tint"].name if p["tint"] else None,
                         problems=problems, orm=any(m.get("orm") for m in textures["materials"].values()),
                         grip=bool(p["job"]["grip"]),
                         facing_bones=facing_bones(_family(Path(source)) if kind == "creature" else None,
                                                   report.get("bones") or {}),
                         opacity_mask="vertex_alpha" if report.get("masked_materials") else None)
    if p["materials_record"]:
        record = p["out_dir"] / f"{p['names']['mesh']}.materials.json"
        shutil.copyfile(p["materials_record"], record)
        sidecar["materials_record"] = record.name
    sidecar["blender"] = {k: report.get(k) for k in ("blender", "scale", "yaw_deg", "offset_m", "dims_before_m",
                                                      "box_after_m", "removed", "bones", "root_head_m")}
    sidecar["materials"] = textures["materials"]
    if p["material"]:
        sidecar["material"] = p["material"]
    p["sidecar"].write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    p["provenance"].write_text(json.dumps(provenance(p["inputs"], p["glb"]), indent=1), encoding="utf-8")
    if not problems:  # kept when something is wrong, to look at what Blender wrote
        shutil.rmtree(p["work"], ignore_errors=True)
    for problem in problems:
        log(f"export_unreal: problem: {problem}")
    return {"ok": not problems, "out_dir": str(p["out_dir"]), "glb": str(p["glb"]), "sidecar": str(p["sidecar"]),
            "provenance": str(p["provenance"]), "tint": str(p["tint"]) if p["tint"] else None,
            "problems": problems, "unreal": sidecar}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the finished GLB (for a creature: the rigged one)")
    parser.add_argument("--kind", required=True, choices=[*up.KINDS, *up.KIND_ALIASES])
    parser.add_argument("--size-cm", type=float, required=True, help="real size in centimetres")
    parser.add_argument("--measure", choices=up.MEASURES, default="height",
                        help="what --size-cm measures: the height (default) or the longest side")
    parser.add_argument("--name", help="asset name (default: from the file name)")
    parser.add_argument("--out", type=Path, help="output folder (default output/unreal/<Name>__<time>/)")
    parser.add_argument("--lods", type=Path, nargs="*", default=None,
                        help="objects: LOD1, LOD2... GLBs (default: <name>_LODn.glb beside a _LOD0 source)")
    parser.add_argument("--collision", choices=up.COLLISIONS,
                        help="objects: box (default), convex, or unreal (Interchange makes its own convex hull)")
    parser.add_argument("--body", type=Path, help="equipment: the rigged body GLB it belongs to")
    parser.add_argument("--body-name", help="equipment: the body's --name, if it was exported with one")
    parser.add_argument("--animations", type=Path, nargs="*", default=[], help="creatures: more animated GLBs")
    parser.add_argument("--tint", type=Path,
                        help="a tint mask to ship as T_<Name>_Tint.png (default: the model's own T_<model>_Tint.png, "
                             "if there is one; none: no tint mask)")
    parser.add_argument("--forward", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="non-humanoid creatures: which way the model faces in Blender (default 0 -1 0)")
    parser.add_argument("--two-sided", action="store_const", const=True, default=None,
                        help="draw both faces of every triangle (default: cloth two sided, the rest one sided)")
    parser.add_argument("--grip", type=Path, help="a weapon or shield: its grip record (grip.json): the pivot "
                                                   "there, a grip socket, the blade up")
    parser.add_argument("--opacity-mask", action="store_true",
                        help="a fitted base body (body_masked.glb): its vertex alpha hides the skin under the pieces")
    parser.add_argument("--material", choices=piece_materials.MATERIALS,
                        help="the one material of this piece: sets its roughness and metal (game materials)")
    parser.add_argument("--blender", type=Path, help="Blender executable (default: found like Finish finds it)")
    parser.add_argument("--check-in-unreal", action="store_true",
                        help="then import it in Unreal 5 with scripts/unreal_import_check.py")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_in_unreal and up.KIND_ALIASES.get(args.kind, args.kind) == "equipment":
        print("export_unreal: equipment is checked on its body's Skeleton, so check both exports together:\n"
              "    python scripts/unreal_import_check.py BODY_EXPORT_DIR PART_EXPORT_DIR", file=sys.stderr)
        return 2
    try:
        result = export_unreal(args.source, kind=args.kind, size_cm=args.size_cm, measure=args.measure,
                               name=args.name, out_dir=args.out, lods=args.lods, collision=args.collision,
                               body=args.body, body_name=args.body_name, animations=args.animations,
                               tint=False if args.tint is not None and str(args.tint).lower() == "none" else args.tint,
                               forward=tuple(args.forward) if args.forward else None, material=args.material,
                               blender=args.blender, two_sided=args.two_sided, opacity_mask=args.opacity_mask,
                               **({"grip": read_grip(args.grip)} if args.grip else {}))
    except (ValueError, RuntimeError) as exc:
        print(f"export_unreal: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ("ok", "out_dir", "glb", "sidecar", "problems")}, indent=1))
    if args.check_in_unreal:
        from unreal_import_check import check

        return 0 if check([Path(result["out_dir"])]) and result["ok"] else 1
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
