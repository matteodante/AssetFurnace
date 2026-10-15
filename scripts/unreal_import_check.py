"""Import exported assets into Unreal with Interchange and check them, with no editor window.

    python scripts/unreal_import_check.py output/unreal/SM_Barrel output/unreal/SK_Warrior ...

Each folder is one `scripts/export_unreal.py` result (a GLB and its `.unreal.json`). With
no `--project`, a blank project is made in `output/unreal-check/` (git-ignored), with the
engine's own Mannequin template content copied in so humanoids can be checked against
Manny. Unreal runs twice:

1. `UnrealEditor-Cmd` (no rendering): imports every GLB the way the sidecar says (an
   equipment piece onto its body's Skeleton, skeletal LODs reduced by Unreal), measures
   size, base, LODs, collision, names, ORM colour space, and for a humanoid asks the IK
   Rig to recognise the skeleton and retargets Manny's idle and walk onto it, measuring
   how far the arms hang and the feet step against Manny's own.
2. `UnrealEditor` off screen: stands everything in a row in the template's daylight,
   lit from the camera's side, and photographs it twice: everyone in the idle, then
   everyone at the widest step of the walk.

Writes `unreal_check.json`, `unreal_check.png` and `unreal_check_walk.png` into every
folder; exits non-zero when any check fails. The same file is the script Unreal runs
(`--inside`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = REPO / "output" / "unreal-check" / "AFCheck" / "AFCheck.uproject"
MANNY_TEMPLATE = Path("Templates/TemplateResources/High/Characters/Content/Mannequins")
MANNY_MESH = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"
MANNY_IDLE = "/Game/Characters/Mannequins/Anims/Unarmed/MM_Idle"
MANNY_WALK = "/Game/Characters/Mannequins/Anims/Unarmed/Walk/MF_Unarmed_Walk_Fwd"
TEMPLATE_LEVEL = "/Engine/Maps/Templates/Template_Default"
MANNY_WIDTH_CM = 120
CONTENT_ROOT = "/Game/AF"
PROJECT_PLUGINS = ("PythonScriptPlugin", "EditorScriptingUtilities", "Interchange", "InterchangeEditor", "IKRig")
CAPTURE_SIZE = (1280, 720)
SLOW = 0.01  # play rate that holds a pose while still evaluating it
CLOSEUP_FOV = 35.0


# --- outside Unreal ---------------------------------------------------------------------

def project_document() -> dict:
    """A content-only project with just what the check needs."""
    return {"FileVersion": 3, "EngineAssociation": "", "Category": "",
            "Description": "AssetFurnace Unreal import check (generated, safe to delete)",
            "Plugins": [{"Name": name, "Enabled": True} for name in PROJECT_PLUGINS]}


def ensure_project(project: Path, engine_root: Path, need_manny: bool) -> Path:
    project.parent.mkdir(parents=True, exist_ok=True)
    if not project.is_file():
        project.write_text(json.dumps(project_document(), indent=1), encoding="utf-8")
    target = project.parent / "Content" / "Characters" / "Mannequins"
    if need_manny and not target.is_dir():
        source = engine_root / MANNY_TEMPLATE
        if not source.is_dir():
            raise SystemExit(f"Manny's template content is not in {source}; humanoids cannot be checked")
        shutil.copytree(source, target)
    return project


def ordered(exports: list[dict]) -> list[dict]:
    """Bodies before the equipment that goes on them."""
    return sorted(exports, key=lambda e: e["sidecar"]["kind"] == "equipment")


def editor_commands(editor: Path, project: Path, job: Path) -> list[list[str]]:
    """The two Unreal runs: import and measure without rendering, then render off screen."""
    script = Path(__file__).resolve().as_posix()
    cmd = editor.with_name("UnrealEditor-Cmd.exe")
    common = ["-unattended", "-nop4", "-nosplash", "-nosound"]
    return [
        [str(cmd), str(project), "-run=pythonscript", f"-script={script} --inside import {job.as_posix()}",
         "-NullRHI", "-stdout", *common],
        # Texture streaming would still be on low mips when the picture is taken.
        [str(editor), str(project), f"-ExecCmds=py {script} --inside capture {job.as_posix()}", "-RenderOffscreen",
         "-NoTextureStreaming", *common],
    ]


def content_folder(project: Path, mesh: str) -> Path:
    """Where one asset's imports live on disk: `<project>/Content/AF/<mesh>`."""
    return project.parent / "Content" / CONTENT_ROOT.removeprefix("/Game/") / mesh


def load_exports(folders: list[Path]) -> list[dict]:
    exports = []
    for folder in folders:
        sidecars = sorted(Path(folder).glob("*.unreal.json"))
        if len(sidecars) != 1:
            raise SystemExit(f"{folder}: expected one .unreal.json, found {len(sidecars)}")
        sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
        exports.append({"dir": str(Path(folder).resolve()), "glb": str((Path(folder) / sidecar["glb"]).resolve()),
                        "sidecar": sidecar})
    return ordered(exports)


def stage_for(exports: list[dict], measured: dict, manny: bool) -> dict:
    """Where the picture run stands everyone and the camera: Manny first, then each
    asset in a row (equipment is worn by its body, not stood apart)."""
    sys.path.insert(0, str(REPO))
    from image_to_3dlab.unreal_profile import closeup_camera, stage_layout

    order, widths, heights, height = [], [], [], 180.0
    if manny:
        order.append("Manny")
        widths.append(MANNY_WIDTH_CM)
        heights.append(180.0)
    for export in exports:
        side = export["sidecar"]
        got = measured.get("assets", {}).get(side["mesh"], {})
        if side["kind"] == "equipment" or "error" in got:
            continue
        bounds = got.get("bounds_cm") or [100, 100, 100]
        order.append(side["mesh"])
        widths.append(max(bounds[0], bounds[1]) + 60)
        heights.append(bounds[2])
        height = max(height, bounds[2])
    stage = stage_layout(widths, height, sun=measured.get("sun") or (0.0, 0.0, -1.0))
    # One close-up per asset too: in the wide shot a figure is a couple of hundred pixels.
    closeups = [{"name": name, "fov": CLOSEUP_FOV,
                 "camera": closeup_camera(where, stage["camera_yaw"], width - 60, tall, fov=CLOSEUP_FOV)}
                for name, where, width, tall in zip(order, stage["positions"], widths, heights) if name != "Manny"]
    return dict(stage, order=order, closeups=closeups)


def closeup_path(picture: Path | str, mesh: str) -> Path:
    """Where the picture run saves one asset's close-up, beside the wide shot."""
    picture = Path(picture)
    return picture.with_name(f"{picture.stem}-{mesh}{picture.suffix}")


def write_results(exports: list[dict], measured: dict, picture: Path | None, picture_walk: Path | None = None) -> bool:
    sys.path.insert(0, str(REPO))
    from image_to_3dlab.unreal_profile import evaluate_unreal

    all_ok = True
    for export in exports:
        mesh = export["sidecar"]["mesh"]
        got = measured.get("assets", {}).get(mesh, {"error": "not imported"})
        results = evaluate_unreal(got, export["sidecar"]) if "error" not in got else [
            {"check": "import", "ok": False, "detail": got["error"]}]
        ok = all(r["ok"] for r in results)
        all_ok &= ok
        folder = Path(export["dir"])
        pictures = {}
        closeup = closeup_path(picture, mesh) if picture else None
        for name, source in (("unreal_check.png", picture), ("unreal_check_walk.png", picture_walk),
                             ("unreal_check_close.png", closeup)):
            (folder / name).unlink(missing_ok=True)  # never leave an older run's picture looking current
            if source and source.is_file():
                shutil.copyfile(source, folder / name)
                pictures[name] = name
        if not pictures.get("unreal_check.png") and measured.get("capture_error"):
            results.append({"check": "picture", "ok": False, "detail": measured["capture_error"].splitlines()[-1]})
            ok = False
            all_ok = False
        (folder / "unreal_check.json").write_text(json.dumps({
            "ok": ok, "engine": measured.get("engine"), "mesh": mesh, "results": results, "measured": got,
            "picture": pictures.get("unreal_check.png"), "picture_walk": pictures.get("unreal_check_walk.png"),
            "picture_close": pictures.get("unreal_check_close.png"),
        }, indent=1), encoding="utf-8")
        print(f"{'PASS' if ok else 'FAIL'} {mesh}")
        for r in results:
            print(f"  {'ok ' if r['ok'] else 'NO '} {r['check']}: {r['detail']}")
    return all_ok


def check(folders: list[Path], project: Path | None = None, editor: Path | None = None,
          timeout: int = 1800, run=subprocess.run) -> bool:
    sys.path.insert(0, str(REPO))
    from image_to_3dlab.unreal_profile import find_unreal_editor

    editor = editor or find_unreal_editor()
    if editor is None:
        raise SystemExit("Unreal Engine 5 was not found. Install it, or set I2L_UNREAL to UnrealEditor.exe.")
    exports = load_exports(folders)
    need_manny = any(e["sidecar"]["expect"].get("manny_compatible") for e in exports)
    project = ensure_project(project or DEFAULT_PROJECT, editor.parents[3], need_manny)
    work = project.parent / "Saved" / "AssetFurnace"
    work.mkdir(parents=True, exist_ok=True)
    for export in exports:
        # Unreal, run headless, does not reliably replace what an earlier check left
        # (the IK rigs above all), so each asset's own folder starts empty. Only the
        # check's content root is touched.
        shutil.rmtree(content_folder(project, export["sidecar"]["mesh"]), ignore_errors=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = work / f"job-{stamp}.json"
    measured_path = work / f"measured-{stamp}.json"
    picture, picture_walk = work / f"capture-{stamp}.png", work / f"capture-{stamp}-walk.png"
    details = {"exports": exports, "measured": str(measured_path), "picture": str(picture),
               "picture_walk": str(picture_walk), "content_root": CONTENT_ROOT, "manny": need_manny}
    job.write_text(json.dumps(details, indent=1), encoding="utf-8")
    for stage, command in zip(("import", "capture"), editor_commands(editor, project, job)):
        if stage == "capture":
            if not measured_path.is_file():
                break
            # The import run found the sizes and where the sun is; the picture run needs both.
            details["stage"] = stage_for(exports, json.loads(measured_path.read_text(encoding="utf-8")), need_manny)
            job.write_text(json.dumps(details, indent=1), encoding="utf-8")
        print("unreal:", " ".join(command[:3]), "...", flush=True)
        with (work / f"unreal-{stamp}-{stage}.log").open("wb") as handle:
            run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if not measured_path.is_file():
        raise SystemExit(f"Unreal wrote no measurements; see the logs in {work}")
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    return write_results(exports, measured, picture, picture_walk)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path, help="export_unreal.py output folders")
    parser.add_argument("--project", type=Path, default=None, help="a .uproject to import into")
    parser.add_argument("--editor", type=Path, default=None, help="UnrealEditor.exe (else I2L_UNREAL or the newest UE 5)")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    return 0 if check(args.folders, args.project, args.editor, args.timeout) else 1


# --- inside Unreal ----------------------------------------------------------------------

def _inside(stage: str, job_path: str) -> None:  # pragma: no cover - runs in Unreal's Python
    import traceback

    import unreal

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    measured_path = Path(job["measured"])
    load = unreal.EditorAssetLibrary.load_asset

    def assets_in(folder):
        return [load(str(p).split(".")[0]) for p in unreal.EditorAssetLibrary.list_assets(folder, recursive=True)]

    def folder_of(export):
        return f"{job['content_root']}/{export['sidecar']['mesh']}"

    def mesh_of(export):
        wanted = export["sidecar"]["mesh"]
        found = [a for a in assets_in(folder_of(export))
                 if isinstance(a, (unreal.StaticMesh, unreal.SkeletalMesh)) and a.get_name() == wanted]
        return found[0] if found else None

    def import_one(export):
        folder = folder_of(export)
        if unreal.EditorAssetLibrary.does_directory_exist(folder):
            unreal.EditorAssetLibrary.delete_directory(folder)
        side = export["sidecar"]
        if side["import"]["reuse_skeleton"]:
            body = next(e for e in job["exports"] if e["sidecar"]["mesh"] == side["skeleton_from"])
            skeleton = mesh_of(body).get_editor_property("skeleton")
            pipeline_path = f"{job['content_root']}/Pipelines/P_{side['mesh']}"
            if unreal.EditorAssetLibrary.does_asset_exist(pipeline_path):
                unreal.EditorAssetLibrary.delete_asset(pipeline_path)
            pipeline = unreal.EditorAssetLibrary.duplicate_asset(
                "/Interchange/Pipelines/DefaultGLTFAssetsPipeline", pipeline_path)
            pipeline.get_editor_property("common_skeletal_meshes_and_animations_properties").set_editor_property(
                "skeleton", skeleton)
            unreal.EditorAssetLibrary.save_loaded_asset(pipeline)
            params = unreal.ImportAssetParameters()
            params.is_automated = True
            params.override_pipelines = [
                unreal.SoftObjectPath(pipeline.get_path_name()),
                unreal.SoftObjectPath("/Interchange/Pipelines/DefaultGLTFPipeline.DefaultGLTFPipeline")]
            source = unreal.InterchangeManager.create_source_data(export["glb"])
            unreal.InterchangeManager.get_interchange_manager_scripted().import_asset(folder, source, params)
        else:
            task = unreal.AssetImportTask()
            task.filename = export["glb"]
            task.destination_path = folder
            task.automated = True
            task.save = True
            task.replace_existing = True
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        if side.get("tint"):
            import_tint(export, folder)
        name_animations(export)
        unreal.EditorAssetLibrary.save_directory(folder)

    def import_tint(export, folder):
        """The material step's tint mask: data, not colour, so linear and uncompressed in hue."""
        task = unreal.AssetImportTask()
        task.filename = str(Path(export["dir"]) / export["sidecar"]["tint"])
        task.destination_path = folder
        task.automated = True
        task.save = True
        task.replace_existing = True
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        for texture in [a for a in assets_in(folder) if isinstance(a, unreal.Texture2D)]:
            if texture.get_name() == Path(export["sidecar"]["tint"]).stem:
                texture.set_editor_property("srgb", False)
                texture.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
                unreal.EditorAssetLibrary.save_loaded_asset(texture)

    def name_animations(export):
        """Interchange calls a glTF animation `<mesh><animation>`; give it back its own name."""
        for animation in [a for a in assets_in(folder_of(export)) if isinstance(a, unreal.AnimSequence)]:
            for wanted in export["sidecar"]["animations"]:
                current = animation.get_path_name().split(".")[0]
                if animation.get_name() != wanted and animation.get_name().endswith(wanted):
                    unreal.EditorAssetLibrary.rename_asset(current, current.rsplit("/", 1)[0] + "/" + wanted)

    def measure(export):
        side = export["sidecar"]
        mesh = mesh_of(export)
        if mesh is None:
            return {"error": f"Interchange did not create {side['mesh']}"}
        everything = assets_in(folder_of(export))
        got = {"class": mesh.get_class().get_name(), "names": sorted({a.get_name() for a in everything})}
        textures = [a for a in everything if isinstance(a, unreal.Texture2D)]
        got["orm_srgb"] = [bool(t.get_editor_property("srgb")) for t in textures if t.get_name().endswith("_ORM")]
        tint = [bool(t.get_editor_property("srgb")) for t in textures if t.get_name().endswith("_Tint")]
        if tint:
            got["tint_srgb"] = any(tint)
        if isinstance(mesh, unreal.StaticMesh):
            box = mesh.get_bounding_box()
            got["bounds_cm"] = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z]
            got["min_z"] = box.min.z
            got["lods"] = mesh.get_num_lods()
            geometry = mesh.get_editor_property("body_setup").get_editor_property("agg_geom")
            got["collision"] = {k: len(geometry.get_editor_property(k)) for k in ("box_elems", "convex_elems")}
        else:
            subsystem = unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem)
            wanted = side["import"]["generate_skeletal_lods"]
            if wanted > 1:
                subsystem.regenerate_lod(mesh, wanted, False, False)
                unreal.EditorAssetLibrary.save_loaded_asset(mesh)
            got["lods"] = subsystem.get_lod_count(mesh)
            bounds = mesh.get_bounds()
            got["bounds_cm"] = [2 * bounds.box_extent.x, 2 * bounds.box_extent.y, 2 * bounds.box_extent.z]
            got["min_z"] = bounds.origin.z - bounds.box_extent.z
            skeleton = mesh.get_editor_property("skeleton")
            got["skeleton"] = skeleton.get_path_name().split(".")[0] if skeleton else None
            got["new_skeletons"] = sum(isinstance(a, unreal.Skeleton) for a in everything)
            got["animations"] = sorted(a.get_name() for a in everything if isinstance(a, unreal.AnimSequence))
            try:
                reference = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
                root = unreal.AnimPoseExtensions.get_bone_pose(reference, "root", unreal.AnimPoseSpaces.WORLD).translation
                got["root_cm"] = [root.x, root.y, root.z]
            except Exception as exc:
                got["root_error"] = repr(exc)
            if side["expect"].get("facing"):
                try:
                    got["forward"] = facing(skeleton)
                except Exception as exc:
                    got["forward_error"] = repr(exc)
            if side["expect"].get("manny_compatible") and job["manny"]:
                got.update(retarget_idle(side, mesh))
        return got

    def retarget_idle(side, mesh):
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        folder = f"{job['content_root']}/{side['mesh']}/Retarget"
        manny = load(MANNY_MESH)
        rigs = []
        recognised = False
        def fresh(name, cls, factory):
            """A new asset even when an earlier check left one by that name."""
            path = f"{folder}/{name}"
            if unreal.EditorAssetLibrary.does_asset_exist(path):
                unreal.EditorAssetLibrary.delete_asset(path)
            asset = tools.create_asset(name, folder, cls, factory)
            if asset is None:
                raise RuntimeError(f"Unreal could not create {path}")
            return asset

        for name, skeletal in ((f"IK_{side['mesh']}_Manny", manny), (f"IK_{side['mesh']}", mesh)):
            rig = fresh(name, unreal.IKRigDefinition, unreal.IKRigDefinitionFactory())
            controller = unreal.IKRigController.get_controller(rig)
            controller.set_skeletal_mesh(skeletal)
            recognised = bool(controller.apply_auto_generated_retarget_definition())
            rigs.append(rig)
        retargeter = fresh(f"RTG_Manny_{side['mesh']}", unreal.IKRetargeter, unreal.IKRetargetFactory())
        controller = unreal.IKRetargeterController.get_controller(retargeter)
        controller.set_ik_rig(unreal.RetargetSourceOrTarget.SOURCE, rigs[0])
        controller.set_ik_rig(unreal.RetargetSourceOrTarget.TARGET, rigs[1])
        # A retargeter made from Python starts with no operations at all (UE 5.6+ runs
        # a stack of them), so it would copy the pelvis and nothing else.
        controller.add_default_ops()
        controller.assign_ik_rig_to_all_ops(unreal.RetargetSourceOrTarget.TARGET, rigs[1])
        controller.auto_map_chains(unreal.AutoMapChainType.FUZZY, True)
        # SkinTokens rigs in a T-pose, Manny stands in an A-pose: without lining the
        # retarget poses up, Manny's idle moves the arms relative to the wrong start
        # and they stay out horizontally.
        controller.auto_align_all_bones(unreal.RetargetSourceOrTarget.TARGET)
        sources = [MANNY_IDLE, MANNY_WALK]
        made = unreal.IKRetargetBatchOperation.duplicate_and_retarget(
            [unreal.EditorAssetLibrary.find_asset_data(path) for path in sources], manny, mesh, retargeter,
            "", "", f"AS_{side['mesh']}_", "")
        result = {"ik_recognised": recognised, "retarget_asset": None, "walk_asset": None}
        if not made:
            return dict(result, retarget="no animation was made")
        # The batch retarget writes beside nothing in particular (the content root) and
        # does not save: keep them with the character, on disk, for the picture run.
        kept = {}
        for asset, label in zip(made, ("Idle", "Walk")):
            path = f"{folder}/AS_{side['mesh']}_{label}"
            if unreal.EditorAssetLibrary.does_asset_exist(path):
                unreal.EditorAssetLibrary.delete_asset(path)
            unreal.EditorAssetLibrary.rename_asset(str(asset.package_name), path)
            kept[label] = path
        unreal.EditorAssetLibrary.save_directory(folder)
        result.update(retarget_asset=kept.get("Idle"), walk_asset=kept.get("Walk"))
        # Read the poses straight from the animations. Manny's idle hangs his arms by his
        # sides, and a T-pose here means the retarget moved nothing.
        manny_pose = arm_drop(load(MANNY_IDLE))
        mine = arm_drop(load(kept["Idle"]))
        result.update(arms_down=mine, manny_arms_down=manny_pose)
        result["retarget"] = "ok" if mine > 0.5 * manny_pose else (
            f"arms not lowered ({mine:.2f} of height, Manny {manny_pose:.2f})")
        if kept.get("Walk"):
            # A walk swings the feet apart; legs that do not follow barely step.
            manny_step, _ = widest_step(load(MANNY_WALK))
            step, _ = widest_step(load(kept["Walk"]))
            result.update(walk_stride=step, manny_walk_stride=manny_step)
            result["walk"] = "ok" if step > 0.5 * manny_step else (
                f"legs barely move ({step:.2f} of height, Manny {manny_step:.2f})")
        return result

    def length_of(animation):
        try:
            return float(animation.get_play_length())
        except AttributeError:
            return float(unreal.AnimationLibrary.get_sequence_length(animation))

    def widest_step(animation, samples=24):
        """(widest fore-aft spread of the feet over the cycle, as a share of the head's
        height; the time it happens). Characters face +Y in their own space, like Manny."""
        length = length_of(animation)
        spreads = []
        for i in range(samples):
            time = length * i / samples
            pose = unreal.AnimPoseExtensions.get_anim_pose_at_time(animation, time, unreal.AnimPoseEvaluationOptions())
            left = unreal.AnimPoseExtensions.get_bone_pose(pose, "foot_l", unreal.AnimPoseSpaces.WORLD).translation
            right = unreal.AnimPoseExtensions.get_bone_pose(pose, "foot_r", unreal.AnimPoseSpaces.WORLD).translation
            spreads.append((abs(left.y - right.y) / max(bone_z(pose, "head"), 1e-3), time))
        return max(spreads) if spreads else (0.0, 0.0)

    def bone_z(pose, bone):
        return unreal.AnimPoseExtensions.get_bone_pose(pose, bone, unreal.AnimPoseSpaces.WORLD).translation.z

    def arm_drop(animation, time=0.5):
        """How far the hands hang below the shoulders, as a share of the head's height."""
        pose = unreal.AnimPoseExtensions.get_anim_pose_at_time(animation, time, unreal.AnimPoseEvaluationOptions())
        shoulders = (bone_z(pose, "upperarm_l") + bone_z(pose, "upperarm_r")) / 2
        hands = (bone_z(pose, "hand_l") + bone_z(pose, "hand_r")) / 2
        return (shoulders - hands) / max(bone_z(pose, "head"), 1e-3)

    def facing(skeleton):
        """The way the toes point in the reference pose: both feet to their balls, averaged,
        so the natural outward turn of each foot cancels out."""
        pose = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
        x = y = 0.0
        for side in ("l", "r"):
            foot = unreal.AnimPoseExtensions.get_bone_pose(pose, f"foot_{side}", unreal.AnimPoseSpaces.WORLD).translation
            ball = unreal.AnimPoseExtensions.get_bone_pose(pose, f"ball_{side}", unreal.AnimPoseSpaces.WORLD).translation
            x, y = x + ball.x - foot.x, y + ball.y - foot.y
        length = max((x * x + y * y) ** 0.5, 1e-6)
        return [x / length, y / length, 0.0]

    def write(data):
        measured_path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")

    if stage == "import":
        # A commandlet starts before the asset registry has seen the project's content:
        # without this an earlier check's assets are invisible, cannot be replaced, and
        # creating them again fails.
        unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous([job["content_root"]], True)
        data = {"engine": unreal.SystemLibrary.get_engine_version(), "assets": {}}
        for export in job["exports"]:
            try:
                import_one(export)
                data["assets"][export["sidecar"]["mesh"]] = measure(export)
            except Exception:
                data["assets"][export["sidecar"]["mesh"]] = {"error": traceback.format_exc()[-1500:]}
            write(data)
        try:
            if job["manny"]:
                data["walk_time"] = widest_step(load(MANNY_WALK))[1]
            # Where the picture level's sun shines, so the camera can stand on its side.
            unreal.EditorLoadingAndSavingUtils.load_map(TEMPLATE_LEVEL)
            actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
            sun = next(a for a in actors if isinstance(a, unreal.DirectionalLight)).get_actor_forward_vector()
            data["sun"] = [sun.x, sun.y, sun.z]
        except Exception:
            data["stage_error"] = traceback.format_exc()[-800:]
        write(data)
        return

    # stage == "capture": a lit level, everything standing in a row, the idle playing.
    # The editor world never advances an animation, so the picture is taken in a
    # Simulate session, where the level runs as it would in the game.
    data = json.loads(measured_path.read_text(encoding="utf-8"))
    state = {"frame": 0}
    leader_tag, follow_tag = "AF_BODY:", "AF_FOLLOWS:"

    def always_loaded(actor):
        # The template level uses World Partition: without this a Simulate session only
        # streams in the cells near a player, and there is none.
        try:
            actor.set_editor_property("is_spatially_loaded", False)
        except Exception:  # some template actors (the partition's own) have no such switch
            pass
        return actor

    walk_tag, close_tag = "AF_WALK:", "AF_CAMERA:"

    def spawn(mesh, where, yaw, animation=None, tags=()):
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        location, rotation = unreal.Vector(where[0], where[1], 0), unreal.Rotator(pitch=0, yaw=yaw, roll=0)
        if isinstance(mesh, unreal.StaticMesh):
            actor = always_loaded(actors.spawn_actor_from_class(unreal.StaticMeshActor, location, rotation))
            actor.static_mesh_component.set_static_mesh(mesh)
            return actor
        actor = always_loaded(actors.spawn_actor_from_class(unreal.SkeletalMeshActor, location, rotation))
        actor.tags = list(tags)
        component = actor.skeletal_mesh_component
        component.set_skeletal_mesh_asset(mesh)
        if animation is not None:
            # Half a second in and nearly still, so the picture shows a pose, not a blur of
            # one. A play rate of 0 would never evaluate the pose at all.
            component.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
            component.set_editor_property("animation_data", unreal.SingleAnimationPlayData(
                anim_to_play=animation, saved_looping=True, saved_playing=True, saved_position=0.5,
                saved_play_rate=SLOW))
        return actor

    def setup():
        # The engine's default template brings sky, sun and floor.
        levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        levels.new_level_from_template(f"{job['content_root']}/L_Check", TEMPLATE_LEVEL)
        stage = job["stage"]
        where = dict(zip(stage["order"], stage["positions"]))
        if job["manny"]:
            spawn(load(MANNY_MESH), where["Manny"], stage["actor_yaw"], load(MANNY_IDLE), tags=[walk_tag + MANNY_WALK])
        for export in job["exports"]:
            side = export["sidecar"]
            mesh, got = mesh_of(export), data["assets"].get(side["mesh"], {})
            if mesh is None:
                continue
            if side["kind"] == "equipment":
                # Worn, not shown apart: Unreal's Leader Pose makes it follow the body.
                body = side["skeleton_from"]
                if body in where:
                    spawn(mesh, where[body], stage["actor_yaw"], tags=[follow_tag + body])
                continue
            tags = [leader_tag + side["mesh"]] + ([walk_tag + got["walk_asset"]] if got.get("walk_asset") else [])
            animation = load(got["retarget_asset"]) if got.get("retarget_asset") else None
            spawn(mesh, where[side["mesh"]], stage["actor_yaw"], animation, tags=tags)
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for actor in actors.get_all_level_actors():
            always_loaded(actor)  # the template's floor and sky too
        camera = always_loaded(actors.spawn_actor_from_class(
            unreal.CameraActor, unreal.Vector(*stage["camera"]), unreal.Rotator(pitch=0, yaw=stage["camera_yaw"], roll=0)))
        camera.tags = [close_tag]
        for view in [{"name": None, "fov": stage["fov"]}] + stage.get("closeups", []):
            lens_actor = camera
            if view["name"] is not None:
                lens_actor = always_loaded(actors.spawn_actor_from_class(
                    unreal.CameraActor, unreal.Vector(*view["camera"]),
                    unreal.Rotator(pitch=0, yaw=stage["camera_yaw"], roll=0)))
                lens_actor.tags = [close_tag + view["name"]]
            lens = lens_actor.get_editor_property("camera_component")
            lens.set_editor_property("field_of_view", view["fov"])
            lens.set_editor_property("aspect_ratio", CAPTURE_SIZE[0] / CAPTURE_SIZE[1])
        levels.editor_play_simulate()

    def game():
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()

    def skeletal_actors():
        return unreal.GameplayStatics.get_all_actors_of_class(game(), unreal.SkeletalMeshActor)

    def wear():
        """Equipment follows its body: Leader Pose, set once the session is running."""
        bodies, followers = {}, []
        for actor in skeletal_actors():
            for tag in (str(t) for t in actor.tags):
                if tag.startswith(leader_tag):
                    bodies[tag[len(leader_tag):]] = actor.skeletal_mesh_component
                elif tag.startswith(follow_tag):
                    followers.append((tag[len(follow_tag):], actor.skeletal_mesh_component))
        for body, component in followers:
            if body in bodies:
                component.set_leader_pose_component(bodies[body])

    def walk():
        """Everyone who has a walk takes the step Manny's is widest at.

        Switching animation inside a running Simulate session leaves the reference pose,
        so the session is ended, the walk set on the level's own actors (the way the idle
        was), and a second session started.
        """
        moment = data.get("walk_time", 0.5)
        switched = data.setdefault("walk_switched", [])
        for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
            if not isinstance(actor, unreal.SkeletalMeshActor):
                continue
            for tag in (str(t) for t in actor.tags):
                if tag.startswith(walk_tag):
                    switched.append(tag[len(walk_tag):])
                    actor.skeletal_mesh_component.set_editor_property("animation_data", unreal.SingleAnimationPlayData(
                        anim_to_play=load(tag[len(walk_tag):]), saved_looping=True, saved_playing=True,
                        saved_position=moment, saved_play_rate=SLOW))

    def look():
        # A scene capture in a Simulate session renders once and starts from a dark
        # exposure; the level viewport keeps rendering and adapts to the daylight.
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_set_viewport_realtime(True)
        stage = job["stage"]
        unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
            unreal.Vector(*stage["camera"]), unreal.Rotator(pitch=0, yaw=stage["camera_yaw"], roll=0))

    def shoot(picture, name=None):
        wanted = close_tag + (name or "")
        cameras = [c for c in unreal.GameplayStatics.get_all_actors_of_class(game(), unreal.CameraActor)
                   if wanted in [str(t) for t in c.tags]]
        unreal.AutomationLibrary.take_high_res_screenshot(
            CAPTURE_SIZE[0], CAPTURE_SIZE[1], str(picture), camera=cameras[0] if cameras else None)

    steps = {20: wear, 25: look, 200: lambda: shoot(job["picture"])}
    at = 230
    for close in job["stage"].get("closeups", []):
        steps[at] = (lambda n=close["name"]: shoot(closeup_path(job["picture"], n), n))
        at += 30
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    steps.update({at: levels.editor_request_end_play, at + 20: walk, at + 30: levels.editor_play_simulate,
                  at + 50: wear, at + 60: look, at + 230: lambda: shoot(job["picture_walk"]),
                  at + 260: lambda: write(data), at + 270: levels.editor_request_end_play})
    last = at + 290

    def tick(_delta):
        frame = state["frame"]
        state["frame"] = frame + 1
        try:
            if frame == 5:
                setup()
            elif frame in steps:
                steps[frame]()
            elif frame == last:
                raise StopIteration
        except BaseException as exc:
            if not isinstance(exc, StopIteration):
                data["capture_error"] = traceback.format_exc()[-1500:]
                write(data)
            unreal.unregister_slate_post_tick_callback(state["handle"])
            unreal.SystemLibrary.quit_editor()

    state["handle"] = unreal.register_slate_post_tick_callback(tick)


if len(sys.argv) >= 4 and sys.argv[1] == "--inside":  # Unreal's `py file.py args`
    _inside(sys.argv[2], sys.argv[3])
elif __name__ == "__main__":
    raise SystemExit(main())
