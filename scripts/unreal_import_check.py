"""Import exported assets into Unreal with Interchange and check them, with no editor window.

    python scripts/unreal_import_check.py output/unreal/SM_Barrel output/unreal/SK_Warrior ...

Each folder is one `scripts/export_unreal.py` result (a GLB and its `.unreal.json`). With
no `--project`, a blank project is made in `output/unreal-check/` (git-ignored), with the
engine's own Mannequin template content copied in so humanoids can be checked against
Manny. Unreal runs twice:

1. `UnrealEditor-Cmd` (no rendering): imports every GLB the way the sidecar says (an
   equipment piece onto its body's Skeleton, skeletal LODs reduced by Unreal), measures
   size, base, LODs, collision, names, ORM colour space, and for a humanoid asks the IK
   Rig to recognise the skeleton and retargets Manny's `MM_Idle` onto it.
2. `UnrealEditor` off screen: stands everything in a lit level, plays the retargeted idle
   next to Manny, measures where the hands and feet really are, and saves a picture.

Writes `unreal_check.json` and `unreal_check.png` into every folder; exits non-zero when
any check fails. The same file is the script Unreal runs (`--inside`).
"""

from __future__ import annotations

import argparse
import json
import os
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
CONTENT_ROOT = "/Game/AF"
PROJECT_PLUGINS = ("PythonScriptPlugin", "EditorScriptingUtilities", "Interchange", "InterchangeEditor", "IKRig")
CAPTURE_SIZE = (1280, 720)


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
        [str(editor), str(project), f"-ExecCmds=py {script} --inside capture {job.as_posix()}", "-RenderOffscreen",
         *common],
    ]


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


def write_results(exports: list[dict], measured: dict, picture: Path | None) -> bool:
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
        shot = folder / "unreal_check.png"
        shot.unlink(missing_ok=True)  # never leave an older run's picture looking current
        if picture and picture.is_file():
            shutil.copyfile(picture, shot)
        elif measured.get("capture_error"):
            results.append({"check": "picture", "ok": False, "detail": measured["capture_error"].splitlines()[-1]})
            ok = False
            all_ok = False
        (folder / "unreal_check.json").write_text(json.dumps({
            "ok": ok, "engine": measured.get("engine"), "mesh": mesh, "results": results, "measured": got,
            "picture": "unreal_check.png" if picture and picture.is_file() else None,
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
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = work / f"job-{stamp}.json"
    measured_path, picture = work / f"measured-{stamp}.json", work / f"capture-{stamp}.png"
    job.write_text(json.dumps({"exports": exports, "measured": str(measured_path), "picture": str(picture),
                               "content_root": CONTENT_ROOT, "manny": need_manny}, indent=1), encoding="utf-8")
    for command in editor_commands(editor, project, job):
        print("unreal:", " ".join(command[:3]), "...", flush=True)
        log = work / f"unreal-{stamp}-{'import' if 'Cmd' in Path(command[0]).name else 'capture'}.log"
        with log.open("wb") as handle:
            run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if not measured_path.is_file():
        raise SystemExit(f"Unreal wrote no measurements; see the logs in {work}")
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    return write_results(exports, measured, picture)


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
        name_animations(export)
        unreal.EditorAssetLibrary.save_directory(folder)

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
        for name, skeletal in ((f"IK_{side['mesh']}_Manny", manny), (f"IK_{side['mesh']}", mesh)):
            rig = tools.create_asset(name, folder, unreal.IKRigDefinition, unreal.IKRigDefinitionFactory())
            controller = unreal.IKRigController.get_controller(rig)
            controller.set_skeletal_mesh(skeletal)
            recognised = bool(controller.apply_auto_generated_retarget_definition())
            rigs.append(rig)
        retargeter = tools.create_asset(f"RTG_Manny_{side['mesh']}", folder, unreal.IKRetargeter, unreal.IKRetargetFactory())
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
        made = unreal.IKRetargetBatchOperation.duplicate_and_retarget(
            [unreal.EditorAssetLibrary.find_asset_data(MANNY_IDLE)], manny, mesh, retargeter,
            "MM_", f"AS_{side['mesh']}_", "", "")
        result = {"ik_recognised": recognised, "retarget_asset": None}
        if not made:
            return dict(result, retarget="no animation was made")
        # The batch retarget writes beside nothing in particular (the content root) and
        # does not save: keep it with the character, on disk, for the picture run.
        animation_path = f"{folder}/AS_{side['mesh']}_Idle"
        if unreal.EditorAssetLibrary.does_asset_exist(animation_path):
            unreal.EditorAssetLibrary.delete_asset(animation_path)
        unreal.EditorAssetLibrary.rename_asset(str(made[0].package_name), animation_path)
        unreal.EditorAssetLibrary.save_directory(folder)
        result["retarget_asset"] = animation_path
        # Read the pose straight from the animations, half a second in: Manny's idle hangs
        # his arms by his sides, and a T-pose here means the retarget moved nothing.
        manny_pose = arm_drop(load(MANNY_IDLE))
        mine = arm_drop(load(animation_path))
        result.update(arms_down=mine, manny_arms_down=manny_pose)
        result["retarget"] = "ok" if mine > 0.5 * manny_pose else (
            f"arms not lowered ({mine:.2f} of height, Manny {manny_pose:.2f})")
        return result

    def bone_z(pose, bone):
        return unreal.AnimPoseExtensions.get_bone_pose(pose, bone, unreal.AnimPoseSpaces.WORLD).translation.z

    def arm_drop(animation, time=0.5):
        """How far the hands hang below the shoulders, as a share of the head's height."""
        pose = unreal.AnimPoseExtensions.get_anim_pose_at_time(animation, time, unreal.AnimPoseEvaluationOptions())
        shoulders = (bone_z(pose, "upperarm_l") + bone_z(pose, "upperarm_r")) / 2
        hands = (bone_z(pose, "hand_l") + bone_z(pose, "hand_r")) / 2
        return (shoulders - hands) / max(bone_z(pose, "head"), 1e-3)

    def facing(skeleton):
        """The way the toes point in the reference pose, from the left foot to its ball."""
        pose = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
        foot = unreal.AnimPoseExtensions.get_bone_pose(pose, "foot_l", unreal.AnimPoseSpaces.WORLD).translation
        ball = unreal.AnimPoseExtensions.get_bone_pose(pose, "ball_l", unreal.AnimPoseSpaces.WORLD).translation
        x, y = ball.x - foot.x, ball.y - foot.y
        length = max((x * x + y * y) ** 0.5, 1e-6)
        return [x / length, y / length, 0.0]

    def write(data):
        measured_path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")

    if stage == "import":
        data = {"engine": unreal.SystemLibrary.get_engine_version(), "assets": {}}
        for export in job["exports"]:
            try:
                import_one(export)
                data["assets"][export["sidecar"]["mesh"]] = measure(export)
            except Exception:
                data["assets"][export["sidecar"]["mesh"]] = {"error": traceback.format_exc()[-1500:]}
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

    def spawn(mesh, x, animation=None, tags=()):
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        if isinstance(mesh, unreal.StaticMesh):
            actor = always_loaded(actors.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(x, 0, 0)))
            actor.static_mesh_component.set_static_mesh(mesh)
            return actor
        actor = always_loaded(actors.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(x, 0, 0)))
        actor.tags = list(tags)
        component = actor.skeletal_mesh_component
        component.set_skeletal_mesh_asset(mesh)
        if animation is not None:
            component.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
            component.set_editor_property("animation_data", unreal.SingleAnimationPlayData(
                anim_to_play=animation, saved_looping=True, saved_playing=True, saved_position=0.5,
                saved_play_rate=1.0))
        return actor

    def setup():
        # The engine's default template brings sky, sun and floor.
        levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        levels.new_level_from_template(f"{job['content_root']}/L_Check", "/Engine/Maps/Templates/Template_Default")
        x, placed = 0.0, {}
        if job["manny"]:
            spawn(load(MANNY_MESH), x, load(MANNY_IDLE))
            x += 120
        for export in job["exports"]:
            side = export["sidecar"]
            mesh, got = mesh_of(export), data["assets"].get(side["mesh"], {})
            if mesh is None:
                continue
            if side["kind"] == "equipment" and side["skeleton_from"] in placed:
                # Worn, not shown apart: Unreal's Leader Pose makes it follow the body.
                spawn(mesh, placed[side["skeleton_from"]], tags=[follow_tag + side["skeleton_from"]])
                continue
            animation = load(got["retarget_asset"]) if got.get("retarget_asset") else None
            width = max((got.get("bounds_cm") or [100, 100, 100])[:2]) + 60
            x += width / 2
            spawn(mesh, x, animation, tags=[leader_tag + side["mesh"]])
            placed[side["mesh"]] = x
            x += width / 2
        height = max([180.0] + [max((g.get("bounds_cm") or [0, 0, 0])[2], 0) for g in data["assets"].values()])
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for actor in actors.get_all_level_actors():
            always_loaded(actor)  # the template's floor and sky too
        # The template's sun lights the row from behind; a second light from the camera's
        # side shows the faces.
        key = always_loaded(actors.spawn_actor_from_class(
            unreal.DirectionalLight, unreal.Vector(0, 0, 400), unreal.Rotator(pitch=-35, yaw=-100, roll=0)))
        key.get_editor_property("directional_light_component").set_mobility(unreal.ComponentMobility.MOVABLE)
        key.get_editor_property("directional_light_component").set_intensity(4.0)
        distance = max(x * 0.6, height) + 100  # a 90-degree lens: half the row fits at that distance
        camera = always_loaded(actors.spawn_actor_from_class(
            unreal.SceneCapture2D, unreal.Vector(x / 2, distance, height * 0.55), unreal.Rotator(pitch=-5, yaw=-90, roll=0)))
        target_path = f"{job['content_root']}/RT_Check"
        if unreal.EditorAssetLibrary.does_asset_exist(target_path):
            unreal.EditorAssetLibrary.delete_asset(target_path)
        target = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "RT_Check", job["content_root"], unreal.TextureRenderTarget2D, unreal.TextureRenderTargetFactoryNew())
        target.set_editor_property("render_target_format", unreal.TextureRenderTargetFormat.RTF_RGBA8)
        target.set_editor_property("size_x", CAPTURE_SIZE[0])
        target.set_editor_property("size_y", CAPTURE_SIZE[1])
        capture = camera.get_editor_property("capture_component2d")
        capture.set_editor_property("texture_target", target)
        capture.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        capture.set_editor_property("capture_every_frame", False)
        state["target"] = target
        levels.editor_play_simulate()

    def game():
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()

    def wear():
        """Equipment follows its body: Leader Pose, set once the session is running."""
        world = game()
        bodies, followers = {}, []
        for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.SkeletalMeshActor):
            for tag in (str(t) for t in actor.tags):
                if tag.startswith(leader_tag):
                    bodies[tag[len(leader_tag):]] = actor.skeletal_mesh_component
                elif tag.startswith(follow_tag):
                    followers.append((tag[len(follow_tag):], actor.skeletal_mesh_component))
        for body, component in followers:
            if body in bodies:
                component.set_leader_pose_component(bodies[body])

    def shoot():
        world = game()
        camera = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.SceneCapture2D)[0]
        camera.get_editor_property("capture_component2d").capture_scene()

    def finish():
        directory, name = os.path.split(job["picture"])
        unreal.RenderingLibrary.export_render_target(game(), state["target"], directory, name)
        write(data)

    def tick(_delta):
        frame = state["frame"]
        state["frame"] = frame + 1
        try:
            if frame == 5:
                setup()
            elif frame == 20:
                wear()
            elif frame == 120:
                shoot()
            elif frame == 130:
                finish()
            elif frame == 135:
                unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()
            elif frame == 150:
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
