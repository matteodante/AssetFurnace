"""Weapons and shields are game assets of their own (the user's decision, 17/10/2026): made beside
a set, never fitted to the body nor tied to it in Unreal, but made well. A canonical picture,
scored by the picture judge; the grip found on it and brought onto the 3D; an Unreal static
mesh with its pivot on the grip, a `grip` socket, its real size. sd-cli, Pixal3D, Blender,
Unreal and the judge are stand-ins (one test runs the real Blender)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab import unreal_profile as up
from image_to_3dlab.slot_parts import candidates, grip, pieces
from image_to_3dlab.slot_parts.camera import Camera
from tests.test_pieces_picks import _with_picture
from tests.test_slot_parts_run import _fakes

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import export_unreal  # noqa: E402
import slot_parts as cli  # noqa: E402

from tests.batch_fakes import tiny_glb  # noqa: E402

CAM = Camera(fov=0.8, dist=3.0, size=64)


def _sword(path: Path) -> Path:
    blade = trimesh.creation.box(extents=(0.12, 0.9, 0.02))
    blade.apply_translation((0, 0.55, 0))
    handle = trimesh.creation.box(extents=(0.04, 0.3, 0.04))
    handle.apply_translation((0, -0.15, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.util.concatenate([blade, handle]).export(path)
    return path


def _record(folder: Path, slot="weapon", words="a longsword") -> dict:
    picture = folder / "pieces" / slot / "s1.png"
    picture.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (217, 217, 217)).save(picture)
    model = _sword(pieces.finished_path(folder / "pieces", slot))
    handle = CAM.project(np.array([[0, -0.3, 0], [0, 0.0, 0]])) / 64 * 1000
    text = f'[{{"bbox_2d": [{handle[:, 0].min() - 300:.0f}, {handle[:, 1].min():.0f}, ' \
           f'{handle[:, 0].max() + 300:.0f}, {handle[:, 1].max():.0f}]}}]'
    record = grip.find(model, picture, CAM, slot=slot, words=words, ask_box=lambda p, q: text)
    grip.write(record, grip.path_for(picture))
    return record


@pytest.fixture
def judged(monkeypatch):
    """The grip and the scores as the judge would give them, without the judge."""
    found = []

    def find_grip(slot, words, picture, model):
        found.append(slot)
        return {"slot": slot, "class": grip.weapon_class(slot, words), "size_cm": 100.0, "point": [0, 0, 0],
                "axis": [0, 1, 0], "by": "judge", "picture": str(picture), "model": str(model)}

    monkeypatch.setattr(cli, "find_grip", find_grip)
    monkeypatch.setattr(cli, "score_held", lambda slot, words, pictures: {seed: 0.1 * seed for seed in pictures})
    return found


# ------------------------------------------------------------------- the set

def test_a_weapon_is_made_beside_the_set_not_fitted_to_the_body(tmp_path, judged):
    body_json, source = _with_picture(tmp_path)
    calls, jobs = [], []
    edit, generate, run = _fakes(source, calls)

    def fitter(job):
        jobs.append(job)
        return {"pieces": {p["slot"]: {} for p in job["pieces"]}, "covered_share": 0.6, "files": ["set.glb"]}

    path = cli.run_set(body_json, "red", "torso", concept=source, weapon="a longsword", root=tmp_path / "sets",
                       run=run, edit=edit, generate=generate, fitter=fitter, checker=lambda *a: {})
    parts = json.loads(path.read_text())
    assert [p["slot"] for p in jobs[0]["pieces"]] == ["torso"]          # the weapon is not fitted
    weapon = parts["pieces"]["weapon"]
    assert weapon["standalone"] is True and Path(weapon["grip"]).name == "grip.json"
    assert json.loads(Path(weapon["grip"]).read_text())["by"] == "judge" and judged == ["weapon"]


def test_a_set_of_weapons_only_needs_no_fit(tmp_path, judged):
    body_json, source = _with_picture(tmp_path)
    calls = []
    edit, generate, run = _fakes(source, calls)

    def fitter(job):
        raise AssertionError("nothing to fit")

    path = cli.run_set(body_json, "arms", {}, concept=source, weapon="a battle axe", shield="a kite shield",
                       root=tmp_path / "sets", run=run, edit=edit, generate=generate, fitter=fitter,
                       checker=lambda *a: {})
    parts = json.loads(path.read_text())
    assert parts["fit"] is None and set(parts["pieces"]) == {"weapon", "shield"}


def test_weapon_pictures_are_scored_and_the_best_is_offered_first(tmp_path, judged):
    body_json, source = _with_picture(tmp_path)
    edit, generate, run = _fakes(source, [])
    cli.run_set(body_json, "arms", {}, concept=source, weapon="a longsword", seeds=(1, 2, 3), stop_at_pictures=True,
                root=tmp_path / "sets", run=run, edit=edit, generate=generate)
    entry = candidates.load(tmp_path / "sets" / "arms" / "stick" / "pieces")["slots"]["weapon"]
    assert entry["auto"] == 3 and entry["candidates"]["3"]["score"] == pytest.approx(0.3)


def test_the_picture_asks_for_the_canonical_pose():
    words = pieces.held_prompt("weapon", "a longsword")
    for part in ("upright", "grip at the bottom", "point or head at the top", "plain"):
        assert part in words, part


# ----------------------------------------------------------------- Unreal

def test_unreal_exports_each_weapon_as_its_own_asset_with_its_grip(tmp_path):
    folder = tmp_path / "sets" / "arms" / "stick"
    record = _record(folder)
    from tests.test_slot_parts_cli import _body

    body_json, *_ = _body(tmp_path)
    (folder / "arms.parts.json").write_text(json.dumps({"set": "arms", "fit": None, "pieces": {
        "weapon": {"standalone": True, "glb": record["model"], "grip": str(grip.path_for(Path(record["picture"])))}}}))
    calls = []

    def exporter(source, **kwargs):
        calls.append((Path(source), kwargs))
        return {"glb": str(source), "problems": []}

    path = cli.unreal_set(folder / "fit", body_json, tmp_path / "unreal", exporter=exporter)
    (source, kwargs), = calls   # no fit: the weapon alone, no body
    assert source == Path(record["model"]) and kwargs["kind"] == "prop" and "body" not in kwargs
    assert kwargs["grip"]["point"] == record["point"] and kwargs["size_cm"] == record["size_cm"]
    assert kwargs["measure"] == "longest" and kwargs["collision"] == "convex"
    assert kwargs["name"] == "ArmsWeapon"   # named after its set, not the body: an asset of its own
    summary = json.loads(path.read_text())
    assert summary["standalone"]["weapon"]["socket"] == "grip" and any("grip" in line for line in summary["how"])


def test_the_export_puts_the_pivot_on_the_grip(tmp_path):
    weapon = tiny_glb(tmp_path / "weapon.glb")
    record = {"point": [0, -0.15, 0], "axis": [0, 1, 0], "class": "sword", "size_cm": 100}
    p = export_unreal.plan(weapon, kind="prop", size_cm=100, measure="longest", grip=record, out_dir=tmp_path / "o")
    assert p["job"]["grip"] == {"point": [0, -0.15, 0], "axis": [0, 1, 0]}
    with pytest.raises(ValueError, match="grip"):
        export_unreal.plan(weapon, kind="creature", size_cm=100, grip=record, out_dir=tmp_path / "o2")
    with pytest.raises(ValueError, match="grip"):
        export_unreal.plan(weapon, kind="prop", size_cm=100, grip={"point": None}, out_dir=tmp_path / "o3")
    side = up.sidecar(kind="prop", names=p["names"], glb="x.glb", size_cm=100, measure="longest", lods=1,
                      collision="convex", humanoid=False, skeleton_from=None, animations=[], tint=None, problems=[],
                      grip=True)
    assert side["grip_socket"] == {"name": "grip"}


def _expected():
    return {"kind": "prop", "problems": [], "orm": False, "skinned": False, "collision": None, "mesh": "SM_W",
            "lods": 1, "grip_socket": {"name": "grip"}, "expect": {"size_cm": 100, "measure": "longest"}}


def test_unreal_checks_the_grip_socket_and_the_pivot_not_the_floor():
    good = {"bounds_cm": [12, 3, 100], "min_z": -30.0, "box_cm": [[-6, -1.5, -30], [6, 1.5, 70]], "names": ["SM_W"],
            "lods": 1, "grip": {"made": True, "location_cm": [0, 0, 0]}}
    checks = {r["check"]: r["ok"] for r in up.evaluate_unreal(good, _expected())}
    assert "base" not in checks and checks["grip"] is True and checks["size"] is True
    away = {**good, "grip": {"made": True, "location_cm": [0, 0, 40]}}
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(away, _expected())}["grip"] is False
    outside = {**good, "box_cm": [[-6, -1.5, 10], [6, 1.5, 110]]}   # the pivot is not on the weapon
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(outside, _expected())}["grip"] is False
    missing = {**good, "grip": {"made": False, "error": "no socket"}}
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(missing, _expected())}["grip"] is False


@pytest.mark.skipif(__import__("image_to_3dlab.blender", fromlist=["find_blender"]).find_blender() is None,
                    reason="needs Blender")
def test_real_blender_puts_the_grip_at_the_origin_and_the_blade_up(tmp_path):
    source = _sword(tmp_path / "sword.glb")
    record = {"point": [0, -0.15, 0], "axis": [0, 1, 0]}
    result = export_unreal.export_unreal(source, kind="prop", size_cm=120, measure="longest", grip=record,
                                         name="Sword", out_dir=tmp_path / "out", log=lambda m: None)
    lo, hi = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))["blender"]["box_after_m"]
    assert hi[2] - lo[2] == pytest.approx(1.2, rel=1e-3)          # the long side, upright (Blender +Z)
    # the model runs from -0.3 (pommel) to 1.0 (tip), the grip at -0.15: scaled to 1.2 m, the grip at 0
    assert lo[2] == pytest.approx(-0.15 * 1.2 / 1.3, abs=0.005) and abs(lo[0] + hi[0]) < 0.01
    from image_to_3dlab.glb_turn import parse_glb

    document, _ = parse_glb(Path(result["glb"]).read_bytes())
    assert "SOCKET_grip" in [n.get("name") for n in document["nodes"]]


# ------------------------------------------------------------ corrections

def test_a_person_moves_the_grip_from_the_command_line(tmp_path, capsys):
    folder = tmp_path / "sets" / "arms" / "stick"
    _record(folder)
    assert cli.main(["grip", "--set-dir", str(folder), "--slot", "weapon", "--at", "0.5,0.3", "--class", "axe",
                     "--size-cm", "85"]) == 0
    saved = grip.load(folder / "pieces" / "weapon" / "grip.json")
    assert saved["by"] == "person" and saved["class"] == "axe" and saved["size_cm"] == 85
    assert cli.main(["grip", "--set-dir", str(folder), "--slot", "weapon", "--class", "banana"]) == 1


def test_agents_and_the_studio_correct_the_grip_with_the_same_code(tmp_path):
    from image_to_3dlab import mcp_server
    from image_to_3dlab.slot_parts import runs
    from tests.test_slot_parts_cli import _body

    body_json, *_ = _body(tmp_path)   # body "stick" under tmp_path/bodies
    folder = tmp_path / "sets" / "arms" / "stick"
    _record(folder)
    saved = runs.save_grip(tmp_path, {"body": "stick", "set": "arms", "slot": "weapon", "class": "mace"})
    assert saved["grip"]["class"] == "mace" and saved["grip"]["by"] == "person"
    with pytest.raises(ValueError, match="slot"):
        runs.save_grip(tmp_path, {"body": "stick", "set": "arms", "slot": "torso", "class": "mace"})
    original = mcp_server.OUTPUT
    try:
        mcp_server.OUTPUT = tmp_path
        answer = mcp_server.pieces_grip("stick", "arms", "weapon", size_cm=70)
    finally:
        mcp_server.OUTPUT = original
    assert answer["saved"] is True and answer["grip"]["size_cm"] == 70


def test_a_set_of_weapons_needs_no_concept_nor_outfit(tmp_path, judged):
    body_json, source = _with_picture(tmp_path)
    edit, generate, run = _fakes(source, [])
    path = cli.run_set(body_json, "arms", {}, weapon="a longbow", root=tmp_path / "sets", run=run, edit=edit,
                       generate=generate, fitter=None, checker=None)
    parts = json.loads(path.read_text())
    assert parts["mode"] == "standalone" and parts["concept"] is None and parts["fit"] is None


# ------------------------------------------------------------------ the studio

def test_the_studio_lists_each_weapons_grip_on_its_picture(tmp_path):
    from image_to_3dlab.slot_parts import runs

    folder = tmp_path / "sets" / "arms" / "stick"
    record = _record(folder)
    (folder / "arms.parts.json").write_text(json.dumps({"set": "arms", "body": {"name": "stick"}, "fit": None,
                                                       "pieces": {"weapon": {"standalone": True, "glb": record["model"],
                                                                             "grip": str(grip.path_for(Path(record["picture"])))}}}))
    listed = runs.list_sets(tmp_path)[0]["grips"]["weapon"]
    assert listed["picture"] == "sets/arms/stick/pieces/weapon/s1.png" and listed["class"] == "sword"
    x0, y0, x1, y1 = record["box"]
    assert listed["at"] == pytest.approx([(x0 + x1) / 2 / 64, (y0 + y1) / 2 / 64], abs=1e-3)
    assert "axe" in listed["classes"] and listed["by"] == "judge"


def test_the_studio_saves_a_moved_grip(tmp_path):
    sys.path.insert(0, str(REPO / "viewer"))
    import slot_parts_api as api

    from tests.test_slot_parts_cli import _body

    _body(tmp_path)
    _record(tmp_path / "sets" / "arms" / "stick")
    saved = api.save_grip(tmp_path, {"set": "arms", "body": "stick", "slot": "weapon", "at": [0.5, 0.4],
                                     "class": "axe"})
    assert saved["grip"]["by"] == "person" and saved["grip"]["class"] == "axe"


def test_the_marker_sits_on_the_grip_and_a_click_moves_it():
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("Node is required to execute browser ES modules")
    url = (REPO / "viewer" / "studio" / "slot-parts.js").as_uri()
    program = f"""import * as m from {json.dumps(url)};
      const grips = {{weapon: {{picture: 'sets/a/b/pieces/weapon/s1.png', at: [0.5, 0.8], class: 'sword', size_cm: 100,
                                by: 'judge', classes: ['sword', 'axe']}}}};
      console.log(JSON.stringify({{
        html: m.gripRows(grips, '/output/'),
        at: m.pointOn({{left: 100, top: 50, width: 200, height: 400}}, 150, 450),
        clamped: m.pointOn({{left: 0, top: 0, width: 100, height: 100}}, 150, -5),
      }}));"""
    out = json.loads(subprocess.run(["node", "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True).stdout)
    assert 'left:50%' in out["html"].replace(" ", "") and 'top:80%' in out["html"].replace(" ", "")
    assert "/output/sets/a/b/pieces/weapon/s1.png" in out["html"] and 'data-grip-slot="weapon"' in out["html"]
    assert out["at"] == [0.25, 1.0] and out["clamped"] == [1.0, 0.0]


# ------------------------------------------------------- every route of the export

def test_the_command_line_and_batches_take_a_grip_record(tmp_path, monkeypatch):
    from image_to_3dlab.batch_steps import export_unreal as export_step
    from tests.test_batch_steps_export_checks import _context, _row

    weapon = tiny_glb(tmp_path / "weapon.glb")
    record = tmp_path / "grip.json"
    record.write_text(json.dumps({"point": [0, -0.1, 0], "axis": [0, 1, 0], "class": "sword", "size_cm": 90}))
    seen = []

    def fake(source, **kwargs):
        seen.append(kwargs)
        out = Path(kwargs.get("out_dir") or tmp_path)
        glb = tiny_glb(out / "SM_W.glb")
        (out / "SM_W.unreal.json").write_text("{}")
        return {"ok": True, "out_dir": str(out), "glb": str(glb), "sidecar": str(out / "SM_W.unreal.json"),
                "problems": []}

    monkeypatch.setattr(export_unreal, "export_unreal", fake)
    assert export_unreal.main([str(weapon), "--kind", "prop", "--size-cm", "90", "--measure", "longest",
                               "--grip", str(record)]) == 0
    assert seen[-1]["grip"]["point"] == [0, -0.1, 0]
    monkeypatch.setattr(export_step, "export_unreal", fake)
    monkeypatch.setattr(export_step, "find_blender", lambda: Path("blender.exe"))
    ctx = _context(tmp_path, _row(tmp_path), "export_unreal", {"grip": str(record)})
    assert export_step.run(ctx).ok and seen[-1]["grip"]["axis"] == [0, 1, 0]


@pytest.mark.skipif(__import__("image_to_3dlab.blender", fromlist=["find_blender"]).find_blender() is None,
                    reason="needs Blender")
def test_real_blender_hulls_a_model_whose_points_repeat(tmp_path):
    # Pixal3D's meshes repeat points (seams, a blade's two faces meeting): the convex collision
    # broke on the first real sword ("found the same BMVert used multiple times")
    box = trimesh.creation.box(extents=(0.1, 1.0, 0.02))
    twice = trimesh.util.concatenate([box, box.copy(), _sword_mesh()])
    source = tmp_path / "doubled.glb"
    twice.export(source)
    result = export_unreal.export_unreal(source, kind="prop", size_cm=100, measure="longest", collision="convex",
                                         grip={"point": [0, 0, 0], "axis": [0, 1, 0]}, name="Doubled",
                                         out_dir=tmp_path / "out", log=lambda m: None)
    assert result["ok"], result["problems"]


def _sword_mesh():
    blade = trimesh.creation.box(extents=(0.12, 0.9, 0.02))
    blade.apply_translation((0, 0.55, 0))
    return blade


def test_a_set_of_weapons_offers_only_the_export():
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("Node is required to execute browser ES modules")
    url = (REPO / "viewer" / "studio" / "slot-parts.js").as_uri()
    program = f"""import * as m from {json.dumps(url)};
      const bodies = [{{name: 'a'}}, {{name: 'b'}}];
      console.log(JSON.stringify({{arms: m.setActions({{set: 's', body: 'a', mode: 'standalone'}}, bodies),
                                  worn: m.setActions({{set: 's', body: 'a', mode: 'dressed_split'}}, bodies)}}));"""
    out = json.loads(subprocess.run(["node", "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True).stdout)
    assert 'data-set-action="fit"' not in out["arms"] and 'data-set-action="refit"' not in out["arms"]
    assert 'data-set-action="unreal"' in out["arms"] and 'height cm' not in out["arms"]
    assert 'data-set-action="fit"' in out["worn"] and 'data-set-action="refit"' in out["worn"]


def test_a_weapon_without_a_pixal3d_camera_waits_for_a_person(tmp_path, capsys):
    import importlib.util

    # a copy of its own, with the real find_grip (the suite stands it in): it stops before any judge
    spec = importlib.util.spec_from_file_location("slot_parts_fresh", REPO / "scripts" / "slot_parts.py")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    picture = tmp_path / "pieces" / "weapon" / "s1.png"
    picture.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), (217, 217, 217)).save(picture)
    record = fresh.find_grip("weapon", "a longsword", picture, tiny_glb(tmp_path / "w.glb"))
    assert record["by"] == "none" and record["point"] is None and record["class"] == "sword"
    assert "no Pixal3D camera" in capsys.readouterr().out
