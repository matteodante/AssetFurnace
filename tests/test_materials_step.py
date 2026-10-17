"""The materials step on every route, as the user closed function 8: one material per asset or
piece, proposed by a rule and chosen by a person; no automatic zones and no zone masks
(section 9 of docs/plans/2026-10-17-step-parity-design.md). It can be left out, chosen,
and run again on its own."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from batch_fakes import fake_step, install_fake_generator, tiny_glb

from image_to_3dlab import batch, mcp_server, step_catalog
from image_to_3dlab.batch_steps import StepResult

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
sys.path.insert(0, str(REPO / "scripts"))
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _spec(tmp_path, steps, row=None):
    asset = tiny_glb(tmp_path / "chest.glb", "a chest")
    # the words it was made from, in its provenance, as a generated model keeps them
    asset.with_suffix(".provenance.json").write_text(json.dumps({
        "license": {"name": "MIT"}, "classification": "commercial-clear",
        "prompt": "an old oak chest with iron bands"}))
    spec = {"schema_version": 1, "name": "mats", "output": {"directory": "out"},
            "rows": [{"id": "SM_Chest", "asset": str(asset), "kind": "prop", "size_cm": 80,
                      "steps": list(steps), **(row or {})}]}
    path = tmp_path / "mats.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.fixture
def exported(monkeypatch):
    seen = []

    def export(ctx):
        seen.append(ctx.options.get("material") or ctx.material)
        return StepResult()

    monkeypatch.setitem(batch.STEPS, "export_unreal", fake_step(export))
    return seen


# ---------------------------------------------------------------------- batches

def test_the_rule_proposes_the_material_and_the_export_takes_it(tmp_path, monkeypatch, exported):
    install_fake_generator(monkeypatch)
    spec = _spec(tmp_path, ["materials", "export_unreal"])
    assert batch.run_batch(batch.load_batch(spec)).exit_code == 0
    assert exported == ["wood"]   # "oak" in the prompt
    record = json.loads((tmp_path / "out" / "mats" / "rows" / "SM_Chest" / "materials" / "material.json").read_text())
    assert record["material"] == "wood" and record["by"] == "rule" and "prompt" in record["reason"]


def test_a_person_chooses_it_and_a_change_reruns_only_from_there(tmp_path, monkeypatch, exported):
    install_fake_generator(monkeypatch)
    spec = _spec(tmp_path, ["materials", "export_unreal"], {"materials": {"material": "metal"}})
    batch.run_batch(batch.load_batch(spec))
    assert exported == ["metal"]
    data = json.loads(spec.read_text())
    data["rows"][0]["materials"] = {"material": "leather"}
    spec.write_text(json.dumps(data))
    batch.run_batch(batch.load_batch(spec))
    assert exported == ["metal", "leather"]


def test_left_out_the_export_keeps_the_models_own_maps(tmp_path, monkeypatch, exported):
    install_fake_generator(monkeypatch)
    batch.run_batch(batch.load_batch(_spec(tmp_path, ["export_unreal"])))
    assert exported == [None]


def test_the_exports_own_material_wins(tmp_path, monkeypatch, exported):
    install_fake_generator(monkeypatch)
    spec = _spec(tmp_path, ["materials", "export_unreal"], {"export_unreal": {"material": "stone"}})
    batch.run_batch(batch.load_batch(spec))
    assert exported == ["stone"]


def test_the_material_survives_a_resume(tmp_path, monkeypatch, exported):
    install_fake_generator(monkeypatch)
    spec = _spec(tmp_path, ["materials", "export_unreal"], {"materials": {"material": "metal"}})
    batch.run_batch(batch.load_batch(spec))
    state = tmp_path / "out" / "mats" / batch.STATE_FILE
    data = json.loads(state.read_text())
    data["rows"]["SM_Chest"]["steps"]["export_unreal"]["status"] = "failed"   # only the export runs again
    data["rows"]["SM_Chest"]["status"] = "failed"
    state.write_text(json.dumps(data))
    batch.run_batch(batch.load_batch(spec), retry_failed=True)
    assert exported == ["metal", "metal"]   # the export again, with the material the done step chose


def test_a_material_not_in_the_list_stops_the_batch_first(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    with pytest.raises(batch.BatchError, match="material"):
        batch.run_batch(batch.load_batch(_spec(tmp_path, ["materials"], {"materials": {"material": "gold"}})))


def test_the_step_is_described_with_its_command_line():
    spec = step_catalog.get("materials")
    assert spec.feature == 8 and spec.skip and spec.replace == ()
    assert [p.name for p in spec.params] == ["material"]
    assert "leather" in step_catalog.choices(spec.param("material"))
    import export_unreal

    flags = {flag for action in export_unreal.build_parser()._actions for flag in action.option_strings}
    assert spec.param("material").cli_flag in flags


def test_an_agent_runs_and_chooses_it(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(mcp_server, "submit_batch", lambda spec, **kw: seen.update(spec=json.loads(Path(spec).read_text()))
                        or {"started": True})
    monkeypatch.setattr(mcp_server, "AGENT_SPECS", tmp_path / "agent")
    assert {s["name"]: s for s in mcp_server.describe_steps()["steps"]}["materials"]["available"]
    result = mcp_server.run_steps("SM_Chest", ["materials", "export_unreal"], asset=str(tiny_glb(tmp_path / "c.glb")),
                                  size_cm=80, options={"materials": {"material": "leather"}})
    assert result["started"] and seen["spec"]["rows"][0]["materials"] == {"material": "leather"}


# ---------------------------------------------------------------------- pieces, for agents

def test_an_agent_corrects_the_material_of_a_piece(tmp_path, monkeypatch):
    from tests.test_viewer_slot_parts import _body_json

    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)
    folder = output / "sets" / "red" / "stick"
    folder.mkdir(parents=True)
    (folder / "red.parts.json").write_text(json.dumps({
        "set": "red", "slots": {"helmet": {}, "boots": {"prompt": "leather boots"}},
        "pieces": {"helmet": {}, "boots": {}}, "concept": {"prompt": "Add red plate armour to the character in image 1"}}))
    monkeypatch.setattr(mcp_server, "OUTPUT", output)
    result = mcp_server.pieces_materials("stick", "red", {"helmet": "leather"})
    assert result["materials"]["helmet"] == {"material": "leather", "by": "person", "reason": "corrected by a person"}
    assert result["materials"]["boots"]["material"] == "leather" and result["materials"]["boots"]["by"] == "rule"
    back = mcp_server.pieces_materials("stick", "red", {"helmet": None})   # back to the rule's
    assert back["materials"]["helmet"]["by"] == "rule"
    bad = mcp_server.pieces_materials("stick", "red", {"cape": "cloth"})
    assert bad["saved"] is False and "cape" in bad["problems"][0]


# ---------------------------------------------------------------------- studio

def test_the_studio_export_can_leave_the_material_out(tmp_path):
    import unreal_export_api as api

    output = tmp_path / "output"
    tiny_glb(output / "props" / "p" / "chest.glb")
    request = api.parse(output, {"file": "props/p/chest.glb", "kind": "prop", "size_cm": 80, "material": "none"})
    assert request["material"] is None
    chosen = api.parse(output, {"file": "props/p/chest.glb", "kind": "prop", "size_cm": 80, "material": "metal"})
    assert chosen["material"] == "metal"


@needs_node
def test_the_studio_material_menu_offers_none():
    url = (REPO / "viewer" / "studio" / "unreal-export.js").as_uri()
    program = (f"import * as m from {json.dumps(url)};\n"
               "const body = (material) => m.requestBody({ source: { file: 'a.glb', moves: [] }, kind: 'prop', size: 80,"
               " measure: 'longest', material });\n"
               "console.log(JSON.stringify({ none: body('none').material, auto: body('').material ?? null,"
               " options: m.MATERIAL_CHOICES }));")
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True,
                            text=True, encoding="utf-8")
    out = json.loads(result.stdout)
    assert out["none"] == "none" and out["auto"] is None
    assert ["none", "Keep its own maps"] in out["options"]
