"""The materials step on every route (section 9 of docs/plans/2026-10-17-step-parity-design.md):
it can be left out, run on its own, and given a zones file of your own in place of the
segmenter's. A zones file is a JSON {zone: material} with a PNG of zone indices in UV
space, or a run's own `.materials.json` corrected by hand with its `_Tint.png`. No zone
painting here, and no model runs: the zones come from the files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from batch_fakes import install_fake_generator
from PIL import Image
from test_materials import SIZE, banded_albedo, make_glb

from image_to_3dlab import batch, material_zones, step_catalog
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import parse_glb
from image_to_3dlab.materials import surfaces

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import game_materials  # noqa: E402


def _bands(path: Path, values=(1, 2, 3), size: int = SIZE) -> Path:
    """A zone mask in the sphere's UV space: three bands of rows, like its albedo."""
    rows = np.minimum((np.arange(size) / size * len(values)).astype(int), len(values) - 1)
    Image.fromarray(np.array(values, np.uint8)[rows][:, None].repeat(size, axis=1), "L").save(path)
    return path


@pytest.fixture
def asset(tmp_path):
    path = make_glb(tmp_path / "chest.glb", banded_albedo())
    path.with_suffix(".provenance.json").write_text(json.dumps({"license": {"name": "MIT"},
                                                                "classification": "commercial-clear"}))
    return path


@pytest.fixture
def zones_file(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": {"1": "metal", "2": "leather", "3": "cloth"}}))
    return path, _bands(tmp_path / "zones.png")


# ---------------------------------------------------------------------- reading the files

def test_a_zones_file_and_its_mask_are_the_zones(asset, zones_file):
    found, record = material_zones.zones_from_file(asset, *zones_file)
    (zone,) = found.values()
    assert zone.zones() == ["metal", "leather", "cloth"] or sorted(zone.zones()) == ["cloth", "leather", "metal"]
    assert zone.material_at((0.5, 0.1)) == "metal" and zone.material_at((0.5, 0.9)) == "cloth"
    assert record["zones_file"]["name"] == "zones.json" and len(record["zones_mask"]["sha256"]) == 64
    assert record["segmenter"] is None and record["components"] == []   # no model found them


@pytest.mark.parametrize("zones, problem", [
    ({"1": "gold"}, "gold"),              # not one of the materials
    ({"one": "metal"}, "zone"),           # not a zone index
    ({}, "zone"),
])
def test_a_wrong_zones_file_is_refused_with_why(asset, tmp_path, zones, problem):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"zones": zones}))
    with pytest.raises(ValueError, match=problem):
        material_zones.zones_from_file(asset, path, _bands(tmp_path / "m.png"))


def test_a_run_of_its_own_corrected_by_hand_is_read_back(asset, tmp_path, zones_file):
    from image_to_3dlab.materials import make_game_materials

    first = make_game_materials(asset, tmp_path / "first", zones=material_zones.zones_from_file(asset, *zones_file)[0])
    record_path = first.sidecar_path
    record = json.loads(record_path.read_text())
    record["materials"][0]["zones"][1]["material"] = "wood"     # the person renames a zone
    record_path.write_text(json.dumps(record))
    tint = tmp_path / "first" / record["materials"][0]["textures"]["tint"]
    found, _ = material_zones.zones_from_file(asset, record_path, tint)
    (zone,) = found.values()
    assert "wood" in zone.zones() and "leather" not in zone.zones()


# ---------------------------------------------------------------------- command line

def test_the_command_line_takes_a_zones_file(asset, tmp_path, zones_file):
    out = tmp_path / "out"
    code = game_materials.main([str(asset), str(out), "--zones-file", str(zones_file[0]),
                                "--zones-mask", str(zones_file[1])])
    assert code == 0
    record = json.loads((out / "chest.materials.json").read_text())
    assert [z["material"] for z in record["zones"]] and (out / "T_chest_Tint.png").is_file()
    provenance = json.loads((out / "chest.provenance.json").read_text())
    assert "zones.json" in json.dumps(provenance)   # where the zones came from


def test_the_command_line_wants_one_source_of_zones(asset, tmp_path, zones_file, capsys):
    with pytest.raises(SystemExit):
        game_materials.main([str(asset), str(tmp_path / "o"), "--zones", "--zones-file", str(zones_file[0]),
                             "--zones-mask", str(zones_file[1])])
    assert game_materials.main([str(asset), str(tmp_path / "o"), "--zones-file", str(zones_file[0])]) == 1
    assert "zones-mask" in capsys.readouterr().out


# ---------------------------------------------------------------------- batches and agents

def _row_spec(tmp_path, asset, materials=None, steps=("materials",)):
    spec = {"schema_version": 1, "name": "mats", "output": {"directory": "out"},
            "rows": [{"id": "SM_Chest", "asset": str(asset), "kind": "prop", "steps": list(steps),
                      **({"materials": materials} if materials is not None else {})}]}
    path = tmp_path / "mats.json"
    path.write_text(json.dumps(spec))
    return path


def test_the_batch_step_packs_the_maps_and_hands_the_model_on(tmp_path, asset, monkeypatch):
    install_fake_generator(monkeypatch)
    report = batch.run_batch(batch.load_batch(_row_spec(tmp_path, asset)))
    assert report.exit_code == 0, report.failed
    folder = tmp_path / "out" / "mats" / "rows" / "SM_Chest" / "materials"
    assert (folder / "chest.glb").is_file() and (folder / "chest.materials.json").is_file()
    assert not (folder / "T_chest_Tint.png").exists()   # zones are opt-in


def test_the_batch_step_takes_a_zones_file_of_your_own(tmp_path, asset, zones_file, monkeypatch):
    install_fake_generator(monkeypatch)
    spec = _row_spec(tmp_path, asset, {"zones_file": str(zones_file[0]), "zones_mask": str(zones_file[1])},
                     steps=("materials", "export_unreal"))
    seen = []

    def export(ctx):
        from image_to_3dlab.batch_steps import StepResult
        seen.append(ctx.asset)
        return StepResult()

    from batch_fakes import fake_step
    monkeypatch.setitem(batch.STEPS, "export_unreal", fake_step(export))
    report = batch.run_batch(batch.load_batch(spec))
    assert report.exit_code == 0, report.failed
    (exported,) = seen
    assert exported.parent.name == "materials" and (exported.parent / "T_chest_Tint.png").is_file()
    # the mask edited in place is a change: the step runs again
    before = batch.load_batch(spec).rows[0].step_fingerprints()["materials"]
    _bands(zones_file[1], values=(3, 2, 1))
    assert batch.load_batch(spec).rows[0].step_fingerprints()["materials"] != before


def test_a_zones_file_that_is_not_there_fails_its_row(tmp_path, asset, monkeypatch):
    install_fake_generator(monkeypatch)
    report = batch.run_batch(batch.load_batch(_row_spec(tmp_path, asset, {"zones_file": "missing.json",
                                                                         "zones_mask": "missing.png"})))
    assert report.exit_code != 0 and "missing.json" in json.dumps(report.failed)


@pytest.mark.parametrize("options, problem", [
    ({"zones": "sam31", "zones_file": "z.json"}, "one source"),
    ({"zones_file": "z.json"}, "zones_mask"),
    ({"materials": ["gold"]}, "materials"),
])
def test_odd_materials_settings_stop_the_batch_first(tmp_path, asset, options, problem, monkeypatch):
    install_fake_generator(monkeypatch)
    (tmp_path / "z.json").write_text("{}")
    with pytest.raises(batch.BatchError, match=problem):
        batch.run_batch(batch.load_batch(_row_spec(tmp_path, asset, options)))


def test_the_step_is_described_for_the_studio_and_agents():
    spec = step_catalog.get("materials")
    assert spec.feature == 8 and spec.cli == "scripts/game_materials.py" and spec.skip
    names = {param.name for param in spec.params}
    assert {"zones", "materials", "zones_file", "zones_mask", "texture_prefix", "ao"} <= names
    assert step_catalog.check("materials", {"zones": "sam9"})[0].startswith("zones must be one of")


def test_every_materials_setting_is_a_flag_of_the_script_and_back():
    flags = {flag for action in game_materials.build_parser()._actions for flag in action.option_strings}
    described = {param.cli_flag for param in step_catalog.get("materials").params if param.cli_flag}
    assert described <= flags
    assert flags - described - {"-h", "--help", "--work"} == set()


def test_an_agent_runs_the_step_with_a_zones_file(tmp_path, asset, zones_file, monkeypatch):
    from image_to_3dlab import mcp_server

    seen = {}

    def submit(spec_path, **kwargs):
        seen["spec"] = json.loads(Path(spec_path).read_text())
        return {"started": True}

    monkeypatch.setattr(mcp_server, "submit_batch", submit)
    monkeypatch.setattr(mcp_server, "AGENT_SPECS", tmp_path / "agent")
    described = {step["name"]: step for step in mcp_server.describe_steps()["steps"]}
    assert described["materials"]["available"] is True
    result = mcp_server.run_steps("SM_Chest", ["materials"], asset=str(asset), options={
        "materials": {"zones_file": str(zones_file[0]), "zones_mask": str(zones_file[1])}})
    assert result["started"] is True
    (row,) = seen["spec"]["rows"]
    assert row["materials"] == {"zones_file": str(zones_file[0]), "zones_mask": str(zones_file[1])}


def test_the_glb_written_keeps_the_mask_as_a_texture(asset, tmp_path, zones_file):
    out = tmp_path / "o"
    assert game_materials.main([str(asset), str(out), "--zones-file", str(zones_file[0]),
                                "--zones-mask", str(zones_file[1])]) == 0
    document, _ = parse_glb((out / "chest.glb").read_bytes())
    extras = document["materials"][0]["extras"]["assetfurnace"]
    assert extras["zones"] and "tintMask" in extras
    assert surfaces(EditGLB((out / "chest.glb").read_bytes()))
