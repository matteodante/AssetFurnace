"""The studio's Game materials panel: the materials step on the model shown, its maps packed
as an ORM, tint zones found by a segmenter or given as a file of your own (section 9 of
docs/plans/2026-10-17-step-parity-design.md), and the mask it makes offered to Export for
Unreal. No zone painting; no model runs here."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_materials import banded_albedo, make_glb
from test_materials_step import _bands

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import materials_api as api  # noqa: E402

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


@pytest.fixture
def output(tmp_path):
    root = tmp_path / "output"
    run = root / "finish" / "chest__finish__20261017-120000"
    run.mkdir(parents=True)
    model = make_glb(run / "chest_20k.glb", banded_albedo())
    model.with_suffix(".provenance.json").write_text(json.dumps({"license": {"name": "MIT"}}))
    return root


MODEL = "finish/chest__finish__20261017-120000/chest_20k.glb"


def _own(tmp_path):
    mask = _bands(tmp_path / "zones.png")
    return {"zones_file": {"name": "zones.json", "text": json.dumps({"zones": {"1": "metal", "2": "leather",
                                                                              "3": "cloth"}})},
            "zones_mask": {"name": "zones.png", "data": base64.b64encode(mask.read_bytes()).decode()}}


def test_packing_the_maps_only(output):
    job = api.start(output, api.parse(output, {"file": MODEL, "zones": "off"}), background=False)
    assert job["status"] == "done", job
    (run,) = api.runs_for(output, MODEL)
    assert run["glb"].endswith("chest_20k.glb") and run["tint"] is None and run["status"] == "pass"
    assert run["folder"].startswith("materials/chest_20k__materials__")


def test_zones_of_your_own_make_the_tint_mask(output, tmp_path):
    job = api.start(output, api.parse(output, {"file": MODEL, "zones": "file", **_own(tmp_path)}), background=False)
    assert job["status"] == "done", job
    (run,) = api.runs_for(output, MODEL)
    assert run["tint"].endswith("_Tint.png") and set(run["zones"]) == {"metal", "leather", "cloth"}
    folder = output / run["folder"]
    assert (folder / "input" / "zones.json").is_file() and (folder / "input" / "zones.png").is_file()
    provenance = json.loads((output / run["glb"]).with_suffix(".provenance.json").read_text())
    assert provenance["zones_from"]["zones_file"]["name"] == "zones.json"


def test_a_run_from_another_model_is_not_offered(output, tmp_path):
    api.start(output, api.parse(output, {"file": MODEL, "zones": "off"}), background=False)
    other = make_glb(output / "finish" / "chest__finish__20261017-120000" / "other.glb", banded_albedo(noise=9))
    assert api.runs_for(output, other.relative_to(output).as_posix()) == []


@pytest.mark.parametrize("data, words", [
    ({"file": "../outside.glb", "zones": "off"}, "Library"),
    ({"file": MODEL, "zones": "sam9"}, "zones"),
    ({"file": MODEL, "zones": "file"}, "zones file"),
    ({"file": MODEL, "zones": "file", "zones_file": {"name": "z.json", "text": "{}"}}, "mask"),
    ({"file": MODEL, "zones": "sam31", "materials": "gold"}, "gold"),
])
def test_odd_requests_are_refused_with_why(output, data, words):
    with pytest.raises(ValueError, match=words):
        api.parse(output, data)


def test_a_wrong_zones_file_fails_the_job_with_why(output, tmp_path):
    own = _own(tmp_path)
    own["zones_file"]["text"] = json.dumps({"zones": {"1": "gold"}})
    job = api.start(output, api.parse(output, {"file": MODEL, "zones": "file", **own}), background=False)
    assert job["status"] == "error" and "gold" in job["error"]
    assert api.runs_for(output, MODEL) == []   # nothing half-made is offered


def test_the_zones_models_missing_are_named(output, monkeypatch):
    from image_to_3dlab import material_zones

    monkeypatch.setattr(material_zones, "readiness", lambda segmenter, ask=True: ["run: bootstrap_materials.py"])
    with pytest.raises(ValueError, match="bootstrap_materials"):
        api.parse(output, {"file": MODEL, "zones": "sam31"})


@needs_node
def test_the_panel_builds_its_request_and_the_export_takes_the_mask():
    panel = (REPO / "viewer" / "studio" / "game-materials.js").as_uri()
    export = (REPO / "viewer" / "studio" / "unreal-export.js").as_uri()
    program = (
        f"import * as m from {json.dumps(panel)};\n"
        f"import * as u from {json.dumps(export)};\n"
        "const out = {};\n"
        "out.off = m.requestFor({ file: 'a.glb', zones: 'off', materials: '' });\n"
        "out.sam = m.requestFor({ file: 'a.glb', zones: 'sam31', materials: 'leather, metal' });\n"
        "try { m.requestFor({ file: 'a.glb', zones: 'file' }); } catch (e) { out.missing = e.message; }\n"
        "out.own = m.requestFor({ file: 'a.glb', zones: 'file', zonesFile: { name: 'z.json', text: '{}' },"
        " zonesMask: { name: 'z.png', data: 'AAAA' } });\n"
        "out.tint = u.requestBody({ source: { file: 'a.glb', moves: [] }, kind: 'prop', size: 90, measure: 'longest',"
        " tint: 'materials/r/T_a_Tint.png' }).tint;\n"
        "out.none = u.requestBody({ source: { file: 'a.glb', moves: [] }, kind: 'prop', size: 90, measure: 'longest',"
        " tint: 'none' }).tint;\n"
        "out.options = u.tintOptions([{ tint: 'materials/r/T_a_Tint.png', zones: ['metal', 'leather'], created: 0 },"
        " { tint: null, zones: [], created: 0 }]);\n"
        "console.log(JSON.stringify(out));")
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True,
                            text=True, encoding="utf-8")
    out = json.loads(result.stdout)
    assert out["off"] == {"file": "a.glb", "zones": "off"}
    assert out["sam"]["materials"] == "leather, metal"
    assert "zones file" in out["missing"]
    assert out["own"]["zones_file"]["name"] == "z.json" and out["own"]["zones_mask"]["data"] == "AAAA"
    assert out["tint"] == {"file": "materials/r/T_a_Tint.png"} and out["none"] == "none"
    assert [o[0] for o in out["options"]] == ["materials/r/T_a_Tint.png"] and "metal" in out["options"][0][1]
