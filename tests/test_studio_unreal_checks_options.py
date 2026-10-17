"""The studio's Export for Unreal gets the export's options (name, collision, detail levels,
tint mask, facing), and the review sheet the checks' thresholds, as the command line and
the batches have them (docs/plans/2026-10-17-step-parity-design.md, section 7, gap 4)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import _check_fixtures as fx
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
sys.path.insert(0, str(REPO / "scripts"))
import export_unreal  # noqa: E402
import review_api  # noqa: E402
import unreal_export_api as api  # noqa: E402

from image_to_3dlab import step_catalog  # noqa: E402

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


@pytest.fixture
def output(tmp_path):
    root = tmp_path / "output"
    (root / "props" / "p").mkdir(parents=True)
    for name in ("barrel_LOD0.glb", "barrel_LOD1.glb"):
        (root / "props" / "p" / name).write_bytes(b"glb")
    (root / "props" / "p" / "T_barrel_LOD0_Tint.png").write_bytes(b"png")
    (root / "props" / "p" / "mask.png").write_bytes(b"png")
    (root / "animate" / "w").mkdir(parents=True)
    (root / "animate" / "w" / "w_rigged.glb").write_bytes(b"glb")
    return root


def test_the_panels_options_become_the_exports_arguments(output):
    request = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 90,
                                 "name": "Barrel Old", "collision": "convex", "lods": "none", "tint": "none"})
    assert request["name"] == "BarrelOld" and request["collision"] == "convex"
    assert request["lods"] == [] and request["tint"] is False
    auto = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 90,
                              "collision": "auto", "lods": "auto", "tint": "auto"})
    assert "collision" not in auto and "lods" not in auto and "tint" not in auto
    mine = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 90,
                              "tint": {"file": "props/p/mask.png"}})
    assert mine["tint"] == (output / "props" / "p" / "mask.png").resolve()
    creature = api.parse(output, {"file": "animate/w/w_rigged.glb", "kind": "creature", "size_cm": 180,
                                  "forward": "x"})
    assert creature["forward"] == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("data, words", [
    ({"collision": "sphere"}, "collision"),
    ({"lods": "some"}, "detail levels"),
    ({"tint": {"file": "../outside.png"}}, "tint"),
    ({"forward": "up"}, "facing"),
])
def test_bad_options_are_explained(output, data, words):
    with pytest.raises(ValueError, match=words):
        api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 90, **data})


def test_no_tint_mask_means_none_even_with_one_beside_the_model(output):
    source = output / "props" / "p" / "barrel_LOD0.glb"
    assert export_unreal.plan(source, kind="prop", size_cm=90)["tint_source"] is not None   # found beside it
    assert export_unreal.plan(source, kind="prop", size_cm=90, tint=False)["tint_source"] is None


def test_the_command_line_says_no_tint_with_none(output, monkeypatch):
    seen = {}
    monkeypatch.setattr(export_unreal, "export_unreal", lambda source, **kw: seen.update(kw) or {
        "ok": True, "glb": "x", "out_dir": "x", "sidecar": "x", "problems": []})
    export_unreal.main([str(output / "props" / "p" / "barrel_LOD0.glb"), "--kind", "prop", "--size-cm", "90",
                        "--tint", "none"])
    assert seen["tint"] is False


def test_a_batch_row_can_say_no_tint_too():
    assert step_catalog.check("export_unreal", {"tint": False}) == []
    assert step_catalog.get("export_unreal").param("tint").off


# ------------------------------------------------------------------ thresholds

def test_the_review_sheet_reads_the_profiles_thresholds():
    rows = review_api.thresholds("creature")
    faces = next(r for r in rows if r["check"] == "budget.faces" and r["setting"] == "max")
    assert faces["value"] == 30000 or isinstance(faces["value"], (int, float))
    assert not any(r["check"] == "license.components" for r in rows)   # the licence gate is not a threshold
    assert all(isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool) for r in rows)


def test_checks_run_from_the_sheet_with_other_thresholds(tmp_path):
    output = tmp_path / "output"
    run = output / "barrel__pixal3d__1"
    run.mkdir(parents=True)
    glb = fx.clean(run / "barrel__pixal3d__1.glb")
    item = review_api.run_checks_request(output, {"asset": "barrel__pixal3d__1/barrel__pixal3d__1.glb",
                                                  "key": "barrel", "profile": "prop",
                                                  "set": {"budget.faces.max": 10}})
    faces = next(r for r in item["report"]["results"] if r["check"] == "budget.faces")
    assert faces["status"] == "fail" and glb.is_file()
    with pytest.raises(ValueError, match="budget.facez"):
        review_api.run_checks_request(output, {"asset": "barrel__pixal3d__1/barrel__pixal3d__1.glb",
                                               "set": {"budget.facez.max": 10}})


def _run(module: str, body: str):
    url = (REPO / "viewer" / "studio" / module).as_uri()
    result = subprocess.run([NODE, "--input-type=module", "--eval", f"import * as m from {json.dumps(url)};\n{body}"],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


@needs_node
def test_the_unreal_panel_sends_its_options():
    out = _run("unreal-export.js", """
      const source = { file: 'props/p/barrel_LOD0.glb', kind: 'prop', moves: [] };
      console.log(JSON.stringify([
        m.requestBody({ source, kind: 'prop', size: 90, measure: 'longest', name: ' Barrel ', collision: 'convex',
                        lods: 'none', tint: 'none', forward: 'auto' }),
        m.requestBody({ source, kind: 'prop', size: 90, measure: 'longest' }),
      ]));""")
    chosen, plain = out
    assert chosen["name"] == "Barrel" and chosen["collision"] == "convex"
    assert chosen["lods"] == "none" and chosen["tint"] == "none" and "forward" not in chosen
    assert not {"name", "collision", "lods", "tint", "forward"} & set(plain)


@needs_node
def test_the_sheet_sends_only_the_thresholds_that_changed():
    out = _run("review-model.js", """
      const rows = [{ check: 'budget.faces', setting: 'max', value: 40000 }, { check: 'uv.overlap', setting: 'warn', value: 4 }];
      console.log(JSON.stringify(m.checksRequest({ file: 'a.glb', id: 'a' }, { profile: 'prop',
        thresholds: rows, edited: { 'budget.faces.max': '20000', 'uv.overlap.warn': '4' } })));""")
    assert out["set"] == {"budget.faces.max": 20000}
