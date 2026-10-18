"""Studio's Export for Unreal panel: its pure choices, run under Node on the shipped file."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(body: str):
    url = (REPO / "viewer" / "studio" / "unreal-export.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


KNIGHT = {"id": "k", "name": "knight", "kind": "character", "rigged": "animate/k/k_rigged.glb",
          "finished": "finish/k/k.glb", "clips": [{"name": "idle", "file": "animate/k/k_idle.glb"}]}
BARREL = {"id": "b", "name": "tavern", "kind": "prop set", "finished": None, "model": "a/s.glb", "rigged": None,
          "clips": [], "props": [{"name": "barrel", "file": "props/s/barrel_LOD0.glb"}]}


@needs_node
def test_a_character_exports_its_rig_and_moves_and_a_prop_what_is_on_screen():
    out = _run(f"""
      console.log(JSON.stringify({{
        knight: m.exportSource({json.dumps(KNIGHT)}, 'animate/k/k_idle.glb'),
        barrel: m.exportSource({json.dumps(BARREL)}, 'props/s/barrel_LOD0.glb'),
        none: m.exportSource(null, null),
      }}));""")
    assert out["knight"] == {"file": "animate/k/k_rigged.glb", "kind": "creature", "moves": ["animate/k/k_idle.glb"]}
    assert out["barrel"] == {"file": "props/s/barrel_LOD0.glb", "kind": "prop", "moves": []}
    assert out["none"] is None


@needs_node
def test_equipment_can_be_worn_by_any_other_rigged_character():
    out = _run(f"""console.log(JSON.stringify(m.bodiesFor({json.dumps([KNIGHT, BARREL])}, {{id: 'x'}})));""")
    assert out == [{"name": "knight", "file": "animate/k/k_rigged.glb"}]
    out = _run(f"""console.log(JSON.stringify(m.bodiesFor({json.dumps([KNIGHT])}, {json.dumps(KNIGHT)})));""")
    assert out == []


@needs_node
def test_the_request_carries_moves_only_for_creatures_and_a_body_only_for_equipment():
    out = _run("""
      const source = {file: 'a.glb', kind: 'creature', moves: ['m.glb']};
      console.log(JSON.stringify({
        creature: m.requestBody({source, kind: 'creature', size: '180', measure: 'height', moves: true, check: false}),
        noMoves: m.requestBody({source, kind: 'creature', size: '180', measure: 'height', moves: false}),
        equipment: m.requestBody({source, kind: 'equipment', size: '180', measure: 'height', body: 'b.glb'}),
        sizes: [m.sizeProblem(''), m.sizeProblem('0'), m.sizeProblem('abc'), m.sizeProblem('60')],
      }));""")
    assert out["creature"] == {"file": "a.glb", "kind": "creature", "size_cm": 180, "measure": "height",
                               "check": False, "animations": ["m.glb"]}
    assert "animations" not in out["noMoves"]
    assert out["equipment"]["body"] == "b.glb" and "animations" not in out["equipment"]
    assert [bool(x) for x in out["sizes"]] == [True, True, True, False]


# --- review round 1 ----------------------------------------------------------------------

@needs_node
def test_a_selected_blender_version_is_what_gets_exported():
    out = _run(f"""
      const version = {{id: 'v1', file: 'blender-edits/versions/v1/model.glb'}};
      console.log(JSON.stringify(m.exportSource({json.dumps(KNIGHT)}, version.file, version)));""")
    assert out == {"file": "blender-edits/versions/v1/model.glb", "kind": "creature", "moves": []}


@needs_node
def test_a_blender_version_without_a_skeleton_goes_out_as_an_object():
    """Code read 18/10/2026: a character's version that lost its skeleton went out as a creature."""
    out = _run(f"""
      const bare = {{id: 'v2', file: 'blender-edits/versions/v2/model.glb', facts: {{skins: 0, joints: 0}}}};
      const rigged = {{id: 'v3', file: 'blender-edits/versions/v3/model.glb', facts: {{skins: 1, joints: 40}}}};
      console.log(JSON.stringify([m.exportSource({json.dumps(KNIGHT)}, bare.file, bare).kind,
        m.exportSource({json.dumps(BARREL)}, rigged.file, rigged).kind]));""")
    assert out == ["prop", "creature"]


@needs_node
def test_a_new_move_makes_a_different_request():
    out = _run("""
      const a = {file: 'r.glb', kind: 'creature', moves: ['idle.glb']};
      console.log(JSON.stringify({
        same: m.sameSource(a, {...a, moves: ['idle.glb']}),
        moreMoves: m.sameSource(a, {...a, moves: ['idle.glb', 'walk.glb']}),
        otherFile: m.sameSource(a, {...a, file: 'x.glb'}),
        none: m.sameSource(a, null),
      }));""")
    assert out == {"same": True, "moreMoves": False, "otherFile": False, "none": False}


# --- review round 2 ----------------------------------------------------------------------

@needs_node
def test_equipment_asks_for_the_wearers_height_and_is_not_checked_alone():
    out = _run("""
      const source = {file: 'helm.glb', kind: 'prop', moves: []};
      console.log(JSON.stringify({
        equipment: m.sizeLabel('equipment'), object: m.sizeLabel('prop'),
        request: m.requestBody({source, kind: 'equipment', size: '180', measure: 'height', body: 'b.glb', check: true}),
      }));""")
    assert out["equipment"] == "Wearer's height (cm)" and out["object"] == "Real size (cm)"
    assert out["request"]["check"] is False


# --- review round 3 ----------------------------------------------------------------------

@needs_node
def test_another_prop_of_the_same_set_needs_its_own_size():
    out = _run(f"""
      const a = m.exportSource({json.dumps(BARREL)}, 'props/s/barrel_LOD0.glb');
      const b = m.exportSource({json.dumps(BARREL)}, 'props/s/chair_LOD0.glb');
      console.log(JSON.stringify({{keep: m.keepsSize(a, a), other: m.keepsSize(a, b), none: m.keepsSize(null, a)}}));""")
    assert out == {"keep": True, "other": False, "none": False}


@needs_node
def test_the_request_carries_a_chosen_material_and_auto_names_the_rules_pick():
    out = _run("""
      const source = {file: 'a.glb', kind: 'prop', moves: []};
      console.log(JSON.stringify({
        chosen: m.requestBody({source, kind: 'prop', size: '60', measure: 'longest', material: 'metal'}),
        auto: m.requestBody({source, kind: 'prop', size: '60', measure: 'longest', material: ''}),
        labels: [m.autoLabel({material: 'wood', reason: 'default for a prop'}), m.autoLabel(null)],
      }));""")
    assert out["chosen"]["material"] == "metal" and "material" not in out["auto"]
    assert out["labels"] == ["Auto: wood (default for a prop)", "Auto"]
