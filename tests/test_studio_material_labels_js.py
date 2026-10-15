"""The labels module's pure parts under Node: a click becomes a material point."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def run(body: str):
    url = (REPO / "viewer" / "studio" / "material-labels.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


@needs_node
def test_a_hit_becomes_a_point_on_the_paint_not_on_a_look():
    """Clay and normal looks swap the mesh's material; the point names the paint's."""
    out = run("""
      const paint = {name: 'skin'}, clay = {name: 'clay'};
      const mesh = {material: clay, userData: {paint}};
      const hit = {object: mesh, uv: {x: 1.25, y: 0.5}, face: {materialIndex: 0}};
      const index = (material) => material === paint ? 3 : null;
      console.log(JSON.stringify(m.pointFromHit(hit, 'leather', index)));""")
    assert out == {"uv": [0.25, 0.5], "material": 3, "label": "leather"}


@needs_node
def test_a_multi_material_mesh_uses_the_face_group():
    out = run("""
      const a = {}, b = {};
      const mesh = {material: [a, b], userData: {}};
      const hit = {object: mesh, uv: {x: 0.1, y: 0.9}, face: {materialIndex: 1}};
      console.log(JSON.stringify(m.pointFromHit(hit, 'metal', (x) => x === b ? 7 : 0)));""")
    assert out == {"uv": [0.1, 0.9], "material": 7, "label": "metal"}


@needs_node
def test_a_hit_without_uv_gives_no_point():
    out = run("console.log(JSON.stringify(m.pointFromHit({object: {material: {}}, face: {}}, 'metal', () => 0)));")
    assert out is None


@needs_node
def test_completeness_matches_the_python_rule():
    out = run("""console.log(JSON.stringify([
      m.isComplete({baked_light: false, materials_look: 'right', points: [{}]}),
      m.isComplete({baked_light: null, materials_look: 'right', points: [{}]}),
      m.isComplete({baked_light: true, materials_look: 'wrong', points: []})]));""")
    assert out == [True, False, False]


@needs_node
def test_a_click_is_told_from_a_drag():
    out = run("console.log(JSON.stringify([m.isClick({x: 10, y: 10}, {x: 12, y: 13}), m.isClick({x: 10, y: 10}, {x: 30, y: 10})]));")
    assert out == [True, False]


@needs_node
def test_choice_buttons_keep_their_values_intact_in_html():
    """'"right"' inside a double-quoted attribute broke the Materials look buttons."""
    out = run("""console.log(JSON.stringify(m.choiceButton('materials_look', 'right', 'Right', 'right')));""")
    assert 'data-value="&quot;right&quot;"' in out and 'aria-pressed="true"' in out
