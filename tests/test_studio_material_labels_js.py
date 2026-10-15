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


@needs_node
def test_only_items_of_the_label_set_get_the_panel():
    out = run("""console.log(JSON.stringify([m.inLabelSet({source: 'material-labels'}),
      m.inLabelSet({source: 'batch:props'}), m.inLabelSet(null)]));""")
    assert out == [True, False, False]


@needs_node
def test_a_point_and_its_marker_leave_together():
    """Saved points have no marker (null); removing any row removes its own marker only."""
    out = run("""
      const points = ['saved', 'new1', 'new2'], markers = [null, 'm1', 'm2'];
      const removed = m.dropAt(points, markers, 1);
      console.log(JSON.stringify({removed, points, markers}));""")
    assert out == {"removed": "m1", "points": ["saved", "new2"], "markers": [None, "m2"]}


@needs_node
def test_a_point_follows_the_paints_uv_channel_transform_and_wrap():
    """three.js samples map.channel's uv through map.matrix, then wraps: the point does the same."""
    out = run("""
      const repeat = 1000, clamp = 1001, mirror = 1002;
      // offset 0.1 in u, as Matrix3.setUvTransform builds it (column-major elements)
      const map = {channel: 1, matrix: {elements: [1, 0, 0, 0, 1, 0, 0.1, 0, 1]}, wrapS: repeat, wrapT: repeat};
      const paint = {map};
      const hit = {object: {material: paint, userData: {}}, uv: {x: 0.5, y: 0.5}, uv1: {x: 0.95, y: 0.25}, face: {}};
      const a = m.pointFromHit(hit, 'metal', () => 0);
      const clamped = m.pointFromHit({...hit, object: {material: {map: {...map, wrapS: clamp}}, userData: {}}}, 'metal', () => 0);
      const mirrored = m.pointFromHit({...hit, object: {material: {map: {...map, wrapS: mirror}}, userData: {}}}, 'metal', () => 0);
      console.log(JSON.stringify([a.uv, clamped.uv, mirrored.uv]));""")
    assert out == [[0.05, 0.25], [1.0, 0.25], [0.95, 0.25]]
