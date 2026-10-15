"""The studio's materials module under Node: the live tint must be the shipped formula.

The browser recolours with a shader; the numbers it feeds that shader, and the formula it
runs, are checked here against ``image_to_3dlab.materials.apply_tint`` and the Blender
preview's ``tint_terms``, so the studio, Blender and Unreal cannot drift apart.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from blender_tint_preview import tint_terms

from image_to_3dlab import materials as m

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def run(body: str):
    url = (REPO / "viewer" / "studio" / "materials.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


ZONES = [{"channel": "R", "mean_linear": [0.2, 0.05, 0.04]},
         {"channel": "G", "mean_linear": [0.1, 0.1, 0.1]},
         {"channel": "B", "mean_linear": [0.02, 0.03, 0.2]}]


@needs_node
def test_hex_and_linear_round_trip_like_python():
    out = run("""console.log(JSON.stringify({
      lin: m.hexToLinear('#808080'), hex: m.linearToHex(m.hexToLinear('#1f3a7a')),
      zero: m.linearToHex([0, 0, 0]), clip: m.linearToHex([2, -1, 0.5])}));""")
    assert out["lin"] == pytest.approx([float(m.srgb_to_linear(np.array(128 / 255)))] * 3)
    assert out["hex"] == "#1f3a7a"
    assert out["zero"] == "#000000"
    assert out["clip"] == "#ff00bc"


@needs_node
def test_shader_constants_match_the_blender_preview():
    tints = {"R": [0.02, 0.05, 0.3], "B": [0.5, 0.4, 0.1]}
    out = run(f"console.log(JSON.stringify(m.tintUniforms({json.dumps(ZONES)}, {json.dumps(tints)})));")
    terms = {t["channel"]: t for t in tint_terms(ZONES, tints)}
    for i, channel in enumerate("RGB"):
        if channel in terms:
            assert out["scale"][i] == pytest.approx(terms[channel]["scale"])
            assert out["delta"][i] == pytest.approx(terms[channel]["delta"])
        else:
            assert out["delta"][i] == [0, 0, 0]       # an untouched zone adds nothing


@needs_node
def test_live_tint_is_the_shipped_formula():
    rng = np.random.default_rng(0)
    albedo = rng.uniform(0, 0.6, (16, 3))
    mask = rng.uniform(0, 1, (16, 3)) * (rng.uniform(size=(16, 1)) > 0.3)
    tints = {"R": [0.02, 0.05, 0.3], "G": [0.3, 0.3, 0.05]}
    out = run(f"""
      const u = m.tintUniforms({json.dumps(ZONES)}, {json.dumps(tints)});
      const a = {json.dumps(albedo.tolist())}, k = {json.dumps(mask.tolist())};
      console.log(JSON.stringify(a.map((px, i) => m.tintPixel(px, k[i], u))));""")
    means = np.array([z["mean_linear"] for z in ZONES])
    wanted = means.copy()
    for i, channel in enumerate("RGB"):
        if channel in tints:
            wanted[i] = tints[channel]
    expected = m.apply_tint(albedo[:, None, :], mask[:, None, :], means, wanted)[:, 0, :]
    np.testing.assert_allclose(np.array(out), expected, atol=1e-9)


@needs_node
def test_shader_patch_samples_the_mask_after_the_base_colour_once():
    out = run("""
      const shader = {fragmentShader: 'uniform vec3 diffuse;\\nvoid main(){\\n#include <map_fragment>\\n}', uniforms: {}};
      m.patchShader(shader, {afTintMask: {value: null}});
      const once = shader.fragmentShader;
      m.patchShader(shader, {afTintMask: {value: null}});
      console.log(JSON.stringify({src: once, same: once === shader.fragmentShader, keys: Object.keys(shader.uniforms)}));""")
    src = out["src"]
    assert src.index("uniform sampler2D afTintMask") < src.index("void main")
    assert src.index("#include <map_fragment>") < src.index("texture2D( afTintMask, vMapUv )")
    assert "0.2126" in src and "clamp(" in src
    assert out["same"] is True
    assert out["keys"] == ["afTintMask"]


@needs_node
def test_check_rows_read_plainly():
    checks = [{"check": "tint_zones", "status": "pass", "value": 3, "message": "3 recolourable zones"},
              {"check": "baked_light_occlusion", "status": "skipped", "value": None, "message": "no bake"},
              {"check": "baked_light_direction", "status": "fail", "value": 1.6, "message": "lit side 1.60x"}]
    out = run(f"console.log(JSON.stringify(m.checkRows({json.dumps(checks)})));")
    assert [r["label"] for r in out] == ["Tint zones", "Painted light", "Painted shadows in creases"]
    assert [r["status"] for r in out] == ["pass", "fail", "skipped"]   # fixed order, worst visible


@needs_node
def test_the_step_runs_on_the_model_the_paint_comes_from():
    out = run("""console.log(JSON.stringify([
      m.materialSource({finished:'f.glb', model:'m.glb', rigged:'r.glb'}),
      m.materialSource({model:'m.glb'}),
      m.materialSource({rigged:'r.glb'}),
      m.materialSource({kind:'prop set', props:[{file:'p.glb'}], model:'s.glb'}),
      m.materialSource({finished:'f.glb'}, {file:'v.glb'}),
      m.materialSource({picture:'p.png'})]));""")
    assert out == ["f.glb", "m.glb", "r.glb", "p.glb", "v.glb", None]
