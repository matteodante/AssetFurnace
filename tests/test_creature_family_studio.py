"""The studio's creature-family controls (viewer/studio/creature-family.js) and their hooks
in steps.js, run through Node like the other studio modules."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from image_to_3dlab import creature_family as cf

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def _run(module: str, body: str):
    url = (REPO / "viewer" / "studio" / module).as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


WOLF = {"name": "wolf", "kind": "character", "stage": "animated", "rigged": "animate/w/w_rigged.glb",
        "fits_moves": True, "family": "digitigrade", "family_label": "Two legs, digitigrade (werewolf, demon)",
        "family_chosen": False, "clips": [{"name": "digitigrade-walk"}, {"name": "walk"}], "props": []}


@needs_node
def test_the_picker_offers_every_family_the_server_knows():
    out = _run("creature-family.js", "console.log(JSON.stringify({ ids: m.FAMILIES.map((f) => f[0]), html: m.familyPicker() }));")
    assert out["ids"] == ["auto", *cf.FAMILIES]
    assert 'data-k="family"' in out["html"] and 'data-run="rigged"' in out["html"]


@needs_node
def test_the_rig_step_asks_for_the_family_and_animate_names_the_test_clips():
    out = _run("steps.js", f"""
      const wolf = {json.dumps(WOLF)};
      const plant = {{ ...wolf, family: 'plant', family_label: 'Plant', fits_moves: false, clips: [], clips_error: null }};
      const ctx = {{ presets: [{{ id: 'walk', label: 'Walk' }}], engines: [] }};
      console.log(JSON.stringify({{ rig: m.stepBody('rigged', wolf, ctx), wolf: m.stepBody('animated', wolf, ctx),
        plant: m.stepBody('animated', plant, ctx), human: m.stepBody('animated', {{ ...wolf, family: 'humanoid' }}, ctx) }}));""")
    assert 'data-k="family"' in out["rig"] and "humanoid?" not in out["rig"]
    assert "digitigrade-walk" in out["wolf"] and 'data-preset="walk"' in out["wolf"]  # test clips and preset moves
    assert "no test clips yet" in out["plant"] and "Preset moves" not in out["plant"]
    assert "Rigged as" not in out["human"]


@needs_node
def test_animate_says_which_clips_come_from_the_library_and_which_are_tests():
    wolf = {**WOLF, "clips": [{"name": "digitigrade-walk"}, {"name": "digitigrade-walk-long"},
                              {"name": "digitigrade-trot"}, {"name": "walk"}],
            "library_clips": ["digitigrade-walk", "digitigrade-walk-long"]}
    out = _run("creature-family.js", f"console.log(JSON.stringify(m.familyNote({json.dumps(wolf)})));")
    assert "CC0 creature library: digitigrade-walk, digitigrade-walk-long" in out
    assert "Test clips to check the rig: digitigrade-trot" in out
