"""The studio redoes a finished step with new values, keeping the steps before it
(docs/plans/2026-10-17-step-parity-design.md, section 7): the Library says what each step
was made with and which later steps came from an earlier version; the Steps panel offers
"Redo" on every finished step; Create chooses the 3D seed and the triangle count."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import activity_api  # noqa: E402
import assets_api  # noqa: E402

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")

IMG = "a-knight__20261005-205524"
GEN = f"{IMG}-cutout__pixal3d__20261005-210000"
GEN2 = f"{IMG}-cutout__pixal3d__20261006-090000"
FIN = f"{IMG}-cutout__pixal3d__202610__finish__20261005-210615"
RIG = "a-knight_40k__rig__20261005-210643"


def _write(path: Path, data: bytes | str, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    _write(out / "images" / IMG / f"{IMG}.png", b"png", 1000)
    _write(out / "conditional" / GEN / f"{GEN}.glb", b"MODEL ONE", 2000)
    _write(out / "conditional" / GEN / f"{GEN}.json",
           json.dumps({"backend": "pixal3d", "parameters": {"seed": 77, "steps": 30}}), 2000)
    _write(out / "finish" / FIN / "a-knight_15k.glb", b"FINISHED", 3000)
    _write(out / "finish" / FIN / "input" / "source.glb", b"MODEL ONE", 3000)
    _write(out / "finish" / FIN / "input" / "source.png", b"png", 3000)
    _write(out / "finish" / FIN / "input" / "settings.json",
           json.dumps({"faces": 15000, "texture_size": 1024, "skip_photo": True}), 3000)
    _write(out / "animate" / RIG / "a-knight_15k_rigged.glb", b"RIGGED", 4000)
    _write(out / "animate" / RIG / "a-knight_15k_rigged.provenance.json",
           json.dumps({"input": {"sha256": hashlib.sha256(b"FINISHED").hexdigest()}}), 4000)
    return out


def _knight(output: Path) -> dict:
    (asset,) = [a for a in assets_api.list_assets(output) if a["picture"]]
    return asset


def test_the_library_says_what_each_step_was_made_with(output):
    asset = _knight(output)
    assert asset["made"]["model"] == {"seed": 77, "engine": "pixal3d"}
    assert asset["made"]["finished"] == {"faces": 15000, "texture_size": 1024, "photo": False}
    assert asset["stale"] == {"finished": False, "rigged": False}


def test_a_later_step_made_from_an_earlier_version_is_marked(output):
    # a new 3D model of the same picture: the finish and the rig came from the old one
    _write(output / "conditional" / GEN2 / f"{GEN2}.glb", b"MODEL TWO", 5000)
    _write(output / "conditional" / GEN2 / f"{GEN2}.json",
           json.dumps({"backend": "pixal3d", "parameters": {"seed": 78}}), 5000)
    asset = _knight(output)
    assert asset["model"].endswith(f"{GEN2}.glb") and asset["made"]["model"]["seed"] == 78
    assert asset["stale"] == {"finished": True, "rigged": False}
    # a new finish of the new model: now the rig is the one left behind
    fin2 = f"{IMG}-cutout__pixal3d__202610__finish__20261006-091000"
    _write(output / "finish" / fin2 / "a-knight_40k.glb", b"FINISHED TWO", 6000)
    _write(output / "finish" / fin2 / "input" / "source.glb", b"MODEL TWO", 6000)
    asset = _knight(output)
    assert asset["stale"] == {"finished": False, "rigged": True}


def test_create_can_send_its_triangle_count_to_the_finish(tmp_path):
    from viewer.generate_api import parse_multipart

    body, ctype = activity_api.multipart({"title": "x", "steps": json.dumps(["finished"]),
                                          "finish_settings": json.dumps({"faces": 8000, "evil": 1})},
                                         {"file": ("mine.glb", b"glb")})
    chain = activity_api.chain_from_form(parse_multipart(ctype, body))
    assert chain.finish_settings == {"faces": 8000}
    from tests.test_viewer_activity_api import _runner

    runner, jobs = _runner(tmp_path)
    _write(tmp_path / "m" / "model.glb", b"glb")
    _write(tmp_path / "p" / "picture.png", b"png")
    chain.made = {"model": "m/model.glb", "picture": "p/picture.png"}
    runner.submit(chain, background=False)
    sent = next(b for m, p, b in jobs.calls if p == "/api/finish")
    assert b'{"faces": 8000}' in sent


def _run(module: str, body: str):
    url = (REPO / "viewer" / "studio" / module).as_uri()
    result = subprocess.run([NODE, "--input-type=module", "--eval", f"import * as m from {json.dumps(url)};\n{body}"],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


KNIGHT = {"name": "knight", "kind": "character", "stage": "rigged", "picture": "p.png", "model": "m.glb",
          "finished": "f_15k.glb", "rigged": "r.glb", "clips": [], "props": [],
          "made": {"model": {"seed": 77, "engine": "pixal3d"}, "finished": {"faces": 15000, "texture_size": 1024,
                                                                            "photo": False}},
          "stale": {"finished": True, "rigged": False}}


@needs_node
def test_create_keeps_a_3d_seed_when_asked_and_sends_the_triangles():
    out = _run("create.js", """
      const fresh = m.modelSeedFor({ newEachTime: true, seed: 5 }, () => 0.5);
      const kept = m.modelSeedFor({ newEachTime: false, seed: 5 }, () => 0.5);
      const form = m.planForm({ steps: ['model', 'finished'], modelSeed: 5, finishSettings: { faces: 8000 } });
      console.log(JSON.stringify({ fresh, kept, finish: form.get('finish_settings'), seed: form.get('model_seed') }));""")
    assert out["kept"] == 5 and out["fresh"] != 5
    assert json.loads(out["finish"]) == {"faces": 8000} and out["seed"] == "5"


@needs_node
def test_redo_offers_the_values_the_step_was_made_with():
    out = _run("steps.js", f"""
      const asset = {json.dumps(KNIGHT)};
      const ctx = {{ engines: [{{ id: 'pixal3d', label: 'Pixal3D', license: 'x' }}], presets: [] }};
      console.log(JSON.stringify({{
        model: m.stepBody('model', asset, ctx, {{ redo: true }}),
        first: m.stepBody('model', {{ ...asset, made: {{}} }}, ctx),
        finish: m.stepBody('finished', asset, ctx, {{ redo: true }}),
      }}));""")
    assert 'data-k="seed"' in out["model"] and 'value="77"' in out["model"] and "Make it again" in out["model"]
    assert 'data-k="newseed"' in out["first"] and "Make 3D model" in out["first"]
    assert '<option value="15000" selected>' in out["finish"] and "Finish again" in out["finish"]
    assert 'data-k="photo" >' in out["finish"] or 'data-k="photo">' in out["finish"]   # unticked, as made
    assert 'value="1024" selected' in out["finish"]


@needs_node
def test_the_steps_say_which_were_made_from_an_earlier_version():
    out = _run("library.js", f"""
      const asset = {json.dumps(KNIGHT)};
      console.log(JSON.stringify({{
        finished: m.staleNote(asset, 'finished'), rigged: m.staleNote(asset, 'rigged'),
        redo: ['picture', 'model', 'finished', 'rigged', 'animated'].map((s) => m.canRedo(asset, s)),
        made: m.madeWith(asset, 'model'),
      }}));""")
    assert out["finished"] == "made from an earlier 3D model" and out["rigged"] is None
    assert out["redo"] == [False, True, True, True, False]
    assert out["made"] == "seed 77"


@needs_node
def test_a_triangle_count_made_outside_the_list_is_offered_as_made():
    out = _run("steps.js", f"""
      const asset = {{ ...{json.dumps(KNIGHT)}, made: {{ finished: {{ faces: 18000 }} }} }};
      console.log(JSON.stringify(m.stepBody('finished', asset, {{ engines: [], presets: [] }}, {{ redo: true }})));""")
    assert '<option value="18000" selected>18,000 triangles · as made</option>' in out
    assert out.index('value="15000"') < out.index('value="18000"') < out.index('value="40000"')


@needs_node
def test_redoing_the_3d_model_tries_a_new_seed_unless_one_is_chosen():
    """Redo with nothing changed would make the same model again: a new seed is the default,
    the one it was made with shown and kept in the box for whoever wants it back."""
    out = _run("steps.js", f"""
      const asset = {json.dumps(KNIGHT)};
      const ctx = {{ engines: [{{ id: 'pixal3d', label: 'Pixal3D', license: 'x' }}], presets: [] }};
      console.log(JSON.stringify(m.stepBody('model', asset, ctx, {{ redo: true }})));""")
    assert 'data-k="newseed" checked' in out and "it was 77" in out and 'data-seedbox hidden' in out


def test_a_seed_kept_in_a_provenance_shaped_record_is_read(tmp_path):
    out = tmp_path / "output"
    run = "a-crate__20261005-205524__sf3d__20261005-210000"
    _write(out / "images" / "a-crate__20261005-205524" / "a-crate__20261005-205524.png", b"png", 1000)
    _write(out / "conditional" / run / f"{run}.glb", b"CRATE", 2000)
    _write(out / "conditional" / run / f"{run}.json",
           json.dumps({"model": {"backend": "sf3d", "parameters": {"seed": 9}}}), 2000)
    (asset,) = [a for a in assets_api.list_assets(out) if a["model"]]
    assert asset["made"]["model"] == {"seed": 9, "engine": "sf3d"}


@needs_node
def test_one_prop_chooses_its_triangles_in_create_too():
    out = _run("create.js", """
      const form = m.planForm({ steps: ['picture', 'model', 'finished'], want: 'prop', finishSettings: { faces: 5000 } });
      console.log(JSON.stringify(form.get('finish_settings')));""")
    assert json.loads(out) == {"faces": 5000}
