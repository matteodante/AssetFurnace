"""From a reference to a concept in the viewer: inputs, the edit job, its record, and `generate`."""

from __future__ import annotations

import functools
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import concept_api  # noqa: E402
import generate_api  # noqa: E402
import image_api  # noqa: E402

from image_to_3dlab.concept import prompts, sidecar  # noqa: E402


def picture(path, colour=(200, 100, 50), size=32, rgba=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgba:
        pixels = np.zeros((size, size, 4), np.uint8)
        pixels[8:24, 8:24] = (*colour, 255)
        Image.fromarray(pixels).save(path)
    else:
        Image.new("RGB", (size, size), colour).save(path)
    return path


@pytest.fixture
def output(tmp_path):
    return tmp_path / "output"


@pytest.fixture
def ready(tmp_path, monkeypatch):
    """Weights and vision file present; sd-cli replaced by a function that writes a picture."""
    vision = tmp_path / "hub" / "mmproj.gguf"
    vision.parent.mkdir(parents=True)
    vision.write_bytes(b"v")
    monkeypatch.setattr(concept_api.edit, "find_vision", lambda hub: vision)
    monkeypatch.setattr(image_api, "resolve_weights", lambda: {"diffusion_model": Path("/w/turbo-Q8_0.gguf"),
                                                               "llm": Path("/w/enc.gguf"), "vae": Path("/w/vae.st")})
    ran = []

    def fake_run(job, manager, weights=None):
        ran.append(job)
        picture(job.output_path, (10, 20, 30))
        job.set_status("done")
        manager.finish(job)

    monkeypatch.setattr(image_api, "run_job", fake_run)
    monkeypatch.setattr(concept_api, "result_mask", lambda path: np.asarray(Image.open(path))[..., 0] >= 0)
    monkeypatch.setattr(image_api, "MANAGER", image_api.ImageJobManager(output_root=tmp_path / "output" / "images"))
    return ran


def fake_render(glb, folder, views=("front",), size=1024, **_):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for view in views:
        picture(folder / f"{view}_colour.png", (150, 150, 150), rgba=True)
        picture(folder / f"{view}_depth.png", (180, 180, 180), rgba=True)
    bones = {"neck_01": ((16, 9), (16, 7)), "head": ((16, 7), (16, 2))}
    for side, sign in (("l", 1), ("r", -1)):
        for i, name in enumerate(("upperarm", "lowerarm", "hand")):
            bones[f"{name}_{side}"] = ((16 + sign * (3 + 4 * i), 10), (16 + sign * (7 + 4 * i), 10))
        for i, name in enumerate(("thigh", "calf", "foot")):
            bones[f"{name}_{side}"] = ((16 + sign * 2, 16 + 5 * i), (16 + sign * 2, 21 + 5 * i))
    placed = {name: {"head": list(h), "tail": list(t)} for name, (h, t) in bones.items()}
    (folder / "bones.json").write_text(json.dumps({v: placed for v in views}))
    return {"bones": folder / "bones.json"}


def test_info_names_the_ways_and_what_is_ready(output, ready):
    data = concept_api.info(output)
    assert data["modes"] == list(prompts.MODES) and data["structures"] == ["depth", "lineart"]
    assert data["views"] == ["side", "back"] and data["max_references"] == 5
    assert data["ready"]["vision"] is True
    assert data["attempts"] == {"default": 1, "most": 4}


def test_an_upload_is_kept_as_a_png_named_by_its_content(output):
    source = picture(output.parent / "in" / "wolf.webp")
    saved = concept_api.save_upload(output, source.read_bytes(), "wolf.webp")
    assert saved["path"].startswith("concept/uploads/") and saved["path"].endswith(".png")
    assert Image.open(output / saved["path"]).size == (32, 32)
    assert concept_api.save_upload(output, source.read_bytes(), "again.webp") == saved
    with pytest.raises(ValueError, match="picture"):
        concept_api.save_upload(output, b"not a picture", "x.png")
    with pytest.raises(ValueError, match="PNG"):
        concept_api.save_upload(output, source.read_bytes(), "wolf.gif")


def test_maps_of_a_model_are_rendered_once_and_kept(output, monkeypatch):
    glb = output / "wolf" / "wolf.glb"
    glb.parent.mkdir(parents=True)
    glb.write_bytes(b"glb")
    calls = []
    monkeypatch.setattr(concept_api.maps, "render", lambda *a, **k: calls.append(a) or fake_render(*a, **k))
    depth = concept_api.prepare(output, "wolf/wolf.glb", "depth", "side")
    assert depth["path"].endswith("side_depth_black.png") and depth["mask"] == depth["path"]
    colour = concept_api.prepare(output, "wolf/wolf.glb", "colour", "front")
    assert colour["path"].endswith("front_colour_grey.png") and colour["mask"].endswith("front_colour.png")
    pose = concept_api.prepare(output, "wolf/wolf.glb", "pose", "front")
    assert pose["path"].endswith("front_pose.png") and pose["mask"] is None
    assert len(calls) == 1  # every view was rendered on the first request
    assert all((output / p["path"]).is_file() for p in (depth, colour, pose))


def test_a_picture_gives_its_lineart_or_itself(output):
    picture(output / "images" / "robot.png")
    lines = concept_api.prepare(output, "images/robot.png", "lineart")
    assert lines["path"].endswith("lineart.png") and (output / lines["path"]).is_file()
    assert concept_api.prepare(output, "images/robot.png", "picture")["path"] == "images/robot.png"
    with pytest.raises(ValueError, match="depth"):
        concept_api.prepare(output, "images/robot.png", "depth")


def test_nothing_outside_the_output_folder_is_read(output, tmp_path):
    picture(tmp_path / "secret.png")
    output.mkdir()
    for path in ("../secret.png", str(tmp_path / "secret.png")):
        with pytest.raises(ValueError, match="output"):
            concept_api.prepare(output, path, "picture")


def test_a_redraw_request_becomes_the_prompt_the_pictures_and_what_the_iou_is_measured_on(output):
    depth = picture(output / "concept" / "maps" / "a" / "front_depth_black.png")
    style = picture(output / "images" / "style.png")
    plan = concept_api.build(output, {"mode": "redraw", "subject": "werewolf", "colours": "grey fur",
                                      "structure": "depth", "shape": [str(depth.relative_to(output))],
                                      "style_refs": ["images/style.png"], "style": "hand-painted"})
    assert plan["prompt"].startswith(prompts.redraw("werewolf", "grey fur", style="hand-painted"))
    assert plan["prompt"].endswith(prompts.style_sentence(2, 1))
    assert plan["references"] == [(depth.resolve(), "shape"), (style.resolve(), "style")]
    assert plan["iou_against"] == depth.resolve() and plan["fit"] is True


def test_completing_a_cut_out_piece_uses_its_own_words_and_measures_nothing(output):
    picture(output / "slots" / "pauldron_cut.png")
    plan = concept_api.build(output, {"mode": "complete_piece", "piece": "a single shoulder pauldron",
                                      "shape": ["slots/pauldron_cut.png"]})
    assert plan["prompt"] == prompts.complete_piece("a single shoulder pauldron")
    assert plan["iou_against"] is None and plan["name"] == "concept complete piece a single shoulder pauldron"


def test_a_request_with_the_wrong_number_of_shape_pictures_is_refused(output):
    front = picture(output / "images" / "front.png")
    with pytest.raises(ValueError, match="2 shape pictures"):
        concept_api.build(output, {"mode": "other_view", "view": "back", "depth": True, "shape": ["images/front.png"]})
    with pytest.raises(ValueError, match="subject"):
        concept_api.build(output, {"mode": "redraw", "colours": "grey", "shape": ["images/front.png"]})
    assert front.is_file()


def test_a_job_runs_the_edit_and_leaves_its_record_with_the_iou(output, ready):
    body = picture(output / "concept" / "maps" / "b" / "front_colour.png", rgba=True)
    grey = picture(output / "concept" / "maps" / "b" / "front_colour_grey.png")
    job = concept_api.start(output, {"mode": "add_piece", "piece": "a steel helmet",
                                     "shape": [str(grey.relative_to(output))], "seed": 7})
    concept_api.wait(job)
    assert job.status == "done" and job.settings["seed"] == 7
    assert job.extra_args[:2] == ["--llm_vision", str(Path(job.extra_args[1]))] and "-r" in job.extra_args
    record = sidecar.read(job.output_path)
    assert record["mode"] == "add_piece" and record["references"][0]["role"] == "shape"
    assert record["models"]["diffusion_model"] == "turbo-Q8_0.gguf" and record["models"]["llm_vision"] == "mmproj.gguf"
    assert 0 < record["iou"] < 1  # the body's alpha against the whole fake picture
    assert job.output_path.parent.name.startswith("concept-add-piece-a-steel-helmet")
    assert body.is_file()


def test_generate_is_one_call_for_other_steps(tmp_path, ready):
    ref = picture(tmp_path / "bible" / "style.png")
    out = concept_api.generate("a troll in this style", [ref], seed=3)
    assert out.is_file() and ready[-1].settings["seed"] == 3
    assert ready[-1].extra_args[-2:] == ["-r", str(ref.resolve())]
    provenance = ready[-1].extra_provenance["concept"]
    assert provenance["references"][0]["name"] == "style.png" and len(provenance["references"][0]["sha256"]) == 64


def test_without_the_vision_file_the_setup_page_is_named(output, ready, monkeypatch):
    monkeypatch.setattr(concept_api.edit, "find_vision", lambda hub: None)
    grey = picture(output / "images" / "body.png")
    with pytest.raises(image_api.MissingWeights, match="mmproj"):
        concept_api.start(output, {"mode": "add_piece", "piece": "a helmet", "shape": ["images/body.png"]})
    assert grey.is_file()


@pytest.fixture
def server(output, monkeypatch):
    output.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", output)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(output.parent)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def request(httpd, method, path, data=None, headers=None):
    sent = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": concept_api.HEADER}
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        sent["Content-Type"] = "application/json"
    sent.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=data, headers=sent)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        conn.close()


def test_the_studio_reads_the_ways_and_starts_a_job(server, output, ready):
    status, data = request(server, "GET", "/api/concept")
    assert status == 200 and data["modes"] == list(prompts.MODES)
    picture(output / "images" / "body.png")
    status, data = request(server, "POST", "/api/concept/jobs",
                           {"mode": "dress", "outfit": "a ranger", "guide": "body", "shape": ["images/body.png"]})
    assert status == 202, data
    assert data["status_url"] == f"/api/image/{data['job_id']}/status"


def test_a_post_without_the_studio_header_is_refused(server, output, ready):
    status, data = request(server, "POST", "/api/concept/jobs", {"mode": "dress"},
                           headers={"X-AssetFurnace-Request": "other"})
    assert status == 403


def test_the_users_judgement_is_kept_in_the_record_and_in_a_log(output, ready):
    grey = picture(output / "images" / "body.png")
    job = concept_api.start(output, {"mode": "add_piece", "piece": "a helmet", "shape": ["images/body.png"], "seed": 1})
    concept_api.wait(job)
    picture_path = job.picture_path()
    saved = concept_api.judge(output, {"picture": picture_path, "verdict": "keep", "note": "the helmet fits"})
    assert saved["judgement"]["verdict"] == "keep" and saved["judgement"]["note"] == "the helmet fits"
    assert sidecar.read(job.output_path)["judgement"]["verdict"] == "keep"
    log = [json.loads(line) for line in (output / "concept" / "judgements.jsonl").read_text().splitlines()]
    assert log[-1]["picture"] == picture_path and log[-1]["mode"] == "add_piece"
    with pytest.raises(ValueError, match="verdict"):
        concept_api.judge(output, {"picture": picture_path, "verdict": "maybe"})
    with pytest.raises(ValueError, match="record"):
        concept_api.judge(output, {"picture": "images/body.png", "verdict": "keep"})
    assert grey.is_file()
