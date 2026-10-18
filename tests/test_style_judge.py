"""The vision-language judge: yes/no questions read as probabilities. No model, no server."""

import base64
import io
import math

import pytest
from PIL import Image

from image_to_3dlab.style import judge


def logprobs(**tokens):
    return [{"token": t, "logprob": math.log(p)} for t, p in tokens.items()]


def test_the_yes_probability_adds_every_spelling_and_ignores_other_tokens():
    top = logprobs(Yes=0.5, yes=0.1, No=0.2, **{" no": 0.1, "Maybe": 0.1})
    assert judge.yes_probability(top) == pytest.approx(0.6 / 0.9)
    assert judge.yes_probability(logprobs(Maybe=1.0)) == pytest.approx(0.5)  # no opinion either way


def test_the_request_sends_references_then_the_concept_and_asks_for_one_token(tmp_path):
    paths = []
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (900, 600), colour).save(path)
        paths.append(path)
    body = judge.build_request(paths[:2], paths[2], "Are the colours as bright?")
    content = body["messages"][0]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 3
    last = images[-1]["image_url"]["url"]
    decoded = Image.open(io.BytesIO(base64.b64decode(last.split(",", 1)[1])))
    assert max(decoded.size) == judge.SIDE and decoded.getpixel((0, 0))[2] > 200  # the concept, last
    text = " ".join(c["text"] for c in content if c["type"] == "text")
    assert "Are the colours as bright?" in text and "yes or no" in text.lower()
    assert body["max_tokens"] == 1 and body["logprobs"] is True and body["temperature"] == 0


def test_the_server_command_names_the_model_the_vision_file_and_the_port(tmp_path):
    command = judge.server_command(tmp_path / "llama-server.exe", tmp_path / "m.gguf", tmp_path / "v.gguf", 8811)
    assert command[0].endswith("llama-server.exe")
    for flag, value in (("-m", "m.gguf"), ("--mmproj", "v.gguf"), ("--port", "8811")):
        assert command[command.index(flag) + 1].endswith(value)
    assert "--host" in command and command[command.index("--host") + 1] == "127.0.0.1"


def test_judging_averages_questions_and_reference_sets(tmp_path):
    asked = []

    def fake(body):
        question = body["messages"][0]["content"][-1]["text"]
        asked.append(question)
        p = 0.9 if "colours" in question else 0.3
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": logprobs(Yes=p, No=1 - p)}]}}]}

    concept = tmp_path / "c.png"
    Image.new("RGB", (8, 8)).save(concept)
    found = judge.Judge(post=fake).judge([[concept], [concept]], concept,
                                         ["Are the colours as bright?", "Is it painted the same way?"])
    assert found["score"] == pytest.approx(0.6)
    assert found["answers"] == [{"question": "Are the colours as bright?", "yes": pytest.approx(0.9)},
                                {"question": "Is it painted the same way?", "yes": pytest.approx(0.3)}]
    assert len(asked) == 4  # two questions, two reference sets


def test_missing_parts_say_how_to_get_them(tmp_path, monkeypatch):
    monkeypatch.setattr(judge.models, "llama_server", lambda repo=None: None)
    with pytest.raises(judge.models.MissingModel, match="bootstrap_style.py --only llama"):
        judge.find_parts(hub=tmp_path)


def test_the_servers_log_file_is_closed_when_it_stops(tmp_path):
    """Code read 18/10/2026: the log stayed open until garbage collection, locked on Windows."""
    from image_to_3dlab.style import judge

    server = judge.LlamaServer.__new__(judge.LlamaServer)
    server.process = None
    handle = server._out = open(tmp_path / "judge.log", "wb")  # noqa: SIM115 - the test closes it through _stop
    server._stop()
    assert handle.closed and server._out is None
