"""The style race: every judge scores the same labelled concepts against one bible."""

import json

import numpy as np
import pytest
import style_race
from PIL import Image

from image_to_3dlab.style import bible as sb
from image_to_3dlab.style import embed, race


class ByRed(embed.Embedder):
    """A stand-in embedding where 'style' is how red a picture is."""

    key = "red"
    record = {"id": "red"}

    def embed(self, images):
        rows = []
        for image in images:
            r, g, b = np.asarray(image, dtype=float).mean(axis=(0, 1)) / 255
            rows.append([r, 1 - r, 0.1])
        rows = np.asarray(rows)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def picture(path, colour):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path)
    return path


@pytest.fixture
def setup(tmp_path):
    root = sb.create(tmp_path / "bible", "b")
    for i in range(4):
        sb.add(root, picture(tmp_path / "src" / f"ok{i}.png", (230 - i * 5, 20, 20)), "red is right", kind="prop")
    concepts = []
    for i in range(3):
        concepts.append({"file": f"c/good{i}.png", "label": "approved", "kind": "prop"})
        picture(tmp_path / "c" / f"good{i}.png", (220 - i * 10, 30, 30))
        concepts.append({"file": f"c/bad{i}.png", "label": "sent_back", "kind": "prop"})
        picture(tmp_path / "c" / f"bad{i}.png", (40 + i * 10, 200, 200))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"concepts": concepts}), encoding="utf-8")
    return root, labels


def test_labels_are_read_relative_to_their_file_and_checked(setup, tmp_path):
    root, labels = setup
    rows = race.load_labels(labels)
    assert len(rows) == 6 and rows[0]["path"] == (tmp_path / "c" / "good0.png").resolve()
    assert [r["approved"] for r in rows[:2]] == [True, False]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"concepts": [{"file": "c/good0.png", "label": "maybe"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        race.load_labels(bad)


def test_a_concept_that_is_in_the_bible_is_left_out(setup, tmp_path):
    root, labels = setup
    rows = race.load_labels(labels)
    inside = sb.load(root)["images"][0]
    rows.append({"path": root / inside["file"], "approved": True, "kind": "prop"})
    kept, dropped = race.without_bible_pictures(root, rows)
    assert len(kept) == 6 and len(dropped) == 1


def test_embedding_scores_separate_the_stand_in_style(setup):
    root, labels = setup
    rows = race.load_labels(labels)
    scores = race.embedding_scores(ByRed(), root, rows, k=2)
    for reading in ("knn", "csls"):
        report = race.report(scores[reading], rows)
        assert report["auc"] == 1.0


def test_judge_scores_use_two_reference_sets_when_the_bible_has_enough(setup, tmp_path):
    root, labels = setup
    rows = race.load_labels(labels)
    seen = []

    class Fake:
        def judge(self, reference_sets, concept, criteria):
            seen.append([len(s) for s in reference_sets])
            return {"score": 0.9 if "good" in concept.name else 0.1, "answers": []}

    scores = race.judge_scores(Fake(), root, rows)
    assert race.report(scores, rows)["auc"] == 1.0
    assert seen[0] == [3, 1]  # 4 approved pictures: three, then the one left


def test_the_script_writes_a_report_for_the_chosen_judges(setup, tmp_path, monkeypatch, capsys):
    root, labels = setup
    monkeypatch.setattr(style_race.embed, "load_embedder", lambda key, device=None: ByRed())
    out = tmp_path / "report.json"
    assert style_race.main([str(root), str(labels), "--judges", "csd", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert {"csd.knn", "csd.csls"} <= set(report["results"])
    assert report["results"]["csd.knn"]["auc"] == 1.0
    assert "csd.knn" in capsys.readouterr().out
