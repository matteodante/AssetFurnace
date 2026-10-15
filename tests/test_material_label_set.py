"""Queueing a label set, following its progress and exporting what is complete."""

from __future__ import annotations

import json

import material_label_set as script
from _check_fixtures import clean

from image_to_3dlab import material_labels as ml
from image_to_3dlab.checks import review


def test_queue_progress_and_export(tmp_path, capsys):
    output = tmp_path / "output"
    (output / "set").mkdir(parents=True)
    first, second = clean(output / "set" / "orc.glb"), clean(output / "set" / "barrel.glb")
    assert script.main(["queue", str(first), "--kind", "creature", "--output", str(output)]) == 0
    assert script.main(["queue", str(second), "--kind", "prop", "--output", str(output)]) == 0
    items = review.fold(review.load(output / "review-log.jsonl"))
    assert {i["source"] for i in items.values()} == {"material-labels"}
    assert {i["kind"] for i in items.values()} == {"creature", "prop"}

    orc = items["orc"]
    ml.record(output / "material-labels.jsonl", "orc", orc["asset"], orc["asset_sha256"], baked_light=False,
              materials_look="right", points=[{"uv": [0.1, 0.2], "material": 0, "label": "skin"}])
    capsys.readouterr()
    assert script.main(["progress", "--output", str(output)]) == 0
    assert "1 of 2 complete" in capsys.readouterr().out

    target = tmp_path / "labels.json"
    assert script.main(["export", str(target), "--output", str(output)]) == 0
    rows = json.loads(target.read_text())
    assert [r["key"] for r in rows] == ["orc"] and rows[0]["kind"] == "creature"


def test_a_file_outside_output_is_refused(tmp_path, capsys):
    output = tmp_path / "output"
    output.mkdir()
    stray = clean(tmp_path / "stray.glb")
    assert script.main(["queue", str(stray), "--kind", "prop", "--output", str(output)]) == 1
    assert "under" in capsys.readouterr().out
