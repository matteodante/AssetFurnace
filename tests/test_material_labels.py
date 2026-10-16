"""The user's material labels: an append-only record, bound to the asset version labelled."""

from __future__ import annotations

import json

import pytest

from image_to_3dlab import material_labels as ml

SHA = "a" * 64
OTHER = "b" * 64


def record(log, **fields):
    defaults = {"key": "mob_01", "asset": "output/x/mob.glb", "asset_sha256": SHA, "now": "2026-10-16T10:00:00+00:00"}
    return ml.record(log, **(defaults | fields))


def test_the_vocabulary_names_materials_not_colours():
    assert {"cloth", "leather", "metal", "skin", "fur", "wood"} <= set(ml.MATERIALS)
    assert len(set(ml.MATERIALS)) == len(ml.MATERIALS)


def test_judgements_add_up_and_the_latest_wins(tmp_path):
    log = tmp_path / "labels.jsonl"
    record(log, baked_light=True)
    record(log, materials_look="wrong", now="2026-10-16T10:01:00+00:00")
    state = record(log, baked_light=False, points=[{"uv": [0.2, 0.3], "material": 0, "label": "leather"}],
                   now="2026-10-16T10:02:00+00:00")
    assert state["baked_light"] is False
    assert state["materials_look"] == "wrong"
    assert state["points"] == [{"uv": [0.2, 0.3], "material": 0, "label": "leather"}]
    assert ml.is_complete(state)
    assert len(log.read_text().splitlines()) == 3            # never rewritten


def test_a_new_version_starts_with_no_labels(tmp_path):
    log = tmp_path / "labels.jsonl"
    record(log, baked_light=True)
    record(log, asset_sha256=OTHER, materials_look="right")
    state = ml.fold(ml.load(log))["mob_01"]
    assert state["asset_sha256"] == OTHER
    assert state["baked_light"] is None and state["materials_look"] == "right"


@pytest.mark.parametrize("bad", [
    {"baked_light": "yes"},
    {"materials_look": "maybe"},
    {"points": [{"uv": [1.5, 0.2], "material": 0, "label": "leather"}]},
    {"points": [{"uv": [0.5, 0.2], "material": 0, "label": "purple"}]},
    {"points": [{"uv": [0.5], "material": 0, "label": "metal"}]},
    {"key": "../escape"},
    {"asset_sha256": "short"},
])
def test_labels_that_do_not_fit_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        record(tmp_path / "labels.jsonl", **bad)
    assert not (tmp_path / "labels.jsonl").exists()


def test_a_damaged_line_is_skipped(tmp_path):
    log = tmp_path / "labels.jsonl"
    record(log, baked_light=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert ml.fold(ml.load(log))["mob_01"]["baked_light"] is True


def test_the_dataset_keeps_only_complete_labels_of_the_current_version(tmp_path):
    log = tmp_path / "labels.jsonl"
    record(log, baked_light=True, materials_look="right",
           points=[{"uv": [0.1, 0.1], "material": 0, "label": "metal"}])
    record(log, key="mob_02", baked_light=False)              # incomplete
    rows = ml.dataset(ml.load(log))
    assert [r["key"] for r in rows] == ["mob_01"]
    assert json.dumps(rows)


def test_progress_counts_a_set_of_keys(tmp_path):
    log = tmp_path / "labels.jsonl"
    record(log, baked_light=True, materials_look="right",
           points=[{"uv": [0.1, 0.1], "material": 0, "label": "metal"}])
    record(log, key="mob_02", baked_light=False)
    progress = ml.progress(ml.load(log), {"mob_01": SHA, "mob_02": SHA, "mob_03": SHA})
    assert progress == {"total": 3, "complete": 1, "started": 1, "missing": 1}


def test_progress_counts_labels_of_the_queued_version_only(tmp_path):
    """A key queued again after its GLB changed is not complete until the new version is labelled."""
    log = tmp_path / "labels.jsonl"
    record(log, baked_light=True, materials_look="right",
           points=[{"uv": [0.1, 0.1], "material": 0, "label": "metal"}])
    queued = {"mob_01": OTHER}
    assert ml.progress(ml.load(log), queued) == {"total": 1, "complete": 0, "started": 0, "missing": 1}
    assert ml.progress(ml.load(log), {"mob_01": SHA})["complete"] == 1


def test_a_version_queued_again_gets_its_old_labels_back(tmp_path):
    log = tmp_path / "labels.jsonl"
    record(log, baked_light=True, materials_look="right")                 # version A
    record(log, asset_sha256=OTHER, baked_light=False)                    # version B
    state = record(log, note="back to A")                                  # version A again
    assert state["asset_sha256"] == SHA
    assert state["baked_light"] is True and state["materials_look"] == "right" and state["note"] == "back to A"
