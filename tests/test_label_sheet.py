"""The label sheet: a tap on a picture becomes a texel of the asset, and the labels land in the project's record."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab import material_labels as ml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "label_sheet.py"


def sheet_module():
    spec = importlib.util.spec_from_file_location("label_sheet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def uv_pass(size: int = 8) -> np.ndarray:
    """A UV pass whose left half shows material 0 and right half is background."""
    rgba = np.zeros((size, size, 4), np.float32)
    rows, cols = np.mgrid[0:size, 0:size]
    rgba[..., 0] = (cols + 0.5) / size
    rgba[..., 1] = 1.0 - (rows + 0.5) / size              # Blender's v points up
    rgba[..., 2] = np.where(cols < size // 2, 1.0, 0.0)  # material index + 1
    rgba[..., 3] = 1.0
    return rgba


def test_a_tap_becomes_the_texel_and_material_the_pixel_shows():
    sheet = sheet_module()
    assert sheet.tap_to_point(uv_pass(), 0.1, 0.3) == {"uv": [0.0625, 0.3125], "material": 0}
    assert sheet.tap_to_point(uv_pass(), 0.9, 0.5) is None           # background


def make_sheet(tmp_path: Path) -> tuple[Path, Path]:
    asset = tmp_path / "knight.glb"
    asset.write_bytes(b"the knight")
    folder = tmp_path / "sheet"
    (folder / "knight").mkdir(parents=True)
    np.savez(folder / "knight" / "view_00_uv.npz", rgba=uv_pass())
    manifest = {"assets": [{"key": "knight", "path": str(asset), "sha256": hashlib.sha256(b"the knight").hexdigest(),
                            "kind": "humanoid", "views": [{"index": 0, "file": "knight/view_00.jpg",
                                                           "uv": "knight/view_00_uv.npz"}], "lit": []}]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder, asset


def test_the_sheets_labels_land_in_the_projects_record(tmp_path):
    sheet = sheet_module()
    folder, _ = make_sheet(tmp_path)
    labels = [{"id": "knight", "data": {"baked_light": True, "materials_look": "wrong", "note": " shiny ",
                                        "points": [{"v": 0, "x": 0.1, "y": 0.3, "label": "metal"},
                                                   {"v": 0, "x": 0.9, "y": 0.3, "label": "cloth"}]}}]
    log = tmp_path / "material-labels.jsonl"
    report = sheet.import_labels(folder, labels, log)
    assert report == {"imported": 1, "points": 1, "dropped_points": 1, "unknown": [], "changed": []}
    row = ml.dataset(ml.load(log))[0]
    assert row["key"] == "knight" and row["baked_light"] is True and row["materials_look"] == "wrong"
    assert row["points"] == [{"uv": [0.0625, 0.3125], "material": 0, "label": "metal"}]
    assert row["asset_sha256"] == hashlib.sha256(b"the knight").hexdigest() and row["note"] == "shiny"


def test_labels_for_an_asset_that_changed_since_the_sheet_was_built_are_refused(tmp_path):
    sheet = sheet_module()
    folder, asset = make_sheet(tmp_path)
    asset.write_bytes(b"regenerated")
    report = sheet.import_labels(folder, [{"id": "knight", "data": {"baked_light": False}}],
                                 tmp_path / "material-labels.jsonl")
    assert report["imported"] == 0 and report["changed"] == ["knight"]


def test_an_unknown_label_word_is_refused_rather_than_guessed(tmp_path):
    sheet = sheet_module()
    folder, _ = make_sheet(tmp_path)
    with pytest.raises(ValueError, match="label"):
        sheet.import_labels(folder, [{"id": "knight", "data": {"points": [{"v": 0, "x": 0.1, "y": 0.3,
                                                                          "label": "unobtainium"}]}}],
                            tmp_path / "material-labels.jsonl")


def test_the_sheet_offers_exactly_the_words_the_record_accepts():
    sheet = sheet_module()
    assert sheet.WORDS == list(ml.MATERIALS)
