"""Licences handed over by the audit (17/10/2026): a research-only class (Qwen, since #94) is
refused only for a run with commercial intent, as `provenance.validate_run_policy` does for a
backend, and otherwise warned about and recorded; a set's record takes its class from the
models that made it, not a fixed one."""

from __future__ import annotations

import json

import pytest

from image_to_3dlab import batch
from image_to_3dlab import unreal_profile as up
from image_to_3dlab.provenance import LICENSES
from image_to_3dlab.slot_parts import sidecar

POLICY = {"allow_conditional": True}


@pytest.mark.parametrize("use_case, distribution", [("game", "public"), ("game", "worldwide"),
                                                    ("showcase", "public"), ("showcase", "private")])
def test_research_only_without_commercial_intent_is_allowed_and_warned(use_case, distribution):
    intent = {"use_case": use_case, "distribution": distribution, "commercial_intent": False}
    assert batch.class_problem("research-only", intent, POLICY) is None
    assert "non-commercial" in batch.class_warning("research-only", intent)


def test_research_only_with_commercial_intent_is_refused():
    intent = {"use_case": "game", "distribution": "worldwide", "commercial_intent": True}
    assert "non-commercial" in batch.class_problem("research-only", intent, POLICY)
    assert batch.class_warning("commercial-clear", intent) is None


def test_a_batch_keeps_the_warning(tmp_path):
    from batch_fakes import tiny_glb

    asset = tiny_glb(tmp_path / "a.glb")
    asset.with_suffix(".provenance.json").write_text(json.dumps({"classification": "research-only"}))
    spec = tmp_path / "lot.json"
    spec.write_text(json.dumps({"schema_version": 1, "name": "lot", "use_case": "game", "distribution": "worldwide",
                                "output": {"directory": str(tmp_path / "out")},
                                "rows": [{"id": "SM_A", "asset": "a.glb", "steps": ["checks"]}]}))
    loaded = batch.load_batch(spec)
    assert any("research-only" in w and "SM_A" in w for w in loaded.warnings)


def test_a_sets_class_is_the_strictest_of_its_models():
    wanted = up.strictest_class([LICENSES["qwen-image-2.1"].classification, LICENSES["pixal3d"].classification])
    record = sidecar.parts_record(mode="standalone" if "standalone" in sidecar.MODES else "dressed_split",
                                  set_name="s", body={"name": "b", "image": "x", "rigged": "y"}, body_json="b.json",
                                  concept=None if "standalone" in sidecar.MODES else "c.png", concept_info={},
                                  chosen={}, pieces={}, fit=None, checks={}, matte=None)
    assert record["classification"] == wanted == sidecar.classification()
    assert wanted != "commercial-conditional" or LICENSES["qwen-image-2.1"].classification == "commercial-conditional"
