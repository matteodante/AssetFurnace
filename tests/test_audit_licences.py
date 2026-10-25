"""Licences handed over by the audit (17/10/2026): a research-only class is held to the rule
`provenance.validate_run_policy` applies to a research-only backend (a private, non-commercial
showcase only), and a set's record takes its class from the models that made it, not a fixed one."""

from __future__ import annotations

import pytest

from image_to_3dlab import batch
from image_to_3dlab import unreal_profile as up
from image_to_3dlab.provenance import LICENSES
from image_to_3dlab.slot_parts import sidecar

POLICY = {"allow_conditional": True}


@pytest.mark.parametrize("use_case, distribution", [("game", "public"), ("game", "worldwide"),
                                                    ("showcase", "public")])
def test_research_only_is_refused_outside_a_private_showcase(use_case, distribution):
    intent = {"use_case": use_case, "distribution": distribution, "commercial_intent": False}
    assert "non-commercial" in batch.class_problem("research-only", intent, POLICY)


def test_research_only_is_allowed_only_for_a_private_non_commercial_showcase():
    intent = {"use_case": "showcase", "distribution": "private", "commercial_intent": False}
    assert batch.class_problem("research-only", intent, POLICY) is None
    assert "non-commercial" in batch.class_problem("research-only", {**intent, "commercial_intent": True}, POLICY)


def test_a_sets_class_is_the_strictest_of_its_models():
    wanted = up.strictest_class([LICENSES["qwen-image-2.1"].classification, LICENSES["pixal3d"].classification])
    record = sidecar.parts_record(mode="standalone" if "standalone" in sidecar.MODES else "dressed_split",
                                  set_name="s", body={"name": "b", "image": "x", "rigged": "y"}, body_json="b.json",
                                  concept=None if "standalone" in sidecar.MODES else "c.png", concept_info={},
                                  chosen={}, pieces={}, fit=None, checks={}, matte=None)
    assert record["classification"] == wanted == sidecar.classification()
    assert wanted != "commercial-conditional" or LICENSES["qwen-image-2.1"].classification == "commercial-conditional"
