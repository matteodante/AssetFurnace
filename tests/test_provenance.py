import pytest

from image_to_3dlab.provenance import LICENSES, validate_run_policy


def test_sf3d_is_allowed_when_conditionals_are_enabled():
    profile = validate_run_policy("sf3d", "game", "worldwide", True)
    assert profile.classification == "commercial-conditional"


def test_hunyuan_is_blocked_for_worldwide_game():
    with pytest.raises(ValueError, match="worldwide"):
        validate_run_policy("hunyuan-comfyui", "game", "worldwide", True)


def test_trellis_inherits_dinov3_conditional_classification():
    profile = validate_run_policy("trellis2", "game", "worldwide", True)
    assert profile.classification == "commercial-conditional"


def test_trellis_is_blocked_when_manifest_disallows_conditionals():
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy("trellis2", "game", "worldwide", False)


def test_qwen_image_is_research_only_as_its_licence_says():
    """Qwen Research License Agreement, Release Date 2026-09-20: Non-Commercial is "for
    research or evaluation purposes only" (1.i) and the grant is "FOR NON-COMMERCIAL
    PURPOSES ONLY" (2.a). The class says what the licence says (audit 2026-10-17)."""
    profile = LICENSES["qwen-image-2.1"]
    assert profile.classification == "research-only"
    assert profile.license_url == "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE"
    conditions = " ".join(profile.conditions)
    assert "research or evaluation" in conditions and "2(b)" in conditions
    assert any("Built with Qwen" in c for c in profile.conditions)


def test_qwen_records_cite_the_licence_not_an_unverified_post():
    from image_to_3dlab import provenance

    text = " ".join(LICENSES["qwen-image-2.1"].conditions) + provenance.QWEN_OUTPUT_RIGHTS
    assert "x.com" not in text and "are yours" not in text
    assert "non-commercial" in provenance.QWEN_OUTPUT_RIGHTS.lower()
    assert not hasattr(provenance, "QWEN_OUTPUT_STATEMENT")


def test_this_projects_non_commercial_use_of_qwen_is_not_refused():
    """AssetFurnace is never commercial: a game or a public showcase made without commercial
    intent keeps working; the record and the licence check say it is non-commercial only."""
    profile = LICENSES["qwen-image-2.1"]
    for use_case, distribution in (("showcase", "private"), ("showcase", "public"), ("game", "worldwide")):
        assert validate_run_policy("qwen-image-2.1", use_case, distribution, allow_conditional=True) is profile


def test_a_commercial_run_with_qwen_is_refused():
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy("qwen-image-2.1", "game", "worldwide", allow_conditional=True,
                            commercial_intent=True)
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy("qwen-image-2.1", "game", "worldwide", allow_conditional=False)


def test_pixal3d_and_trellis_record_the_dinov3_redistribution_terms():
    """DINOv3 License (2025-08-14) 1.b.i: who passes the DINO materials on gives a copy of
    the licence and shows "Built with DINOv3"."""
    for key in ("pixal3d", "trellis2"):
        conditions = " ".join(LICENSES[key].conditions)
        assert "Built with DINOv3" in conditions and "copy of the DINOv3 License" in conditions, key
