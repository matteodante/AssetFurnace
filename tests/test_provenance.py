import json

import pytest

from image_to_3dlab.provenance import (
    LICENSES,
    derived_record,
    finalize_output,
    licence_record_beside,
    sha256_file,
    validate_run_policy,
)


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


def test_qwen_image_is_conditional_not_research_only():
    """Qwen confirmed outputs are the user's and the model runs locally, so the only
    condition is the 'Built with Qwen' credit. An agent driving a game run with a
    manifest must not be refused for using it."""
    profile = LICENSES["qwen-image-2.1"]
    assert profile.classification == "commercial-conditional"
    assert profile.folder == "conditional"
    assert validate_run_policy("qwen-image-2.1", "game", "worldwide", allow_conditional=True) is profile
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy("qwen-image-2.1", "game", "worldwide", allow_conditional=False)
    assert any("Built with Qwen" in c for c in profile.conditions)


def test_qwen_outputs_are_yours_and_the_statement_is_cited():
    """Qwen confirmed on 2026-09-21 that outputs are not licensed Materials."""
    conditions = " ".join(LICENSES["qwen-image-2.1"].conditions)
    assert "https://x.com/QwenDevs/status/2101917379785838660" in conditions
    assert "are yours" in conditions
    assert "inherits the restriction" not in conditions


# No shipped model is research-only today (Qwen moved to conditional on 2026-10-07), but the
# gate stays for the next non-commercial model; these tests give it a stand-in profile.
@pytest.fixture
def research_model(monkeypatch):
    import dataclasses

    from image_to_3dlab import provenance
    stand_in = dataclasses.replace(LICENSES["qwen-image-2.1"], classification="research-only",
                                   folder="research_only")
    monkeypatch.setitem(provenance.LICENSES, "research-model", stand_in)
    return "research-model"


def test_research_only_is_refused_even_when_conditional_is_allowed(research_model):
    """allow_conditional is not consent to a non-commercial model. A manifest written for
    the Hunyuan or SF3D cases must not silently pick up a stricter licence."""
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(research_model, use_case="game", distribution="private",
                            allow_conditional=True)


def test_research_only_is_refused_for_public_distribution(research_model):
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(research_model, use_case="showcase", distribution="public",
                            allow_conditional=True)


def test_research_only_is_allowed_for_a_private_showcase(research_model):
    profile = validate_run_policy(research_model, use_case="showcase", distribution="private",
                                  allow_conditional=True)
    assert profile.classification == "research-only"


def test_research_only_still_needs_allow_conditional(research_model):
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy(research_model, use_case="showcase", distribution="private",
                            allow_conditional=False)


def test_qwen_wording_says_the_pictures_are_yours_without_warnings():
    """Qwen confirmed outputs are the user's (2026-09-21); the record says so plainly."""
    from image_to_3dlab.provenance import QWEN_OUTPUT_RIGHTS

    assert "are yours" in QWEN_OUTPUT_RIGHTS
    assert "ambiguous" not in QWEN_OUTPUT_RIGHTS and "non-commercial" not in QWEN_OUTPUT_RIGHTS.lower()


def test_validate_run_policy_accepts_canonical_sf3d_identity():
    profile = validate_run_policy(
        "sf3d", "game", "worldwide", True, model_identity="stabilityai/stable-fast-3d"
    )
    assert profile is LICENSES["sf3d"]


def test_validate_run_policy_rejects_unknown_identity():
    with pytest.raises(ValueError, match="someone/unknown-repo"):
        validate_run_policy(
            "sf3d", "game", "worldwide", True, model_identity="someone/unknown-repo"
        )


def test_validate_run_policy_allows_unknown_identity_when_unverified_allowed():
    profile = validate_run_policy(
        "sf3d", "game", "worldwide", True, model_identity="someone/unknown-repo",
        allow_unverified=True,
    )
    assert profile is LICENSES["sf3d"]


def test_finalize_output_records_canonical_identity(tmp_path):
    image = tmp_path / "input.png"
    generated = tmp_path / "out.glb"
    image.write_bytes(b"image")
    generated.write_bytes(b"glb")
    profile = LICENSES["sf3d"]

    _, sidecar = finalize_output(
        generated=generated,
        image=image,
        output_root=tmp_path,
        backend="sf3d",
        profile=profile,
        intent={"use_case": "showcase", "distribution": "private", "commercial_intent": False},
        parameters={"model": "stabilityai/stable-fast-3d"},
        source_manifest=None,
        model_identity="stabilityai/stable-fast-3d",
    )

    record = json.loads(sidecar.read_text())
    assert record["model"]["identity"] == "stabilityai/stable-fast-3d"
    assert record["output"]["classification"] == profile.classification
    assert "verified" not in record.get("license", {})


def test_finalize_output_marks_unknown_identity_unverified(tmp_path):
    image = tmp_path / "input.png"
    generated = tmp_path / "out.glb"
    image.write_bytes(b"image")
    generated.write_bytes(b"glb")
    profile = LICENSES["sf3d"]

    _, sidecar = finalize_output(
        generated=generated,
        image=image,
        output_root=tmp_path,
        backend="sf3d",
        profile=profile,
        intent={"use_case": "showcase", "distribution": "private", "commercial_intent": False},
        parameters={"model": "someone/unknown-repo"},
        source_manifest=None,
        model_identity="someone/unknown-repo",
    )

    record = json.loads(sidecar.read_text())
    assert record["model"]["identity"] == "someone/unknown-repo"
    assert record["output"]["classification"] == "unverified"
    assert record["license"]["name"] == "Unverified — user-declared model"
    assert record["license"]["verified"] is False
    assert "url" not in record["license"]
    assert record["license"]["conditions"] == [
        "Verify this model's licence yourself; this run used an identity not in the reviewed profile."
    ]


# --- A model made from another one (Finish, Fix a spot) inherits its licence --------------

PIXAL3D = {"backend": "pixal3d",
           "license": {"name": LICENSES["pixal3d"].license_name, "url": LICENSES["pixal3d"].license_url},
           "output": {"sha256": "abc", "classification": "commercial-conditional"}}


def test_a_derived_model_carries_the_generator_s_licence_and_its_record(tmp_path):
    source, made = tmp_path / "wolf.glb", tmp_path / "wolf_20k.glb"
    source.write_bytes(b"raw")
    made.write_bytes(b"finished")
    (tmp_path / "wolf.json").write_text(json.dumps(PIXAL3D))
    record = derived_record("Finish", source, made, licence_record_beside(source))
    assert record["license"] == PIXAL3D["license"]
    assert record["classification"] == record["output"]["classification"] == "commercial-conditional"
    assert record["source_record"] == PIXAL3D
    assert record["input"]["sha256"] == sha256_file(source)
    assert record["output"]["sha256"] == sha256_file(made)


def test_the_sidecar_beats_the_run_record_and_a_record_without_a_licence_is_skipped(tmp_path):
    model = tmp_path / "m.glb"
    model.write_bytes(b"x")
    (tmp_path / "m.provenance.json").write_text(json.dumps({"seed": 1}))
    (tmp_path / "m.json").write_text(json.dumps(PIXAL3D))
    assert licence_record_beside(model) == PIXAL3D
    (tmp_path / "m.provenance.json").write_text(json.dumps({"license": {"name": "X"}}))
    assert licence_record_beside(model)["license"]["name"] == "X"


def test_an_upload_with_no_record_is_said_to_have_none_not_given_one(tmp_path):
    source, made = tmp_path / "upload.glb", tmp_path / "upload_20k.glb"
    source.write_bytes(b"raw")
    made.write_bytes(b"finished")
    (tmp_path / "upload.json").write_text("not json {")
    assert licence_record_beside(source) is None
    record = derived_record("Finish", source, made, None)
    assert record["license"] is None and record["classification"] is None
    assert record["source_record"] is None


@pytest.mark.parametrize("classes, strictest", [
    (("commercial-conditional", "territory-restricted"), "territory-restricted"),
    (("commercial-clear", None), "commercial-clear"),
    ((None, None), None),
    (("commercial-conditional", "something-new"), "something-new"),
])
def test_a_model_made_with_several_takes_the_most_restricted_class(classes, strictest):
    from image_to_3dlab import provenance

    assert provenance.stricter_class(*classes) == strictest
