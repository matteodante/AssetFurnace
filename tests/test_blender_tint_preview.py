"""Tests for the pure parts of the tint preview: parsing, tint terms and light aiming.

The Blender half only wires these numbers into nodes; the numbers are checked here
against the shipping formula in ``image_to_3dlab.materials``.
"""

import numpy as np
import pytest
from blender_tint_preview import (
    light_rotation,
    mask_for_material,
    parse_tints,
    tint_terms,
)

from image_to_3dlab import materials as m


def test_parse_tints_reads_hex_as_linear():
    tints = parse_tints(["R=#ff0000", "B=#808080"])
    assert tints["R"] == pytest.approx([1.0, 0.0, 0.0])
    assert tints["B"] == pytest.approx([m.srgb_to_linear(np.array(128 / 255)).item()] * 3)
    with pytest.raises(ValueError):
        parse_tints(["Q=#ffffff"])
    with pytest.raises(ValueError):
        parse_tints(["R=red"])


def test_tint_terms_reproduce_apply_tint():
    """Per zone: out = albedo + mask * Y(albedo) * scale * delta, as the nodes compute it."""
    zones = [{"channel": "R", "mean_linear": [0.2, 0.05, 0.04]},
             {"channel": "G", "mean_linear": [0.1, 0.1, 0.1]}]
    tints = {"R": [0.02, 0.05, 0.3]}
    terms = tint_terms(zones, tints)
    assert [t["channel"] for t in terms] == ["R"]          # untouched zones add nothing

    rng = np.random.default_rng(0)
    albedo = rng.uniform(0.0, 0.5, (4, 4, 3))
    mask = np.zeros((4, 4, 3))
    mask[:2, :, 0] = 1.0
    nodes = albedo.copy()
    for term in terms:
        index = "RGB".index(term["channel"])
        nodes += (mask[..., index] * m.luminance(albedo) * term["scale"])[..., None] * np.array(term["delta"])
    means = np.array([z["mean_linear"] for z in zones] + [[0.0, 0.0, 0.0]])
    wanted = means.copy()
    wanted[0] = tints["R"]
    np.testing.assert_allclose(np.clip(nodes, 0, 1), m.apply_tint(albedo, mask, means, wanted))


def test_light_rotation_points_minus_z_along_the_light():
    """A Blender light shines down its local -Z; the rotation must turn -Z onto `toward`."""
    for toward in ([0, 0, -1], [1, 0, 0], [0.3, -0.8, -0.5]):
        rx, ry, rz = light_rotation(toward)
        cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
        rot = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
               @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
               @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
        shine = rot @ np.array([0, 0, -1.0])
        np.testing.assert_allclose(shine, np.array(toward) / np.linalg.norm(toward), atol=1e-9)


def test_each_blender_material_gets_its_own_mask():
    sidecar = {"textures": {"tint_mask": "T_k_Tint.png"}, "materials": [
        {"material": 0, "name": "head", "textures": {"tint_mask": "T_k_head_Tint.png"}},
        {"material": 1, "name": "body", "textures": {"tint_mask": "T_k_body_Tint.png"}}]}
    assert mask_for_material(sidecar, "body") == "T_k_body_Tint.png"
    assert mask_for_material(sidecar, "head.001") == "T_k_head_Tint.png"   # Blender's duplicate suffix
    assert mask_for_material(sidecar, "other") is None
    single = {"textures": {"tint_mask": "T_k_Tint.png"},
              "materials": [{"material": 0, "name": "x", "textures": {"tint_mask": "T_k_Tint.png"}}]}
    assert mask_for_material(single, "x") == "T_k_Tint.png"
    assert mask_for_material(single, "anything") is None
    twins = {"materials": [{"material": 0, "name": "arm", "textures": {"tint_mask": "a.png"}},
                           {"material": 1, "name": "arm", "textures": {"tint_mask": "b.png"}}]}
    assert mask_for_material(twins, "arm") is None                         # ambiguous: leave it alone


def test_an_exact_blender_name_wins_over_the_suffix_rule():
    """'metal.001' can be the material's real name, not only Blender's duplicate suffix."""
    sidecar = {"materials": [{"material": 0, "name": "metal.001", "textures": {"tint_mask": "a.png"}},
                             {"material": 1, "name": "metal", "textures": {"tint_mask": "b.png"}}]}
    assert mask_for_material(sidecar, "metal.001") == "a.png"
    assert mask_for_material(sidecar, "metal") == "b.png"


def test_a_sole_mask_is_given_only_to_its_own_material():
    """One textured material beside a factor-only one: the plain one gets no mask."""
    sidecar = {"textures": {"tint_mask": "T_k_Tint.png"},
               "materials": [{"material": 0, "name": "cloth", "textures": {"tint_mask": "T_k_Tint.png"}}]}
    assert mask_for_material(sidecar, "cloth") == "T_k_Tint.png"
    assert mask_for_material(sidecar, "cloth.001") == "T_k_Tint.png"
    assert mask_for_material(sidecar, "plain") is None
