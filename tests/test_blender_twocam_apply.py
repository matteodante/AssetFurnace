"""Tests for the bpy-free parts of `scripts/blender_twocam_apply.py`.

Blender itself is exercised by running the script on a real rig (see docs/twocam-mocap.md);
these cover the argument parsing, the F-curve walk across Blender's two action layouts,
and the limb-paint unpicking, with stand-ins shaped like bpy's objects.
"""

from types import SimpleNamespace as NS

import blender_twocam_apply as apply
import pytest


def test_arguments_after_the_double_dash_are_read():
    opts = apply.parse_args(["blender", "-b", "--", "rig.json", "m.json", "out", "--armature", "rig",
                             "--name", "roar", "--keep-paint"])
    assert (opts["armature"], opts["name"], opts["keep_paint"], opts["glb"]) == ("rig", "roar", True, True)
    assert apply.parse_args(["--", "a", "b", "c", "--no-glb"])["glb"] is False


def test_a_flag_without_its_value_is_an_error():
    with pytest.raises(SystemExit):
        apply.parse_args(["--", "a", "b", "c", "--armature"])


def test_fcurves_are_found_in_blender_5s_layered_actions_and_in_old_ones():
    layered = NS(layers=[NS(strips=[NS(channelbags=[NS(fcurves=["a", "b"]), NS(fcurves=["c"])])])])
    assert list(apply.iter_fcurves(layered)) == ["a", "b", "c"]
    assert list(apply.iter_fcurves(NS(layers=[], fcurves=["x"]))) == ["x"]
    assert list(apply.iter_fcurves(None)) == []


def test_identity_check():
    eye = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    assert apply.is_identity(eye)
    eye[0][3] = 0.5
    assert not apply.is_identity(eye)


def test_joint_error_in_millimetres():
    assert apply.max_joint_error_mm([[[0, 0, 0], [1, 0, 0]]], [[[0, 0, 0.002], [1, 0, 0]]]) == pytest.approx(2.0)
    assert apply.max_joint_error_mm(None, [[[0, 0, 0]]]) is None


class Node(NS):
    def __eq__(self, other):  # bpy compares wrappers by what they point at
        return isinstance(other, Node) and self.name == other.name

    __hash__ = object.__hash__


def _painted_material(textured=True):
    vc = Node(name="Color Attribute", type="VERTEX_COLOR", layer_name="legpaint")
    mix = Node(name="Mix", type="MIX")
    tex = Node(name="Image Texture", type="TEX_IMAGE")
    bsdf = Node(name="Principled BSDF", type="BSDF_PRINCIPLED")
    tex_out = NS(node=tex)
    links = [NS(from_node=vc, to_node=mix, from_socket="alpha"),
             NS(from_node=vc, to_node=mix, from_socket="colour"),
             NS(from_node=mix, to_node=bsdf, from_socket="result")]
    if textured:
        links.append(NS(from_node=tex, to_node=mix, from_socket=tex_out))
    return NS(use_nodes=True, node_tree=NS(links=links)), mix, vc, tex_out


def test_the_paint_mix_and_the_texture_feeding_it_are_found():
    material, mix, vc, tex_out = _painted_material()
    found = apply.paint_mix_nodes(material)
    assert len(found) == 1
    assert found[0][0] == mix and found[0][1] == vc and found[0][2] is tex_out


def test_an_untextured_paint_mix_is_found_with_no_source():
    material, *_ = _painted_material(textured=False)
    assert apply.paint_mix_nodes(material)[0][2] is None


def test_materials_without_paint_are_left_alone():
    assert apply.paint_mix_nodes(NS(use_nodes=False)) == []
    assert apply.paint_mix_nodes(None) == []
