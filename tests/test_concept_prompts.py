"""The four ways of turning a reference into a concept, as the words sent with the pictures."""

import pytest

from image_to_3dlab.concept import prompts


def test_redraw_names_the_map_and_writes_the_colours_instead_of_showing_them():
    text = prompts.redraw("werewolf", "grey fur and pale claws", structure="depth", parts="fur, claws")
    assert text.startswith("Image 1 is a depth map (lighter is nearer) of a werewolf.")
    assert "fills exactly the outline and pose of image 1" in text
    assert "with grey fur and pale claws, designed from scratch (fur, claws)" in text
    assert text.endswith(prompts.BACKGROUND + ".")


def test_redraw_from_a_line_drawing_says_so():
    assert prompts.redraw("robot", "an orange body", structure="lineart").startswith(
        "Image 1 is a line drawing of a robot.")


def test_a_redraw_needs_its_colours_in_words():
    with pytest.raises(ValueError, match="colours"):
        prompts.redraw("robot", "  ")


def test_an_unknown_structure_or_view_is_refused():
    with pytest.raises(ValueError, match="structure"):
        prompts.redraw("robot", "orange", structure="normals")
    with pytest.raises(ValueError, match="view"):
        prompts.other_view("three_quarter")


def test_dress_keeps_the_body_or_follows_the_skeleton():
    on_body = prompts.dress("a ranger in a green tunic", guide="body")
    assert on_body.startswith("Dress the character in image 1 as a ranger in a green tunic.")
    assert "same body, proportions, head size, T-pose" in on_body
    on_pose = prompts.dress("a ranger in a green tunic", guide="pose")
    assert on_pose.startswith("Image 1 is an OpenPose skeleton. Draw a ranger in a green tunic,")
    assert "exactly at the skeleton's joints" in on_pose


def test_add_piece_changes_nothing_else():
    text = prompts.add_piece("a steel helmet")
    assert text.startswith("Add a steel helmet to the character in image 1, fitted closely to the body.")
    assert "Change nothing else" in text


def test_other_view_with_and_without_the_depth_of_that_side():
    side = prompts.other_view("side")
    assert "seen from its left side, in profile" in side and "Image 2" not in side
    back = prompts.other_view("back", depth=True)
    assert "seen from behind" in back
    assert back.endswith("Image 2 is a depth map of its body seen from behind (lighter is nearer): "
                         "follow its outline exactly.")


def test_the_style_words_go_before_the_background_and_empty_style_leaves_no_gap():
    styled = prompts.add_piece("a helmet", style="hand-painted look")
    assert styled.endswith("hand-painted look, " + prompts.BACKGROUND + ".")
    plain = prompts.add_piece("a helmet")
    assert ", ," not in plain and ". ," not in plain


def test_how_many_shape_pictures_each_way_needs():
    assert prompts.shape_pictures("redraw") == 1
    assert prompts.shape_pictures("other_view", depth=True) == 2
    assert prompts.shape_pictures("other_view") == 1
    with pytest.raises(ValueError):
        prompts.shape_pictures("inpaint")
