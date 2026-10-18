"""viewer/request_utils.py: the one place the Studio APIs check a request, read its body and
find the files it names (code read 18/10/2026: five copies with five sets of rules)."""

from __future__ import annotations

import io
from email.message import Message

import pytest
import request_utils as ru

BAD = ["", "../secret.png", "images/../../secret.png", "/etc/passwd", "C:/Windows/win.ini", "images\\a.png",
       "\\\\server\\share\\a.png"]


@pytest.fixture
def output(tmp_path):
    out = tmp_path / "output"
    (out / "images").mkdir(parents=True)
    (out / "images" / "a.png").write_bytes(b"png")
    (out / "models").mkdir()
    (out / "models" / "m.glb").write_bytes(b"glb")
    (tmp_path / "secret.png").write_bytes(b"no")
    return out


def test_a_library_file_is_found_and_everything_else_is_refused(output):
    assert ru.inside_output(output, "images/a.png") == (output / "images" / "a.png").resolve()
    for relative in [*BAD, None, 3]:
        with pytest.raises(ru.OutsideOutput):
            ru.inside_output(output, relative)
    with pytest.raises(ru.OutsideOutput):
        ru.inside_output(output, "images/a.png", suffixes=(".glb",))
    with pytest.raises(FileNotFoundError, match="missing.png is gone"):
        ru.inside_output(output, "images/missing.png", missing="{relative} is gone")


def test_every_studio_api_holds_paths_by_the_same_rule(output):
    """The five pages that each had their own copy now refuse the same paths."""
    import concept_api
    import review_api
    import slot_parts_api
    import unreal_export_api

    checks = [lambda r: review_api._inside(output, r, "asset"), lambda r: concept_api.inside(output, r),
              lambda r: slot_parts_api._inside(output, r), lambda r: unreal_export_api.resolve_picture(output, r)]
    for check in checks:
        assert check("images/a.png") == (output / "images" / "a.png").resolve()
        for relative in BAD:
            with pytest.raises(ValueError):
                check(relative)
    assert unreal_export_api.resolve(output, "models/m.glb") == (output / "models" / "m.glb").resolve()


class _Handler:
    def __init__(self, body: bytes, lengths: list[str]):
        self.headers = Message()
        for value in lengths:
            self.headers["Content-Length"] = value
        self.rfile = io.BytesIO(body)
        self.close_connection = False


def test_a_body_is_read_once_with_a_single_content_length():
    assert ru.read_body(_Handler(b'{"a": 1}', ["8"]), 100) == b'{"a": 1}'
    assert ru.read_json(_Handler(b'{"a": 1}', ["8"]), 100) == {"a": 1}
    for lengths in ([], ["8", "8"], ["x"], ["200"]):
        assert ru.read_body(_Handler(b"x" * 200, lengths), 100) == b""
    with pytest.raises(ValueError):
        ru.read_json(_Handler(b"not json", ["8"]), 100)
    handler = _Handler(b"12345", ["5"])
    ru.drain(handler)
    assert handler.rfile.read() == b"" and handler.close_connection


def test_the_old_names_still_reach_the_shared_checks():
    import blender_edit_api

    assert blender_edit_api.check_request is ru.check_request and blender_edit_api.drain is ru.drain
    assert blender_edit_api.local_address is ru.local_address and blender_edit_api.check_host is ru.check_host
