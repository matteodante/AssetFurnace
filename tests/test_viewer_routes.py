"""The studio server's route tables (viewer/generate_api.py): every path it answers itself,
in the order it tries them, with the method that answers. A route added, moved or removed
changes this list on purpose; the guards before the routes are tests/test_viewer_request_guard.py."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import generate_api  # noqa: E402

ROUTES = [
    ("GET", "/api/hf/status", "_get_hf_status"),
    ("GET", "/api/catalog", "_get_catalog"),
    ("GET", "/api/recipes", "_get_recipes"),
    ("GET", "/api/chains", "_get_chains"),
    ("GET", "/api/chains/{}", "_get_chain"),
    ("GET", "/api/thumb", "_get_thumb"),
    ("GET", "/api/activity", "_get_activity"),
    ("GET", "/api/assets", "_get_assets"),
    ("GET", "/api/update-check", "_get_update_check"),
    ("GET", "/api/welcome", "_get_welcome"),
    ("GET", "/api/setup", "_get_setup"),
    ("GET", "/api/image/{}/{}", "_get_image"),
    ("GET", "/api/image/defaults", "_get_image_defaults"),
    ("GET", "/api/backends", "_get_backends"),
    ("GET", "/api/setup/{}/{events|status}", "_get_download"),
    ("GET", "/api/setup/run/{}/events", "_get_setup_events"),
    ("GET", "/api/generate/{}/events", "_events"),
    ("GET", "/api/generate/{}/status", "_status"),
    ("GET", "/api/generate/{}/{manifest.json|result.glb}", "_artifact"),
    ("GET", "/api/finish/capabilities", "_get_finish_capabilities"),
    ("GET", "/api/finish/runs", "_get_finish_runs"),
    ("GET", "/api/finish/{}/events", "_finish_events"),
    ("GET", "/api/finish/{}/status", "_finish_status"),
    ("GET", "/api/finish/{}/{record.json|result.glb}", "_finish_artifact"),
    ("GET", "/api/animate/models", "_get_animate_models"),
    ("GET", "/api/animate/{}/{}", "_get_animate"),
    ("GET", "/api/props/tools", "_get_props_tools"),
    ("GET", "/api/props/runs", "_get_props_runs"),
    ("GET", "/api/props/{}/{}", "_get_props"),
    ("GET", "/api/rig/rebind/{}/events", "_rig_events"),
    ("GET", "/api/rig/rebind/{}/status", "_rig_status"),
    ("GET", "/api/rig/rebind/{}/{report.json|result.glb|rig.json|scene.blend}", "_rig_artifact"),
    ("POST", "/api/chains", "_post_chain"),
    ("POST", "/api/chains/{}/cancel", "_cancel_chain"),
    ("POST", "/api/blender/install", "_start_blender_install"),
    ("POST", "/api/gltfpack/install", "_start_gltfpack_install"),
    ("POST", "/api/hf/sign-in", "_post_hf_sign_in"),
    ("POST", "/api/setup/run", "_start_setup"),
    ("POST", "/api/setup/{}/{cancel|download|rebuild|remove}", "_backend_download"),
    ("POST", "/api/generate", "_create_job"),
    ("POST", "/api/trellis/input-advice", "_trellis_input_advice"),
    ("POST", "/api/rig/rebind", "_create_rig_job"),
    ("POST", "/api/generate/{}/cancel", "_cancel_job"),
    ("POST", "/api/rig/rebind/{}/cancel", "_cancel_rig_job"),
    ("POST", "/api/finish", "_create_finish_job"),
    ("POST", "/api/finish/{}/cancel", "_cancel_finish_job"),
    ("POST", "/api/finish/runs/{}/resume", "_resume_finish_job"),
    ("POST", "/api/props", "_create_props_job"),
    ("POST", "/api/animate/rig", "_create_animate_rig"),
    ("POST", "/api/animate/play", "_create_animate_play"),
    ("POST", "/api/animate/{}/cancel", "_cancel_animate"),
    ("POST", "/api/props/{}/cancel", "_cancel_props_job"),
    ("POST", "/api/props/runs/{}/turn", "_turn_prop"),
    ("POST", "/api/image", "_create_image_job"),
    ("POST", "/api/assets/{}/name", "_post_asset_name"),
    ("POST", "/api/assets/{}/hidden", "_post_asset_hidden"),
    ("POST", "/api/image/{}/cancel", "_cancel_image_job"),
]


def test_the_route_list_is_pinned():
    tables = (("GET", generate_api.GET_ROUTES), ("POST", generate_api.POST_ROUTES))
    assert [(method, route.text(), route.handler) for method, routes in tables for route in routes] == ROUTES


def test_the_pages_answer_first_in_this_order():
    assert [page.__name__ for page in generate_api.PAGE_APIS] == [
        "blender_edit_api", "review_api", "material_labels_api", "concept_api", "unreal_export_api", "batch_api",
        "slot_parts_api"]
    assert generate_api.FRONT_DOOR == ("", "/", "/viewer", "/viewer/")


def test_every_route_names_a_handler_that_takes_its_parts():
    for route in (*generate_api.GET_ROUTES, *generate_api.POST_ROUTES):
        handler = getattr(generate_api.Handler, route.handler)
        given = sum(not isinstance(want, str) for want in route.parts)
        assert len(inspect.signature(handler).parameters) - 1 == given, route


def test_a_route_matches_its_parts_and_hands_over_the_others():
    route = generate_api.Route(("api", "setup", generate_api.ANY, ("events", "status")), "_get_download")
    assert route.match(["api", "setup", "trellis", "events"]) == ["trellis", "events"]
    assert route.match(["api", "setup", "trellis", "remove"]) is None
    assert route.match(["api", "setup", "trellis"]) is None
    assert route.match(["api", "setup", "trellis", "status", "x"]) is None
    assert generate_api.Route(("api", "catalog"), "_get_catalog").match(["api", "catalog"]) == []
