"""Tests for game-ready materials: tint mask, baked-light checks, derived maps, ORM.

Every fixture is built here, small and synthetic: a UV sphere whose texture rows map to
latitude, so a texel's surface normal is known exactly and light can be painted into
the albedo on purpose. The shipping functions are imported, never re-derived.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import materials as m
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb, parse_glb

SIZE = 128
RED, GREEN, BLUE = (200, 40, 30), (40, 160, 60), (40, 60, 190)
LIGHT = np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])


# --------------------------------------------------------------------------- fixtures

def sphere(rings: int = 32, segments: int = 64):
    """Lat-long sphere: v = 0 is the +Y pole, so texture row tracks latitude."""
    positions, uvs = [], []
    for i in range(rings + 1):
        theta = np.pi * i / rings
        for j in range(segments + 1):
            phi = 2 * np.pi * j / segments
            positions.append((np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)))
            uvs.append((j / segments, i / rings))
    faces = []
    row = segments + 1
    for i in range(rings):
        for j in range(segments):
            a, b = i * row + j, i * row + j + 1
            c, d = a + row, b + row
            faces += [(a, b, c), (b, d, c)]          # outward, counter-clockwise
    return np.array(positions, np.float32), np.array(uvs, np.float32), np.array(faces, np.uint32)


def texel_normals(size: int = SIZE) -> np.ndarray:
    """The sphere's outward normal under every texel centre."""
    v, u = (np.mgrid[0:size, 0:size] + 0.5) / size
    theta, phi = np.pi * v, 2 * np.pi * u
    return np.dstack([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])


def banded_albedo(colours=(RED, GREEN, BLUE), size: int = SIZE, noise: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    rows = np.arange(size) / size
    band = np.minimum((rows * len(colours)).astype(int), len(colours) - 1)
    img = np.array(colours, np.float32)[band][:, None, :].repeat(size, axis=1)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def lit(albedo: np.ndarray, ambient: float) -> np.ndarray:
    """Paint a Lambert key light into an albedo, in linear light."""
    shade = ambient + (1 - ambient) * np.clip(texel_normals(albedo.shape[0]) @ LIGHT, 0, 1)
    linear = m.srgb_to_linear(albedo / 255.0) * shade[..., None]
    return np.clip(m.linear_to_srgb(linear) * 255 + 0.5, 0, 255).astype(np.uint8)


def png(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def make_glb(path, albedo: np.ndarray, mr: np.ndarray | None = None, rigged: bool = False,
             metallic_factor: float | None = 0.0, roughness_factor: float = 1.0,
             occlusion_from_mr: bool = False, base_extensions: dict | None = None):
    positions, uvs, faces = sphere()
    blobs, views, accessors = [], [], []

    def add_view(data: bytes, target=None):
        offset = sum(len(b) for b in blobs)
        blobs.append(data + b"\0" * (-len(data) % 4))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    def add_accessor(array: np.ndarray, kind: str, component: int, target=None, minmax=False):
        view = add_view(array.tobytes(), target)
        acc = {"bufferView": view, "componentType": component, "count": len(array), "type": kind}
        if minmax:
            acc["min"], acc["max"] = array.min(0).tolist(), array.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    pos = add_accessor(positions, "VEC3", 5126, 34962, minmax=True)
    uv = add_accessor(uvs, "VEC2", 5126, 34962)
    idx = add_accessor(faces.reshape(-1), "SCALAR", 5125, 34963)
    images = [{"bufferView": add_view(png(albedo)), "mimeType": "image/png", "name": "albedo"}]
    textures = [{"source": 0, "sampler": 0}]
    pbr = {"baseColorTexture": {"index": 0}, "roughnessFactor": roughness_factor}
    if metallic_factor is not None:
        pbr["metallicFactor"] = metallic_factor
    if base_extensions:
        pbr["baseColorTexture"]["extensions"] = base_extensions
    if mr is not None:
        images.append({"bufferView": add_view(png(mr)), "mimeType": "image/png", "name": "mr"})
        textures.append({"source": 1, "sampler": 0})
        pbr["metallicRoughnessTexture"] = {"index": 1}
    attributes = {"POSITION": pos, "TEXCOORD_0": uv}
    document = {
        "asset": {"version": "2.0", "generator": "test_materials"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "body"}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": idx, "material": 0}]}],
        "materials": [{"name": "skin", "pbrMetallicRoughness": pbr}],
        "textures": textures, "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987}],
    }
    if rigged:
        joints = add_accessor(np.zeros((len(positions), 4), np.uint8), "VEC4", 5121, 34962)
        weights = add_accessor(np.tile([1.0, 0, 0, 0], (len(positions), 1)).astype(np.float32),
                               "VEC4", 5126, 34962)
        attributes.update({"JOINTS_0": joints, "WEIGHTS_0": weights})
        ibm = add_accessor(np.eye(4, dtype=np.float32).reshape(1, 16), "MAT4", 5126)
        times = add_accessor(np.array([0.0, 1.0], np.float32), "SCALAR", 5126, minmax=True)
        values = add_accessor(np.array([[0, 0, 0, 1], [0, 0.38, 0, 0.92]], np.float32), "VEC4", 5126)
        document["nodes"] = [{"mesh": 0, "skin": 0, "name": "body"}, {"name": "root"}]
        document["scenes"] = [{"nodes": [0, 1]}]
        document["skins"] = [{"joints": [1], "inverseBindMatrices": ibm}]
        document["animations"] = [{"name": "turn",
                                   "samplers": [{"input": times, "output": values}],
                                   "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}]}]
    if occlusion_from_mr:
        document["materials"][0]["occlusionTexture"] = {"index": 1, "strength": 0.8}
    if base_extensions:
        document["extensionsUsed"] = sorted(base_extensions)
    binary = b"".join(blobs)
    document["buffers"] = [{"byteLength": len(binary)}]
    document["bufferViews"] = views
    document["accessors"] = accessors
    path.write_bytes(build_glb(document, binary))
    return path


@pytest.fixture
def banded(tmp_path):
    return make_glb(tmp_path / "banded.glb", banded_albedo())


def statuses(records):
    return {r["check"]: r["status"] for r in records}


# --------------------------------------------------------------------------- colour

def test_srgb_round_trip_and_linear_luminance():
    values = np.linspace(0, 1, 11)
    np.testing.assert_allclose(m.linear_to_srgb(m.srgb_to_linear(values)), values, atol=1e-6)
    assert m.srgb_to_linear(np.array(0.5)) == pytest.approx(0.214, abs=1e-3)
    assert m.luminance(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)


def test_three_distinct_colours_make_three_zones():
    rng = np.random.default_rng(0)
    lab = np.concatenate([rng.normal(c, 1.5, (n, 3)) for c, n in
                          (((50, 60, 40), 500), ((60, -50, 40), 300), ((40, 10, -60), 200))])
    clusters = m.cluster_colours(m.zone_features(lab), seed=0)
    assert len(clusters.shares) == 3
    np.testing.assert_allclose(sorted(clusters.shares, reverse=True), [0.5, 0.3, 0.2], atol=0.01)


def test_lightness_steps_alone_do_not_split_a_zone():
    """Shading inside one material is lightness only; at half weight it stays one zone."""
    rng = np.random.default_rng(0)
    lab = np.c_[rng.uniform(30, 50, 2000), rng.normal(20, 1, 2000), rng.normal(10, 1, 2000)]
    assert len(m.cluster_colours(m.zone_features(lab), seed=0).shares) == 1


def test_a_saturated_material_in_shade_stays_one_zone():
    """Shade lowers lightness and chroma together and keeps hue: same pigment, one zone."""
    rng = np.random.default_rng(0)
    lit_red, shaded_red = (30, 64, 34), (22, 48, 22)      # 20 apart, same hue and C/L
    lab = np.concatenate([rng.normal(c, 1.0, (600, 3)) for c in (lit_red, shaded_red)])
    assert len(m.cluster_colours(m.zone_features(lab), seed=0).shares) == 1


def test_skin_is_not_the_red_cloth_beside_it():
    """Measured on the warrior: skin merged into red cloth when k-means started too coarse."""
    rng = np.random.default_rng(0)
    groups = (((4, -2, 0), 3400), ((16, 4, 7), 2000), ((14, 29, 15), 2000),
              ((32, 0, 0), 1700), ((56, 27, 27), 250), ((57, 6, 1), 300))
    lab = np.concatenate([rng.normal(c, 1.5, (n, 3)) for c, n in groups])
    clusters = m.cluster_colours(m.zone_features(lab), seed=0)
    skin = clusters.labels[-550:-300]
    cloth = clusters.labels[5400:7400]
    assert np.bincount(skin).argmax() != np.bincount(cloth).argmax()


def test_same_pigment_needs_matching_hue_and_chroma_ratio():
    assert m.same_pigment((30, 64, 34), (22, 48, 22))
    assert not m.same_pigment((56, 27, 27), (15, 32, 17))     # skin vs red: hue differs
    assert not m.same_pigment((60, 10, 5), (20, 10, 5))       # same hue, chroma not scaled
    assert not m.same_pigment((30, 2, 3), (20, 1, 2))         # greys: hue is noise


# --------------------------------------------------------------------------- tint zones

def test_banded_sphere_passes_tint_zones_with_three_channels(banded, tmp_path):
    result = m.make_game_materials(banded, tmp_path / "out", samples=20000)
    assert statuses(result.checks)["tint_zones"] == "pass"
    assert [z["channel"] for z in result.zones] == ["R", "G", "B"]
    # The equator band covers half a sphere's area, each cap a quarter.
    assert result.zones[0]["area_share"] == pytest.approx(0.5, abs=0.04)
    mask = np.asarray(Image.open(result.textures["tint_mask"]))
    rows = (np.arange(SIZE) + 0.5) / SIZE
    middle, top = (rows > 0.4) & (rows < 0.6), rows < 0.25
    assert mask[middle][..., 0].mean() > 250          # equator = largest zone = R
    assert mask[middle][..., 1:].max() < 5
    top_zone = "GB"[int(np.argmax([mask[top][..., 1].mean(), mask[top][..., 2].mean()]))]
    assert mask[top][..., "RGB".index(top_zone)].mean() > 250


def test_uniform_albedo_fails_tint_zones_and_writes_no_glb(tmp_path):
    flat = make_glb(tmp_path / "flat.glb", banded_albedo(colours=((230, 228, 225),)))
    result = m.make_game_materials(flat, tmp_path / "out", samples=20000)
    assert statuses(result.checks)["tint_zones"] == "fail"
    assert result.status == "fail"
    assert result.glb_path is None
    assert result.sidecar_path.exists()            # the reason travels back with the asset


def test_two_colours_warn_and_leave_the_third_channel_empty(tmp_path):
    two = make_glb(tmp_path / "two.glb", banded_albedo(colours=(RED, BLUE)))
    result = m.make_game_materials(two, tmp_path / "out", samples=20000)
    assert statuses(result.checks)["tint_zones"] == "warn"
    assert result.status == "warn"
    mask = np.asarray(Image.open(result.textures["tint_mask"]))
    assert mask[..., 2].max() == 0
    assert result.glb_path is not None


def test_zones_from_materials_needs_material_slots(banded, tmp_path):
    """One material slot is one zone: not enough to tint, so it goes back."""
    result = m.make_game_materials(banded, tmp_path / "out", zones_from="materials", samples=20000)
    assert statuses(result.checks)["tint_zones"] == "fail"


# --------------------------------------------------------------------------- tint formula

def test_default_tint_reproduces_the_albedo_exactly():
    rng = np.random.default_rng(0)
    albedo = rng.uniform(0.05, 0.9, (8, 8, 3))
    mask = np.zeros((8, 8, 3))
    mask[:4, :, 0] = 1.0
    means = np.array([albedo[:4].reshape(-1, 3).mean(0), [0.2, 0.2, 0.2], [0.1, 0.3, 0.5]])
    np.testing.assert_allclose(m.apply_tint(albedo, mask, means, means), albedo, atol=1e-9)


def test_tint_moves_the_zone_mean_to_the_tint_and_leaves_the_rest():
    rng = np.random.default_rng(0)
    # One material: a single colour carrying brightness detail, as a de-lit albedo does.
    albedo = rng.uniform(0.5, 1.0, (8, 8, 1)) * np.array([0.6, 0.18, 0.12])
    mask = np.zeros((8, 8, 3))
    mask[:4, :, 0] = 1.0
    mean = albedo[:4].reshape(-1, 3).mean(0)
    means = np.array([mean, mean, mean])
    tints = means.copy()
    tints[0] = [0.05, 0.1, 0.3]
    out = m.apply_tint(albedo, mask, means, tints)
    np.testing.assert_allclose(out[:4].reshape(-1, 3).mean(0), tints[0], atol=1e-6)
    np.testing.assert_allclose(out[4:], albedo[4:])
    # Brightness detail inside the zone survives the recolour.
    assert np.corrcoef(m.luminance(out[:4]).ravel(), m.luminance(albedo[:4]).ravel())[0, 1] > 0.99


# --------------------------------------------------------------------------- baked light

def test_clean_albedo_passes_every_light_check(banded):
    records = m.run_checks(banded, samples=20000)
    got = statuses(records)
    assert got["baked_light_direction"] == "pass"
    assert got["baked_light_highlights"] == "pass"
    assert got["baked_light_occlusion"] == "skipped"     # no AO bake was given


def test_painted_key_light_fails_and_points_at_the_light(tmp_path):
    asset = make_glb(tmp_path / "lit.glb", lit(banded_albedo(), ambient=0.4))
    record = {r["check"]: r for r in m.run_checks(asset, samples=20000)}["baked_light_direction"]
    assert record["status"] == "fail"
    assert record["value"] > 2.0                      # a lone sphere shows it plainly
    assert np.dot(record["details"]["direction"], LIGHT) > 0.9


def test_light_ratio_tracks_the_painted_ambient(tmp_path):
    """Ambient 0.8 is a soft wash (ratio ~1.27, pass); 0.75 lands in warn."""
    soft = make_glb(tmp_path / "soft.glb", lit(banded_albedo(), ambient=0.8))
    mid = make_glb(tmp_path / "mid.glb", lit(banded_albedo(), ambient=0.75))
    assert statuses(m.run_checks(soft, samples=20000))["baked_light_direction"] == "pass"
    assert statuses(m.run_checks(mid, samples=20000))["baked_light_direction"] == "warn"


def spotted(centres_where, count: int, seed: int, radius2: int = 4) -> np.ndarray:
    """Banded albedo with small near-white discs at texels chosen by `centres_where`."""
    albedo = banded_albedo()
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(centres_where)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for i in rng.choice(len(rows), size=count, replace=False):
        albedo[(yy - rows[i]) ** 2 + (xx - cols[i]) ** 2 < radius2] = 250
    return albedo


def test_painted_highlights_fail(tmp_path):
    """Highlights one key light painted: bright spots on the side facing it."""
    facing = texel_normals() @ LIGHT > 0.85
    asset = make_glb(tmp_path / "shiny.glb", spotted(facing, 30, seed=3, radius2=9))
    record = {r["check"]: r for r in m.run_checks(asset, samples=20000)}["baked_light_highlights"]
    assert record["status"] == "fail"
    assert record["value"] > 0.02
    assert np.dot(record["details"]["direction"], LIGHT) > 0.8


def test_light_specks_facing_every_way_are_a_material_not_a_highlight(tmp_path):
    """Cream trim beside dark cloth is bright and colourless too, but faces everywhere."""
    band = np.zeros((SIZE, SIZE), bool)
    band[48:80] = True
    asset = make_glb(tmp_path / "trim.glb", spotted(band, 120, seed=5))
    record = {r["check"]: r for r in m.run_checks(asset, samples=20000)}["baked_light_highlights"]
    assert record["details"]["candidates"] > 0.05
    assert record["details"]["concentration"] < 0.4
    assert record["status"] == "pass"


def test_painted_occlusion_fails_when_the_ao_bake_is_given(tmp_path):
    rng = np.random.default_rng(4)
    ao = np.clip(rng.uniform(0.3, 1.0, (SIZE // 8, SIZE // 8)), 0, 1)
    ao = np.asarray(Image.fromarray((ao * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR))
    ao_path = tmp_path / "ao.png"
    Image.fromarray(ao).save(ao_path)
    albedo = banded_albedo()
    painted = m.srgb_to_linear(albedo / 255.0) * (ao[..., None] / 255.0)
    painted = (m.linear_to_srgb(painted) * 255 + 0.5).astype(np.uint8)
    dirty = make_glb(tmp_path / "dirty.glb", painted)
    clean = make_glb(tmp_path / "clean.glb", albedo)
    assert statuses(m.run_checks(dirty, ao=ao_path, samples=20000))["baked_light_occlusion"] == "fail"
    assert statuses(m.run_checks(clean, ao=ao_path, samples=20000))["baked_light_occlusion"] == "pass"


def test_failed_light_check_stops_before_writing_maps(tmp_path):
    asset = make_glb(tmp_path / "lit.glb", lit(banded_albedo(), ambient=0.3))
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    assert result.status == "fail"
    assert result.glb_path is None
    assert not any(p.suffix == ".png" for p in (tmp_path / "out").iterdir())


def test_a_rejected_rerun_leaves_no_game_ready_glb_behind(tmp_path):
    """A batch reads `<stem>.glb` as "done": an older pass must not survive a new rejection."""
    asset = tmp_path / "knight.glb"
    make_glb(asset, banded_albedo())
    first = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    assert first.glb_path.exists()
    make_glb(asset, lit(banded_albedo(), ambient=0.3))
    again = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    assert again.status == "fail"
    assert not (tmp_path / "out" / "knight.glb").exists()


def test_writing_into_the_assets_own_folder_is_refused(banded):
    """The output GLB shares the source's name, so that folder would overwrite the source."""
    before = banded.read_bytes()
    with pytest.raises(ValueError):
        m.make_game_materials(banded, banded.parent, samples=20000)
    assert banded.read_bytes() == before


def test_records_follow_the_shared_check_contract(banded):
    for record in m.run_checks(banded, samples=20000):
        assert {"check", "status", "value", "threshold", "message"} <= set(record)
        assert set(record) <= {"check", "status", "value", "threshold", "message", "details"}
        assert record["status"] in {"pass", "warn", "fail", "skipped"}
        json.dumps(record)                       # plain JSON, no numpy scalars


# --------------------------------------------------------------------------- derived maps

def test_flat_albedo_gives_a_flat_normal_even_at_island_borders():
    """Masked blurs keep the black gutter out, so seams do not emboss a rim."""
    lum = np.full((64, 64), 0.6)
    coverage = np.zeros((64, 64), bool)
    coverage[8:40, 8:56] = True
    lum[~coverage] = 0.0
    normal = m.normal_from_albedo(lum, coverage, strength=8.0)
    inside = normal[coverage].astype(int)
    assert np.abs(inside[:, 0] - 128).max() <= 1
    assert np.abs(inside[:, 1] - 128).max() <= 1
    assert inside[:, 2].min() >= 254


def test_normal_convention_matches_surface_detail():
    from surface_detail import normal_from_height as reference
    ramp = np.tile(np.linspace(0.0, 1.0, 32), (32, 1))
    ours = m.normal_from_height(ramp, strength=4.0)
    theirs = reference(ramp.astype(np.float32), strength=4.0)
    np.testing.assert_array_equal(ours, theirs)


def test_brighter_stripe_reads_as_raised():
    lum = np.full((64, 64), 0.3)
    lum[:, 32:] = 0.7
    normal = m.normal_from_albedo(lum, np.ones((64, 64), bool), strength=32.0).astype(int)
    # Height rises to the right across the edge, so X tilts below neutral there.
    assert normal[:, 30:34, 0].mean() < 120


def test_orm_keeps_existing_varying_maps_and_scales_by_factors(tmp_path):
    rng = np.random.default_rng(5)
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    mr[..., 2] = 255
    asset = make_glb(tmp_path / "pbr.glb", banded_albedo(), mr=mr, metallic_factor=0.5)
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    orm = np.asarray(Image.open(result.textures["orm"]))
    assert result.maps["roughness"] == "kept"
    np.testing.assert_array_equal(orm[..., 1], mr[..., 1])
    assert np.abs(orm[..., 2].astype(int) - 128).max() <= 1     # 255 x metallicFactor 0.5
    assert orm[..., 0].min() == 255                              # no AO baked: fully open


def test_constant_roughness_is_derived_per_zone(tmp_path):
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = 237                                             # TRELLIS-style constant
    asset = make_glb(tmp_path / "flat_mr.glb", banded_albedo(), mr=mr)
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000,
                                   roughness_by_zone={"R": 0.3})
    orm = np.asarray(Image.open(result.textures["orm"])).astype(int)
    assert result.maps["roughness"] == "derived"
    mask = np.asarray(Image.open(result.textures["tint_mask"]))
    zone_r = mask[..., 0] > 250
    assert abs(orm[zone_r][:, 1].mean() - 0.3 * 255) < 15
    assert orm[~zone_r & (mask.max(-1) > 250)][:, 1].mean() > 0.6 * 255


def test_pack_orm_puts_channels_in_gltf_order():
    ao, rough, metal = np.full((2, 2), 0.25), np.full((2, 2), 0.5), np.full((2, 2), 1.0)
    orm = m.pack_orm(ao, rough, metal)
    assert orm.dtype == np.uint8
    assert tuple(orm[0, 0]) == (64, 128, 255)


# --------------------------------------------------------------------------- GLB and sidecar

def test_rewritten_glb_keeps_geometry_rig_and_animation(tmp_path):
    source = make_glb(tmp_path / "rigged.glb", banded_albedo(), rigged=True)
    result = m.make_game_materials(source, tmp_path / "out", samples=20000)
    before_doc, before_bin = parse_glb(source.read_bytes())
    after = EditGLB(result.glb_path.read_bytes())
    for key in ("nodes", "skins", "animations", "meshes", "accessors"):
        assert after.document[key] == before_doc[key]
    image_views = {img["bufferView"] for img in before_doc["images"]}
    for index, view in enumerate(before_doc["bufferViews"]):
        if index in image_views:
            continue
        old = before_bin[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
        new_view = after.document["bufferViews"][index]
        start = new_view.get("byteOffset", 0)
        assert after.binary[start:start + new_view["byteLength"]] == old
    np.testing.assert_array_equal(after.accessor(0), sphere()[0])


def test_rewritten_material_declares_normal_orm_and_tint_mask(banded, tmp_path):
    result = m.make_game_materials(banded, tmp_path / "out", samples=20000)
    glb = EditGLB(result.glb_path.read_bytes())
    material = glb.document["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 1.0 and pbr["roughnessFactor"] == 1.0
    assert "occlusionTexture" not in material                    # nothing baked
    tint = material["extras"]["assetfurnace"]["tintMask"]

    def image_of(texture_index):
        image = glb.document["images"][glb.document["textures"][texture_index]["source"]]
        view = glb.document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        return np.asarray(Image.open(io.BytesIO(glb.binary[start:start + view["byteLength"]])))

    np.testing.assert_array_equal(image_of(tint["index"]), np.asarray(Image.open(result.textures["tint_mask"])))
    np.testing.assert_array_equal(image_of(pbr["metallicRoughnessTexture"]["index"]),
                                  np.asarray(Image.open(result.textures["orm"])))
    assert image_of(material["normalTexture"]["index"]).shape == (SIZE, SIZE, 3)
    assert result.maps["normal"] == "derived"


def test_occlusion_bake_is_packed_and_declared(banded, tmp_path):
    ao = np.full((SIZE, SIZE), 200, np.uint8)
    Image.fromarray(ao).save(tmp_path / "ao.png")
    result = m.make_game_materials(banded, tmp_path / "out", ao=tmp_path / "ao.png", samples=20000)
    material = EditGLB(result.glb_path.read_bytes()).document["materials"][0]
    assert material["occlusionTexture"] == material["pbrMetallicRoughness"]["metallicRoughnessTexture"]
    assert np.asarray(Image.open(result.textures["orm"]))[..., 0].max() == 200


def test_sidecar_carries_zones_formula_and_checks(banded, tmp_path):
    result = m.make_game_materials(banded, tmp_path / "out", samples=20000, texture_prefix="T_Probe")
    sidecar = json.loads(result.sidecar_path.read_text())
    assert sidecar["schema"] == m.SIDECAR_SCHEMA
    assert [z["channel"] for z in sidecar["zones"]] == ["R", "G", "B"]
    assert all(len(z["mean_linear"]) == 3 for z in sidecar["zones"])
    tint = sidecar["tint"]
    assert tint["space"] == "linear"
    assert tint["luminance_weights"] == list(m.LUMINANCE_WEIGHTS)
    assert "t - c" in tint["formula"]
    assert sidecar["textures"]["tint_mask"] == "T_Probe_Tint.png"
    assert sidecar["textures"]["orm_channels"] == {"R": "occlusion", "G": "roughness", "B": "metallic"}
    assert {r["check"] for r in sidecar["checks"]} == {
        "tint_zones", "baked_light_direction", "baked_light_occlusion", "baked_light_highlights"}


def test_existing_normal_map_is_kept(tmp_path):
    asset = make_glb(tmp_path / "n.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    normal = np.zeros((SIZE, SIZE, 3), np.uint8)
    normal[..., 0] = 100
    normal[..., 1] = 128
    normal[..., 2] = 255
    data = png(normal)
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
    binary += data + b"\0" * (-len(data) % 4)
    document["buffers"][0]["byteLength"] = len(binary)
    document["images"].append({"bufferView": len(document["bufferViews"]) - 1, "mimeType": "image/png"})
    document["textures"].append({"source": len(document["images"]) - 1, "sampler": 0})
    document["materials"][0]["normalTexture"] = {"index": len(document["textures"]) - 1}
    asset.write_bytes(build_glb(document, binary))
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    assert result.maps["normal"] == "kept"
    assert "normal" not in result.textures


def test_set_view_bytes_relays_the_buffer():
    document = {"buffers": [{"byteLength": 12}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4},
                                {"buffer": 0, "byteOffset": 4, "byteLength": 4},
                                {"buffer": 0, "byteOffset": 8, "byteLength": 4}]}
    binary = b"AAAABBBBCCCC"
    binary = m.set_view_bytes(document, binary, 1, b"XYZXYZ")
    views = document["bufferViews"]
    spans = [binary[v["byteOffset"]:v["byteOffset"] + v["byteLength"]] for v in views]
    assert spans == [b"AAAA", b"XYZXYZ", b"CCCC"]
    assert all(v["byteOffset"] % 4 == 0 for v in views)
    assert document["buffers"][0]["byteLength"] == len(binary)


# --------------------------------------------------------------------------- script

def test_script_exit_codes_follow_the_verdict(banded, tmp_path):
    import game_materials

    assert game_materials.main([str(banded), str(tmp_path / "ok"), "--samples", "20000"]) == 0
    assert (tmp_path / "ok" / "banded.glb").exists()
    lit_asset = make_glb(tmp_path / "lit.glb", lit(banded_albedo(), ambient=0.3))
    assert game_materials.main([str(lit_asset), str(tmp_path / "ko"), "--samples", "20000"]) == 2
    assert game_materials.main([str(banded), "--check-only", "--samples", "20000"]) == 0


def test_script_parses_zone_overrides():
    import game_materials

    assert game_materials.parse_zone_values("R=0.3,B=0.9") == {"R": 0.3, "B": 0.9}
    with pytest.raises(ValueError):
        game_materials.parse_zone_values("Q=0.3")
    assert struct.calcsize("<I") == 4


# --------------------------------------------------------------------------- review fixes

def test_packed_occlusion_already_in_the_asset_is_kept(tmp_path):
    """An ORM input already carries AO in red; without a new bake it must survive."""
    rng = np.random.default_rng(6)
    orm = np.zeros((SIZE, SIZE, 3), np.uint8)
    orm[..., 0] = rng.integers(90, 200, (SIZE, SIZE))
    orm[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "orm.glb", banded_albedo(), mr=orm, occlusion_from_mr=True)
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    out = np.asarray(Image.open(result.textures["orm"]))
    np.testing.assert_array_equal(out[..., 0], orm[..., 0])
    assert result.maps["occlusion"] == "kept"
    material = EditGLB(result.glb_path.read_bytes()).document["materials"][0]
    orm_index = material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
    assert material["occlusionTexture"]["index"] == orm_index
    assert material["occlusionTexture"]["strength"] == 0.8


def test_missing_metallic_data_means_dielectric(tmp_path):
    """No map and no factor: the design says 0, not glTF's default of 1."""
    asset = make_glb(tmp_path / "bare.glb", banded_albedo(), metallic_factor=None)
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    assert np.asarray(Image.open(result.textures["orm"]))[..., 2].max() == 0


def test_material_names_become_safe_file_names():
    assert m.safe_name("arm/left") == "arm-left"
    assert m.safe_name('a:b*c?"d<e>|f') == "a-b-c-d-e-f"
    assert m.safe_name("") == "material"
    assert m.safe_name("..") == "material"


def two_material_glb(path, names):
    """The banded sphere drawn twice, by two materials sharing its texture."""
    make_glb(path, banded_albedo())
    document, binary = parse_glb(path.read_bytes())
    primitive = document["meshes"][0]["primitives"][0]
    document["meshes"][0]["primitives"].append({**primitive, "material": 1})
    pbr = document["materials"][0]["pbrMetallicRoughness"]
    document["materials"] = [{"name": names[0], "pbrMetallicRoughness": dict(pbr)},
                             {"name": names[1], "pbrMetallicRoughness": dict(pbr)}]
    path.write_bytes(build_glb(document, binary))
    return path


def test_material_names_with_slashes_still_save(tmp_path):
    asset = two_material_glb(tmp_path / "two.glb", ["arm/left", "arm/left"])
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    sidecar = json.loads(result.sidecar_path.read_text())
    files = [x["textures"]["tint_mask"] for x in sidecar["materials"]]
    assert len(set(files)) == 2                       # same name twice: still two files
    assert all("/" not in f and (tmp_path / "out" / f).exists() for f in files)


def test_provenance_travels_with_the_result(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", banded_albedo())
    (tmp_path / "knight.provenance.json").write_text(json.dumps({
        "schema_version": 1, "classification": "commercial-conditional",
        "license": {"name": "MIT"}, "models": [{"id": "pixal3d", "license": "MIT"}]}))
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    record = json.loads((tmp_path / "out" / "knight.provenance.json").read_text())
    assert record["models"] == [{"id": "pixal3d", "license": "MIT"}]   # no model added
    assert record["classification"] == "commercial-conditional"
    assert record["operation"].startswith("Game materials")
    assert record["input"]["sha256"] == m.sha256_file(asset)
    assert record["output"]["sha256"] == m.sha256_file(result.glb_path)


def test_a_source_without_provenance_says_its_licence_is_unknown(banded, tmp_path):
    m.make_game_materials(banded, tmp_path / "out", samples=20000)
    record = json.loads((tmp_path / "out" / "banded.provenance.json").read_text())
    assert record["classification"] == "unknown"
    assert record["models"] == []


def test_texture_transform_is_read_and_carried_to_every_new_map(tmp_path):
    """Scaling v by 0.3 shows only the top (red) band: the analysis must see that too."""
    squeezed = make_glb(tmp_path / "squeezed.glb", banded_albedo(),
                        base_extensions={"KHR_texture_transform": {"scale": [1.0, 0.3]}})
    assert statuses(m.run_checks(squeezed, samples=20000))["tint_zones"] == "fail"
    transform = {"KHR_texture_transform": {"offset": [0.0, 0.0], "scale": [1.0, 0.999]}}
    nearly = make_glb(tmp_path / "nearly.glb", banded_albedo(), base_extensions=transform)
    result = m.make_game_materials(nearly, tmp_path / "out", samples=20000)
    material = EditGLB(result.glb_path.read_bytes()).document["materials"][0]
    for info in (material["normalTexture"], material["pbrMetallicRoughness"]["metallicRoughnessTexture"],
                 material["extras"]["assetfurnace"]["tintMask"]):
        assert info["extensions"] == transform


def test_uv_transform_follows_the_three_js_reading_of_the_extension():
    uv = np.array([[0.5, 0.25]])
    moved = m.transform_uv(uv, {"offset": [0.1, 0.2], "scale": [2.0, 4.0], "rotation": 0.0})
    np.testing.assert_allclose(moved, [[1.1, 1.2]])
    turned = m.transform_uv(np.array([[1.0, 0.0]]), {"rotation": np.pi / 2})
    np.testing.assert_allclose(turned, [[0.0, -1.0]], atol=1e-12)


def test_uv_transform_rotates_before_scaling_like_three_js():
    """three.js Matrix3.setUvTransform: u' = sx (c u + s v) + tx, v' = sy (-s u + c v) + ty."""
    r, sx, sy, tx, ty = 0.7, 2.0, 0.5, 0.1, -0.3
    u, v = 0.3, 0.8
    c, s_ = np.cos(r), np.sin(r)
    expected = [[sx * (c * u + s_ * v) + tx, sy * (-s_ * u + c * v) + ty]]
    got = m.transform_uv(np.array([[u, v]]), {"rotation": r, "scale": [sx, sy], "offset": [tx, ty]})
    np.testing.assert_allclose(got, expected)


def test_height_does_not_leak_between_nearby_uv_islands():
    """A bright island 4 texels from a dark one must not tilt the dark one's border."""
    lum = np.zeros((64, 64))
    coverage = np.zeros((64, 64), bool)
    coverage[8:56, 8:28] = True
    coverage[8:56, 32:56] = True
    lum[:, :30] = 0.9
    lum[:, 30:] = 0.1
    normal = m.normal_from_albedo(lum, coverage, strength=64.0).astype(int)
    assert np.abs(normal[coverage][:, :2] - 128).max() <= 1


# --------------------------------------------------------------------------- second review

def test_provenance_keeps_the_classification_finalize_output_stores_under_output(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", banded_albedo())
    (tmp_path / "knight.provenance.json").write_text(json.dumps({
        "schema_version": 1, "license": {"name": "MIT"},
        "output": {"path": "x.glb", "sha256": "0", "classification": "commercial-clear"}}))
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    record = json.loads((tmp_path / "out" / "knight.provenance.json").read_text())
    assert record["output"]["classification"] == "commercial-clear"
    assert record["output"]["sha256"] == m.sha256_file(result.glb_path)


def test_maps_on_another_uv_set_are_refused_not_scrambled(tmp_path):
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = np.random.default_rng(7).integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "uv1.glb", banded_albedo(), mr=mr)
    document, binary = parse_glb(asset.read_bytes())
    document["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]["texCoord"] = 1
    asset.write_bytes(build_glb(document, binary))
    with pytest.raises(ValueError, match="UV"):
        m.make_game_materials(asset, tmp_path / "out", samples=20000)


def test_derived_roughness_starts_from_the_materials_own_value(tmp_path):
    """A glossy material (roughnessFactor 0.1) with no map stays glossy."""
    asset = make_glb(tmp_path / "glossy.glb", banded_albedo(), roughness_factor=0.1)
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    orm = np.asarray(Image.open(result.textures["orm"])).astype(int)
    assert result.maps["roughness"] == "derived"
    assert abs(np.median(orm[..., 1]) - 0.1 * 255) < 8


def test_derived_maps_see_the_colour_the_factor_leaves(tmp_path):
    """baseColorFactor (1, 0, 0) hides green; stripes painted only in green are not relief."""
    albedo = np.full((SIZE, SIZE, 3), 120, np.uint8)
    albedo[:, ::8, 1] = 250
    asset = make_glb(tmp_path / "red.glb", albedo)
    document, binary = parse_glb(asset.read_bytes())
    document["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 0.0, 0.0, 1.0]
    asset.write_bytes(build_glb(document, binary))
    surface = next(iter(m._surfaces(EditGLB(asset.read_bytes())).values()))
    lum = m.surface_luminance(surface)
    assert float(np.ptp(lum)) < 1e-6


def test_transparent_texels_do_not_count(tmp_path):
    """A MASK material's transparent black half is not a colour zone."""
    rgba = np.zeros((SIZE, SIZE, 4), np.uint8)
    rgba[:SIZE // 4, :, :3] = RED                 # visible: the top cap...
    rgba[SIZE // 4:SIZE // 2, :, :3] = GREEN      # ...and the band down to the equator
    rgba[:SIZE // 2, :, 3] = 255                  # the lower half is transparent black
    asset = make_glb(tmp_path / "cutout.glb", rgba)
    document, binary = parse_glb(asset.read_bytes())
    document["materials"][0]["alphaMode"] = "MASK"
    asset.write_bytes(build_glb(document, binary))
    zones = {r["check"]: r for r in m.run_checks(asset, samples=20000)}["tint_zones"]
    # Of the visible half, the cap is 29% and the band 71%; black would have been a third zone.
    assert zones["details"]["shares"] == [pytest.approx(0.71, abs=0.04), pytest.approx(0.29, abs=0.04)]


@pytest.mark.parametrize("mode, uv, expected", [
    (10497, 1.25, 0.25),        # REPEAT
    (33071, 1.25, 0.999),       # CLAMP_TO_EDGE
    (33648, 1.25, 0.75),        # MIRRORED_REPEAT
    (33648, 2.25, 0.25),
])
def test_lookup_honours_the_sampler_wrap(mode, uv, expected):
    image = np.arange(1000, dtype=np.int32).reshape(1, 1000)
    got = m._lookup(image, np.array([[uv, 0.0]]), (mode, 10497))
    assert got[0] == int(expected * 1000)


def test_coverage_wraps_tiled_triangles_under_repeat():
    uv = np.array([[1.1, 0.1], [1.4, 0.1], [1.1, 0.4]])
    covered = m.uv_coverage(uv, np.array([[0, 1, 2]]), (20, 20), (10497, 10497))
    assert covered[2:8, 2:8].any()
    clamped = m.uv_coverage(uv, np.array([[0, 1, 2]]), (20, 20), (33071, 33071))
    assert not clamped[2:8, 2:8].any()


def test_only_nodes_in_the_scene_are_analysed(tmp_path):
    asset = make_glb(tmp_path / "extra.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    document["materials"].append({"name": "hidden", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}})
    document["meshes"].append({"primitives": [{**document["meshes"][0]["primitives"][0], "material": 1}]})
    document["nodes"].append({"mesh": 1, "name": "not in any scene"})
    asset.write_bytes(build_glb(document, binary))
    assert list(m._surfaces(EditGLB(asset.read_bytes()))) == [0]


# --------------------------------------------------------------------------- third review

def test_names_differing_only_in_case_get_separate_files(tmp_path):
    """Skin and skin are one file on a case-insensitive disk (Windows, macOS)."""
    asset = two_material_glb(tmp_path / "case.glb", ["Skin", "skin"])
    result = m.make_game_materials(asset, tmp_path / "out", samples=20000)
    files = [x["textures"]["tint_mask"] for x in json.loads(result.sidecar_path.read_text())["materials"]]
    assert len({f.casefold() for f in files}) == 2


def test_strips_and_fans_are_triangles_too():
    strip = m.primitive_triangles(np.arange(5), 5)
    np.testing.assert_array_equal(strip, [[0, 1, 2], [2, 1, 3], [2, 3, 4]])
    fan = m.primitive_triangles(np.arange(5), 6)
    np.testing.assert_array_equal(fan, [[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    np.testing.assert_array_equal(m.primitive_triangles(np.arange(6), 4), [[0, 1, 2], [3, 4, 5]])
    assert m.primitive_triangles(np.arange(6), 1) is None        # lines have no surface


def test_a_strip_primitive_is_analysed(tmp_path):
    asset = make_glb(tmp_path / "strip.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    primitive = document["meshes"][0]["primitives"][0]
    # The same sphere again, as one long strip over its first faces' vertices.
    document["meshes"][0]["primitives"].append({"attributes": primitive["attributes"], "mode": 5, "material": 0})
    asset.write_bytes(build_glb(document, binary))
    surface = m._surfaces(EditGLB(asset.read_bytes()))[0]
    assert len(surface.faces) == 2 and len(surface.faces[1]) > 0


def test_provenance_describes_the_bytes_that_were_read(tmp_path, monkeypatch):
    """If the source is replaced mid-run, the record still names what produced the result."""
    asset = make_glb(tmp_path / "knight.glb", banded_albedo())
    (tmp_path / "knight.provenance.json").write_text(json.dumps({"classification": "commercial-clear"}))
    original = m.sha256_file(asset)
    analyse = m._analyse

    def swap_midway(*args, **kwargs):
        result = analyse(*args, **kwargs)
        make_glb(asset, banded_albedo(colours=(BLUE, GREEN, RED)))
        (tmp_path / "knight.provenance.json").write_text(json.dumps({"classification": "research-only"}))
        return result

    monkeypatch.setattr(m, "_analyse", swap_midway)
    m.make_game_materials(asset, tmp_path / "out", samples=20000)
    record = json.loads((tmp_path / "out" / "knight.provenance.json").read_text())
    assert record["input"]["sha256"] == original
    assert record["classification"] == "commercial-clear"
