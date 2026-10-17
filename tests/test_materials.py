"""Tests for game-ready materials: tint mask, baked-light checks, derived maps, ORM.

Every fixture is built here, small and synthetic: a UV sphere whose texture rows map to
latitude, so a texel's surface normal is known exactly and light can be painted into
the albedo on purpose. The shipping functions are imported, never re-derived.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from image_to_3dlab import materials as m
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb, parse_glb

SIZE = 128
RED, GREEN, BLUE = (200, 40, 30), (40, 160, 60), (40, 60, 190)
LIGHT = np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])


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


def banded_albedo(colours=(RED, GREEN, BLUE), size: int = SIZE, noise: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    rows = np.arange(size) / size
    band = np.minimum((rows * len(colours)).astype(int), len(colours) - 1)
    img = np.array(colours, np.float32)[band][:, None, :].repeat(size, axis=1)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


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


def test_srgb_round_trip_and_linear_luminance():
    values = np.linspace(0, 1, 11)
    np.testing.assert_allclose(m.linear_to_srgb(m.srgb_to_linear(values)), values, atol=1e-6)
    assert m.srgb_to_linear(np.array(0.5)) == pytest.approx(0.214, abs=1e-3)
    assert m.luminance(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)


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


def test_writing_into_the_assets_own_folder_is_refused(banded):
    """The output GLB shares the source's name, so that folder would overwrite the source."""
    before = banded.read_bytes()
    with pytest.raises(ValueError):
        m.make_game_materials(banded, banded.parent)
    assert banded.read_bytes() == before


def test_orm_keeps_existing_varying_maps_and_scales_by_factors(tmp_path):
    rng = np.random.default_rng(5)
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    mr[..., 2] = 255
    asset = make_glb(tmp_path / "pbr.glb", banded_albedo(), mr=mr, metallic_factor=0.5)
    result = m.make_game_materials(asset, tmp_path / "out")
    orm = np.asarray(Image.open(result.textures["orm"]))
    assert result.maps["roughness"] == "kept"
    np.testing.assert_array_equal(orm[..., 1], mr[..., 1])
    assert np.abs(orm[..., 2].astype(int) - 128).max() <= 1     # 255 x metallicFactor 0.5
    assert orm[..., 0].min() == 255                              # no AO baked: fully open


def test_pack_orm_puts_channels_in_gltf_order():
    ao, rough, metal = np.full((2, 2), 0.25), np.full((2, 2), 0.5), np.full((2, 2), 1.0)
    orm = m.pack_orm(ao, rough, metal)
    assert orm.dtype == np.uint8
    assert tuple(orm[0, 0]) == (64, 128, 255)


def test_rewritten_glb_keeps_geometry_rig_and_animation(tmp_path):
    source = make_glb(tmp_path / "rigged.glb", banded_albedo(), rigged=True)
    result = m.make_game_materials(source, tmp_path / "out")
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


def test_occlusion_bake_is_packed_and_declared(banded, tmp_path):
    ao = np.full((SIZE, SIZE), 200, np.uint8)
    Image.fromarray(ao).save(tmp_path / "ao.png")
    result = m.make_game_materials(banded, tmp_path / "out", ao=tmp_path / "ao.png")
    material = EditGLB(result.glb_path.read_bytes()).document["materials"][0]
    assert material["occlusionTexture"] == material["pbrMetallicRoughness"]["metallicRoughnessTexture"]
    assert np.asarray(Image.open(result.textures["orm"]))[..., 0].max() == 200


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
    result = m.make_game_materials(asset, tmp_path / "out")
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


def test_packed_occlusion_already_in_the_asset_is_kept(tmp_path):
    """An ORM input already carries AO in red; without a new bake it must survive."""
    rng = np.random.default_rng(6)
    orm = np.zeros((SIZE, SIZE, 3), np.uint8)
    orm[..., 0] = rng.integers(90, 200, (SIZE, SIZE))
    orm[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "orm.glb", banded_albedo(), mr=orm, occlusion_from_mr=True)
    result = m.make_game_materials(asset, tmp_path / "out")
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
    result = m.make_game_materials(asset, tmp_path / "out")
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
    result = m.make_game_materials(asset, tmp_path / "out")
    sidecar = json.loads(result.sidecar_path.read_text())
    files = [x["textures"]["orm"] for x in sidecar["materials"]]
    assert len(set(files)) == 2                       # same name twice: still two files
    assert all("/" not in f and (tmp_path / "out" / f).exists() for f in files)


def test_provenance_travels_with_the_result(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", banded_albedo())
    (tmp_path / "knight.provenance.json").write_text(json.dumps({
        "schema_version": 1, "classification": "commercial-conditional",
        "license": {"name": "MIT"}, "models": [{"id": "pixal3d", "license": "MIT"}]}))
    result = m.make_game_materials(asset, tmp_path / "out")
    record = json.loads((tmp_path / "out" / "knight.provenance.json").read_text())
    assert record["models"] == [{"id": "pixal3d", "license": "MIT"}]   # no model added
    assert record["classification"] == "commercial-conditional"
    assert record["operation"].startswith("Game materials")
    assert record["input"]["sha256"] == m.sha256_file(asset)
    assert record["output"]["sha256"] == m.sha256_file(result.glb_path)


def test_a_source_without_provenance_says_its_licence_is_unknown(banded, tmp_path):
    m.make_game_materials(banded, tmp_path / "out")
    record = json.loads((tmp_path / "out" / "banded.provenance.json").read_text())
    assert record["classification"] == "unknown"
    assert record["models"] == []


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


def test_provenance_keeps_the_classification_finalize_output_stores_under_output(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", banded_albedo())
    (tmp_path / "knight.provenance.json").write_text(json.dumps({
        "schema_version": 1, "license": {"name": "MIT"},
        "output": {"path": "x.glb", "sha256": "0", "classification": "commercial-clear"}}))
    result = m.make_game_materials(asset, tmp_path / "out")
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
        m.make_game_materials(asset, tmp_path / "out")


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


def test_names_differing_only_in_case_get_separate_files(tmp_path):
    """Skin and skin are one file on a case-insensitive disk (Windows, macOS)."""
    asset = two_material_glb(tmp_path / "case.glb", ["Skin", "skin"])
    result = m.make_game_materials(asset, tmp_path / "out")
    files = [x["textures"]["orm"] for x in json.loads(result.sidecar_path.read_text())["materials"]]
    assert len({f.casefold() for f in files}) == 2


def test_strips_and_fans_are_triangles_too():
    strip = m.primitive_triangles(np.arange(5), 5)
    np.testing.assert_array_equal(strip, [[0, 1, 2], [2, 1, 3], [2, 3, 4]])
    fan = m.primitive_triangles(np.arange(5), 6)
    np.testing.assert_array_equal(fan, [[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    np.testing.assert_array_equal(m.primitive_triangles(np.arange(6), 4), [[0, 1, 2], [3, 4, 5]])
    assert m.primitive_triangles(np.arange(6), 1) is None        # lines have no surface


def test_retained_maps_with_another_wrap_are_refused(tmp_path):
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = np.random.default_rng(8).integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "wrap.glb", banded_albedo(), mr=mr)
    document, binary = parse_glb(asset.read_bytes())
    document["samplers"].append({"wrapS": 33071, "wrapT": 33071})
    document["textures"][1]["sampler"] = 1
    asset.write_bytes(build_glb(document, binary))
    with pytest.raises(ValueError, match="UV"):
        m.make_game_materials(asset, tmp_path / "out")


def test_an_ao_bake_cannot_follow_a_transformed_base_colour(tmp_path):
    """blender_bake_ao.py bakes on the raw UVs; the ORM would carry the base's transform."""
    Image.fromarray(np.full((SIZE, SIZE), 200, np.uint8)).save(tmp_path / "ao.png")
    asset = make_glb(tmp_path / "moved.glb", banded_albedo(),
                     base_extensions={"KHR_texture_transform": {"offset": [0.1, 0.0]}})
    with pytest.raises(ValueError, match="AO"):
        m.make_game_materials(asset, tmp_path / "out", ao=tmp_path / "ao.png")


def test_unlit_materials_are_refused(tmp_path):
    asset = make_glb(tmp_path / "unlit.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    document["materials"][0]["extensions"] = {"KHR_materials_unlit": {}}
    document["extensionsUsed"] = ["KHR_materials_unlit"]
    asset.write_bytes(build_glb(document, binary))
    with pytest.raises(ValueError, match="unlit"):
        m.make_game_materials(asset, tmp_path / "out")


def test_a_webp_requirement_goes_when_its_last_texture_does(tmp_path):
    rng = np.random.default_rng(9)
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "webp.glb", banded_albedo(), mr=mr)
    document, binary = parse_glb(asset.read_bytes())
    document["textures"][1] = {"sampler": 0, "extensions": {"EXT_texture_webp": {"source": 1}}}
    document["extensionsUsed"] = ["EXT_texture_webp"]
    document["extensionsRequired"] = ["EXT_texture_webp"]
    asset.write_bytes(build_glb(document, binary))
    result = m.make_game_materials(asset, tmp_path / "out")
    out = EditGLB(result.glb_path.read_bytes()).document
    assert "EXT_texture_webp" not in out.get("extensionsRequired", [])
    assert "EXT_texture_webp" not in out.get("extensionsUsed", [])


def test_the_sidecar_records_the_source_hash(banded, tmp_path):
    result = m.make_game_materials(banded, tmp_path / "out")
    assert json.loads(result.sidecar_path.read_text())["source_sha256"] == m.sha256_file(banded)


def test_a_texture_also_used_outside_materials_is_never_overwritten(tmp_path):
    """EXT_structural_metadata can point at the same texture: replacing it would corrupt that data."""
    rng = np.random.default_rng(10)
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "meta.glb", banded_albedo(), mr=mr)
    document, binary = parse_glb(asset.read_bytes())
    document["extensions"] = {"EXT_structural_metadata": {"propertyTextures": [
        {"class": "c", "properties": {"p": {"index": 1, "channels": [1]}}}]}}
    asset.write_bytes(build_glb(document, binary))
    result = m.make_game_materials(asset, tmp_path / "out")
    out = EditGLB(result.glb_path.read_bytes())
    image = out.document["images"][out.document["textures"][1]["source"]]
    view = out.document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    kept = np.asarray(Image.open(io.BytesIO(out.binary[start:start + view["byteLength"]])).convert("RGB"))
    np.testing.assert_array_equal(kept, mr)
    assert out.document["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"] != 1


def test_one_ao_image_is_refused_for_several_material_atlases(tmp_path):
    Image.fromarray(np.full((SIZE, SIZE), 200, np.uint8)).save(tmp_path / "ao.png")
    asset = two_material_glb(tmp_path / "two.glb", ["head", "body"])
    with pytest.raises(ValueError, match="AO"):
        m.make_game_materials(asset, tmp_path / "out", ao=tmp_path / "ao.png")


def test_suffixed_names_never_land_on_a_name_already_taken(tmp_path):
    asset = two_material_glb(tmp_path / "three.glb", ["foo-2", "foo"])
    document, binary = parse_glb(asset.read_bytes())
    primitive = document["meshes"][0]["primitives"][0]
    document["meshes"][0]["primitives"].append({**primitive, "material": 2})
    document["materials"].append({"name": "foo", "pbrMetallicRoughness": dict(document["materials"][0]["pbrMetallicRoughness"])})
    asset.write_bytes(build_glb(document, binary))
    result = m.make_game_materials(asset, tmp_path / "out")
    files = [x["textures"]["orm"].casefold() for x in json.loads(result.sidecar_path.read_text())["materials"]]
    assert len(set(files)) == 3


@pytest.mark.parametrize("extension", ["EXT_meshopt_compression", "KHR_draco_mesh_compression"])
def test_compressed_geometry_is_refused_before_it_is_misread(tmp_path, extension):
    asset = make_glb(tmp_path / "packed.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    document["extensionsUsed"] = [extension]
    document["extensionsRequired"] = [extension]
    asset.write_bytes(build_glb(document, binary))
    with pytest.raises(ValueError, match="compress"):
        m.make_game_materials(asset, tmp_path / "out")


def test_an_image_sharing_its_bytes_with_another_is_not_overwritten(tmp_path):
    rng = np.random.default_rng(11)
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "shared.glb", banded_albedo(), mr=mr)
    document, binary = parse_glb(asset.read_bytes())
    document["images"].append({"bufferView": document["images"][1]["bufferView"], "mimeType": "image/png"})
    document["textures"].append({"source": 2, "sampler": 0})
    document["materials"][0]["emissiveTexture"] = {"index": 2}
    asset.write_bytes(build_glb(document, binary))
    result = m.make_game_materials(asset, tmp_path / "out")
    out = EditGLB(result.glb_path.read_bytes())
    view = out.document["bufferViews"][out.document["images"][2]["bufferView"]]
    start = view.get("byteOffset", 0)
    kept = np.asarray(Image.open(io.BytesIO(out.binary[start:start + view["byteLength"]])).convert("RGB"))
    np.testing.assert_array_equal(kept, mr)


def test_provenance_is_written_before_the_sidecar_publishes_the_run(banded, tmp_path, monkeypatch):
    """If the licence record cannot be written, no sidecar may say the run is done."""
    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(m, "write_provenance", full_disk)
    with pytest.raises(OSError):
        m.make_game_materials(banded, tmp_path / "out")
    assert not (tmp_path / "out" / "banded.materials.json").exists()
    assert not (tmp_path / "out" / "banded.glb").exists()


@pytest.mark.parametrize("prefix", ["../escape", "a/b", "C:/x", ".."])
def test_a_texture_prefix_cannot_leave_the_output_folder(banded, tmp_path, prefix):
    with pytest.raises(ValueError, match="prefix"):
        m.make_game_materials(banded, tmp_path / "out", texture_prefix=prefix)


def test_extras_that_are_not_an_object_are_kept(tmp_path):
    asset = make_glb(tmp_path / "extras.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    document["materials"][0]["extras"] = ["their", "data"]
    asset.write_bytes(build_glb(document, binary))
    result = m.make_game_materials(asset, tmp_path / "out")
    assert EditGLB(result.glb_path.read_bytes()).document["materials"][0]["extras"] == ["their", "data"]


# --------------------------------------------------------------------------- v2: nothing invented

def test_no_normal_map_is_invented(banded, tmp_path):
    """The normal map is the finish step's bake from the dense mesh, or none at all."""
    result = m.make_game_materials(banded, tmp_path / "out")
    material = EditGLB(result.glb_path.read_bytes()).document["materials"][0]
    assert "normalTexture" not in material
    assert result.maps["normal"] == "none"
    assert not list((tmp_path / "out").glob("*_N.png"))


def test_without_a_map_roughness_and_metal_are_the_factors(tmp_path):
    asset = make_glb(tmp_path / "plain.glb", banded_albedo(), metallic_factor=0.25, roughness_factor=0.6)
    result = m.make_game_materials(asset, tmp_path / "out")
    orm = np.asarray(Image.open(result.textures["orm"]))
    assert result.maps["roughness"] == result.maps["metallic"] == "factor"
    assert set(np.unique(orm[..., 1])) == {153} and set(np.unique(orm[..., 2])) == {64}


def test_the_sidecar_says_what_was_made_and_leaves_zones_for_later(banded, tmp_path):
    result = m.make_game_materials(banded, tmp_path / "out", texture_prefix="T_Probe")
    sidecar = json.loads(result.sidecar_path.read_text())
    assert sidecar["schema"] == m.SIDECAR_SCHEMA
    assert sidecar["zones"] == [] and sidecar["checks"] == []
    assert sidecar["textures"]["orm"] == "T_Probe_ORM.png"
    assert sidecar["textures"]["orm_channels"] == {"R": "occlusion", "G": "roughness", "B": "metallic"}
    assert sidecar["untextured"] == []


def test_islands_two_texels_apart_keep_their_own_numbers():
    raw = np.zeros((32, 32), bool)
    raw[4:28, 4:14] = True
    raw[4:28, 16:28] = True                 # a two-texel gutter at columns 14-15
    grown = ndimage.binary_dilation(raw, iterations=1)
    labels = m.island_labels(grown, core=raw)
    assert labels[10, 10] != labels[10, 20]
    assert labels[10, 14] == labels[10, 10] and labels[10, 15] == labels[10, 20]


# --------------------------------------------------------------------------- script

def test_script_writes_the_result_and_refuses_with_a_reason(banded, tmp_path, capsys):
    import game_materials

    assert game_materials.main([str(banded), str(tmp_path / "ok")]) == 0
    assert (tmp_path / "ok" / "banded.glb").exists()
    assert game_materials.main([str(banded), str(banded.parent)]) == 1      # its own folder
    assert "refused" in capsys.readouterr().out


# --------------------------------------------------------------------------- PR 11 review

def test_a_failed_rerun_leaves_the_published_result_untouched(banded, tmp_path, monkeypatch):
    """Everything is staged first; the published files change only when the whole run succeeds."""
    out = tmp_path / "out"
    m.make_game_materials(banded, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(m, "write_provenance", full_disk)
    with pytest.raises(OSError):
        m.make_game_materials(banded, out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_ktx2_textures_are_refused_with_a_reason(tmp_path):
    asset = make_glb(tmp_path / "basis.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    document["images"][0]["mimeType"] = "image/ktx2"
    document["textures"][0] = {"sampler": 0, "extensions": {"KHR_texture_basisu": {"source": 0}}}
    document["extensionsUsed"] = document["extensionsRequired"] = ["KHR_texture_basisu"]
    asset.write_bytes(build_glb(document, binary))
    with pytest.raises(ValueError, match="KTX2"):
        m.make_game_materials(asset, tmp_path / "out")


def test_optional_meshopt_is_refused_too(tmp_path):
    """Its payload lives outside the core bufferViews and would be lost when the buffer is relaid."""
    asset = make_glb(tmp_path / "meshopt.glb", banded_albedo())
    document, binary = parse_glb(asset.read_bytes())
    document["extensionsUsed"] = ["EXT_meshopt_compression"]
    asset.write_bytes(build_glb(document, binary))
    with pytest.raises(ValueError, match="meshopt"):
        m.make_game_materials(asset, tmp_path / "out")


def test_a_texture_used_by_a_mesh_extension_is_never_overwritten(tmp_path):
    rng = np.random.default_rng(12)
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1] = rng.integers(40, 220, (SIZE, SIZE))
    asset = make_glb(tmp_path / "features.glb", banded_albedo(), mr=mr)
    document, binary = parse_glb(asset.read_bytes())
    document["meshes"][0]["primitives"][0]["extensions"] = {
        "EXT_mesh_features": {"featureIds": [{"featureCount": 2, "texture": {"index": 1, "channels": [0]}}]}}
    asset.write_bytes(build_glb(document, binary))
    result = m.make_game_materials(asset, tmp_path / "out")
    out = EditGLB(result.glb_path.read_bytes())
    image = out.document["images"][out.document["textures"][1]["source"]]
    view = out.document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    kept = np.asarray(Image.open(io.BytesIO(out.binary[start:start + view["byteLength"]])).convert("RGB"))
    np.testing.assert_array_equal(kept, mr)


def test_a_missing_metallic_factor_is_recorded_as_assumed_not_metal(tmp_path):
    """glTF 2.0.1: metallicFactor defaults to 1, so a viewer shows such an asset as metal.
    Treating it as not metal is a choice, and the record says so (audit 2026-10-17)."""
    asset = make_glb(tmp_path / "plain.glb", banded_albedo(), metallic_factor=None)
    result = m.make_game_materials(asset, tmp_path / "out")
    assert result.maps["metallic"].startswith("assumed 0")
    assert "default 1" in result.maps["metallic"]
    assert result.maps["roughness"] == "factor"
