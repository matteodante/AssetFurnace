"""The CC0 clip library: what it fetches, from where, and that nothing unverified is kept."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import re

import pytest

from image_to_3dlab import clip_library as cl


def test_every_file_is_pinned_to_a_commit_with_its_hash_and_size():
    assert re.fullmatch(r"[0-9a-f]{40}", cl.MESH2MOTION["commit"])
    for name, (sha, size) in cl.MESH2MOTION["files"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", sha) and size > 0
        assert cl.MESH2MOTION["commit"] in cl.file_url(name)
    assert cl.TOTAL_BYTES == sum(size for _, size in cl.MESH2MOTION["files"].values())


def test_only_the_cc0_animation_files_are_fetched():
    """The repository also ships a CMU sample and per-file licensed variations; neither
    is covered by its CC0 notice, so neither may ever be in the download list."""
    assert cl.MESH2MOTION["license"] == "CC0-1.0"
    assert cl.MESH2MOTION["license_url"].endswith(f"{cl.MESH2MOTION['commit']}/LICENSE-CC0.MD")
    for name in cl.MESH2MOTION["files"]:
        assert name.startswith("static/animations/") and name.endswith("-animations.glb")
        assert "CarnegieMellon" not in name and "variation" not in name


def test_an_empty_folder_is_not_installed(tmp_path):
    assert not cl.installed(tmp_path)


def _fake(files: dict[str, bytes]):
    table = {name: (hashlib.sha256(data).hexdigest(), len(data)) for name, data in files.items()}

    def opener(url):
        name = next(n for n in files if url.endswith(n))
        return io.BytesIO(files[name])

    return table, opener


def test_a_verified_download_is_kept_with_its_source_record(tmp_path, monkeypatch):
    import bootstrap_clip_library as boot

    table, opener = _fake({"static/animations/fox-animations.glb": b"fox", "static/animations/snake-animations.glb": b"sn"})
    monkeypatch.setitem(cl.MESH2MOTION, "files", table)
    boot.download(tmp_path, opener=opener)
    assert (tmp_path / "fox-animations.glb").read_bytes() == b"fox"
    record = json.loads((tmp_path / cl.SOURCE_FILE).read_text(encoding="utf-8"))
    assert record["license"] == "CC0-1.0" and record["commit"] == cl.MESH2MOTION["commit"]
    assert record["files"]["fox-animations.glb"]["sha256"] == table["static/animations/fox-animations.glb"][0]
    assert cl.installed(tmp_path)


def test_a_file_with_the_wrong_hash_is_never_kept(tmp_path, monkeypatch):
    import bootstrap_clip_library as boot

    table, opener = _fake({"static/animations/fox-animations.glb": b"fox"})
    table["static/animations/fox-animations.glb"] = ("0" * 64, 3)
    monkeypatch.setitem(cl.MESH2MOTION, "files", table)
    with pytest.raises(SystemExit, match="checksum"):
        boot.download(tmp_path, opener=opener)
    assert list(tmp_path.iterdir()) == []


def test_nothing_is_fetched_without_a_yes(tmp_path, monkeypatch, capsys):
    import bootstrap_clip_library as boot

    monkeypatch.setattr(cl, "pack_dir", lambda: tmp_path / "mesh2motion")
    monkeypatch.setattr(boot, "download", lambda *a, **k: pytest.fail("downloaded without a yes"))
    assert boot.main([], ask=lambda _: "n") == 1
    out = capsys.readouterr().out
    assert "CC0" in out and "MB" in out and cl.MESH2MOTION["commit"][:7] in out
    assert boot.main(["--check"]) == 1


def test_the_library_is_in_the_catalogue_with_its_licence_and_size():
    import backend_catalog as bc
    import download_api as dl

    entry = bc.BY_ID["clip-library"]
    assert entry.kind == "tool" and entry.license_name == "CC0-1.0"
    assert entry.bytes_expected == cl.TOTAL_BYTES
    assert entry.weights[0].path == cl.pack_dir()
    command = dl.COMMANDS["clip-library"]
    assert command[1].endswith("bootstrap_clip_library.py") and "--yes" in command


needs_pack = pytest.mark.skipif(not cl.installed(), reason="clip library not installed")


@needs_pack
@pytest.mark.parametrize("name", sorted(cl.SOURCES))
def test_every_library_chain_and_clip_is_in_its_file(name):
    from image_to_3dlab.glb_skin import SkinnedGLB

    src = cl.SOURCES[name]
    skin = SkinnedGLB(cl.pack_dir() / src["file"])
    parents, heads = skin.skeleton()
    roles = cl.source_roles(name)
    chains = [roles[k] for k in ("spine", "neck", "tail", "body") if roles[k]]
    chains += [c["bones"] for k in ("legs", "arms", "wings") for c in roles[k]]
    for bones in chains:
        for a, b in itertools.pairwise(bones):
            assert parents[b] == a, (a, b)  # joint to joint, along the tree
    for k in ("legs", "arms", "wings"):
        for c in roles[k]:
            x = heads[c["bones"][-1]] @ cl.FRAME["left"]
            assert (x > 0) == (c["side"] == "L"), c["bones"][-1]
    names = {n for n, _ in skin.animations()}
    assert {clip for clip, _ in src["clips"].values()} <= names


@needs_pack
@pytest.mark.parametrize("name", sorted(cl.SOURCES))
def test_a_library_clip_carried_onto_its_own_skeleton_comes_back(name):
    """The whole path on real data: the chains named here, sampled and retargeted onto the
    same skeleton, put every chain joint back where the clip has it."""
    import chain_retarget as cr
    import creature_motion as cm
    import numpy as np

    from image_to_3dlab.glb_skin import SkinnedGLB

    src = cl.SOURCES[name]
    path = cl.pack_dir() / src["file"]
    parents, heads = SkinnedGLB(path).skeleton()
    order = [b for b, p in parents.items() if p is None]
    for b in order:
        order += [c for c, p in parents.items() if p == b and c not in order]
    rig = cm.Rig(parents, order, heads)
    roles = cl.source_roles(name)
    target = {**roles, "family": "serpent" if name == "snake" else name, "ground": 0.0}
    clip, loop = next(iter(src["clips"].values()))
    source = cr.load_source(path, roles, clip, loop=loop)
    chain = {b for k in ("spine", "neck", "tail", "body") for b in roles[k]}
    chain |= {b for k in ("legs", "arms", "wings") for c in roles[k] for b in c["bones"]}
    size = float(np.ptp(np.array(list(heads.values())), axis=0).max())
    for pose, (turn, root) in zip(source.poses, cr.retarget(rig, target, source)):
        got = cm.posed_heads(rig, cm.accumulate(rig, turn), root)
        assert max(float(np.linalg.norm(got[b] - pose[b][:3, 3])) for b in chain) < 0.02 * size
