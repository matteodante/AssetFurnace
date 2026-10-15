"""Fakes shared by the batch tests: a generator that runs in microseconds and needs no GPU.

The fake generator writes a small, deterministic "GLB" derived from the input image and
the parameters, so two runs with the same inputs produce the same bytes and a batch can be
compared with single runs byte for byte. Nothing here ships: the generator and its licence
profile are installed per test with `monkeypatch`.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

from image_to_3dlab import manifest_run, provenance
from image_to_3dlab.provenance import LicenseProfile

FAKE_PROFILE = LicenseProfile(
    classification="commercial-clear",
    folder="clear",
    license_name="Test licence",
    license_url="https://example.invalid/licence",
    conditions=(),
)


def tiny_glb(path: Path, note: str = "") -> Path:
    """A minimal but whole binary glTF 2: header, length and a JSON chunk. `note` makes
    two files' bytes differ."""
    document = json.dumps({"asset": {"version": "2.0"}, "extras": {"note": note}}).encode()
    document += b" " * (-len(document) % 4)
    body = struct.pack("<II", len(document), 0x4E4F534A) + document
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


def tiny_png(path: Path, seed: int = 0) -> Path:
    """A valid 1x1 PNG whose colour depends on `seed`, so every row has its own bytes."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    pixel = bytes([0, seed % 256, (seed * 7) % 256, (seed * 13) % 256])
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(pixel))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


class FakeGenerator:
    """Counts its calls and can raise on a chosen image, to simulate an interruption."""

    def __init__(self, interrupt_on: str | None = None, fail_on: str | None = None):
        self.calls: list[str] = []
        self.interrupt_on = interrupt_on
        self.fail_on = fail_on

    def run(self, images, working, parameters, log):
        stem = images[0].stem
        self.calls.append(stem)
        if stem == self.interrupt_on:
            self.interrupt_on = None   # once: the resumed run gets through
            raise KeyboardInterrupt
        if stem == self.fail_on:
            raise RuntimeError(f"fake failure on {stem}")
        digest = hashlib.sha256(
            images[0].read_bytes() + json.dumps(parameters, sort_keys=True).encode()
        ).digest()
        working.mkdir(parents=True, exist_ok=True)
        (working / "fake.log").write_text(f"generated {stem}\n")
        out = working / f"{stem}.glb"
        out.write_bytes(b"glTF" + digest)
        record = working / f"{stem}.json"
        record.write_text(json.dumps({"backend": "fake", "parameters": parameters}))
        views = working / f"{stem}.svviews"
        views.mkdir(exist_ok=True)
        (views / "transforms.json").write_text(json.dumps({"camera": stem}))
        return manifest_run.Generated(glb=out, parameters=dict(parameters),
                                      companions={".json": record, ".svviews": views})


def install_fake_generator(monkeypatch, generator: FakeGenerator | None = None,
                           ready: bool = True) -> FakeGenerator:
    generator = generator or FakeGenerator()
    monkeypatch.setitem(provenance.LICENSES, "fake", FAKE_PROFILE)
    # The provenance record asks git for this repo's commit on every output: seconds each
    # on a busy machine, and not what these tests are about.
    monkeypatch.setattr(provenance, "_pipeline_revision", lambda: None)
    monkeypatch.setattr(provenance, "_git_revision", lambda repo: None)
    monkeypatch.setitem(manifest_run.GENERATORS, "fake", manifest_run.Generator(
        name="fake",
        run=generator.run,
        readiness=lambda: [] if ready else ["fake weights missing: run bootstrap_fake.py"],
        parameters=frozenset({"seed"}),
    ))
    return generator


class FakeFinish:
    """A finish step that appends a marker to the asset and writes a derived sidecar."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[str] = []
        self.fail_on = fail_on

    def run(self, ctx):
        from image_to_3dlab.batch_steps import StepResult

        self.calls.append(ctx.row.id)
        if ctx.row.id == self.fail_on:
            return StepResult(ok=False, reason="mesh exploded")
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        out = ctx.step_dir / f"{ctx.row.id}.glb"
        out.write_bytes(ctx.asset.read_bytes() + b"finished")
        sidecar = out.with_suffix(".provenance.json")
        sidecar.write_text(json.dumps({"source": str(ctx.sidecar)}))
        return StepResult(asset=out, sidecar=sidecar, files=[out, sidecar])


def fake_step(run, preflight=None):
    namespace = SimpleNamespace(run=run)
    if preflight is not None:
        namespace.preflight = preflight
    return namespace


def write_spec(root: Path, rows: int = 10, steps=("generate", "finish"), **top) -> Path:
    """A JSON batch spec with `rows` rows, each with its own image under `root/refs`."""
    spec = {
        "schema_version": 1,
        "name": top.pop("name", "test-batch"),
        "use_case": "game",
        "distribution": "private",
        "commercial_intent": True,
        "license_policy": {"allow_conditional": True},
        "output": {"directory": "out"},
        "defaults": {"backend": "fake", "kind": "prop", "faces": 5000,
                     "steps": list(steps), "parameters": {"seed": 7}},
        "rows": [],
        **top,
    }
    for index in range(1, rows + 1):
        name = f"row{index:02d}"
        tiny_png(root / "refs" / f"{name}.png", seed=index)
        spec["rows"].append({"id": f"SM_{name}", "images": [f"refs/{name}.png"]})
    path = root / "spec.json"
    path.write_text(json.dumps(spec, indent=2))
    return path
