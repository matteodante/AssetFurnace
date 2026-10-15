"""glTF validity through Khronos' glTF-Validator: report parsing always, the real binary when installed."""

from __future__ import annotations

import json

import _check_fixtures as fx
import pytest

from image_to_3dlab import checks
from image_to_3dlab.checks import asset as asset_module
from image_to_3dlab.checks import validator

# Trimmed from a real 2.0.0-dev.3.10 report on a GLB whose index accessor overruns its buffer.
BROKEN_REPORT = {
    "validatorVersion": "2.0.0-dev.3.10",
    "issues": {"numErrors": 1, "numWarnings": 1, "numInfos": 2, "numHints": 0, "messages": [
        {"code": "ACCESSOR_TOTAL_OFFSET_ALIGNMENT", "message": "info message", "severity": 2, "pointer": "/x"},
        {"code": "ACCESSOR_TOO_LONG", "severity": 0, "pointer": "/accessors/0",
         "message": "Accessor (offset: 0, length: 883200) does not fit referenced bufferView [0] length 17664."},
        {"code": "UNUSED_OBJECT", "message": "This object may be unused.", "severity": 1, "pointer": "/y"},
    ], "truncated": False},
}


def test_errors_fail_and_name_the_first_problem():
    found = validator.grade(BROKEN_REPORT)
    assert found["status"] == "fail"
    assert found["value"] == {"errors": 1, "warnings": 1}
    assert "ACCESSOR_TOO_LONG" in found["message"]
    assert [m["severity"] for m in found["details"]["messages"]] == ["error", "warning"]


def test_warnings_alone_warn_and_a_clean_report_passes():
    warn = json.loads(json.dumps(BROKEN_REPORT))
    warn["issues"]["numErrors"] = 0
    assert validator.grade(warn)["status"] == "warn"
    assert validator.grade({"issues": {"numErrors": 0, "numWarnings": 0, "messages": []}})["status"] == "pass"


def test_without_the_validator_the_check_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_module, "find_validator", lambda *a, **k: None)
    report = checks.run_checks(checks.Asset(glb=fx.clean(tmp_path / "a.glb")), only=["gltf.valid"])
    assert report["results"] == []
    assert "bootstrap_gltf_validator" in report["skipped"][0]["reason"]


def test_find_validator_honours_the_override(tmp_path):
    binary = tmp_path / "gltf_validator.exe"
    binary.write_bytes(b"")
    assert asset_module.find_validator({"I2L_GLTF_VALIDATOR": str(binary)}) == binary
    assert asset_module.find_validator({"I2L_GLTF_VALIDATOR": str(tmp_path / "nope")}) is None
    (tmp_path / "vendor" / "gltf-validator").mkdir(parents=True)
    (tmp_path / "vendor" / "gltf-validator" / "gltf_validator").write_bytes(b"")
    assert asset_module.find_validator({}, repo=tmp_path).name == "gltf_validator"


needs_validator = pytest.mark.skipif(asset_module.find_validator() is None,
                                     reason="glTF-Validator not installed")


@needs_validator
def test_the_real_validator_passes_a_clean_asset(tmp_path):
    report = checks.run_checks(checks.Asset(glb=fx.clean(tmp_path / "a.glb")), only=["gltf.valid"])
    assert report["results"][0]["status"] in ("pass", "warn"), report["results"][0]
    assert report["results"][0]["value"]["errors"] == 0


@needs_validator
def test_the_real_validator_stops_a_broken_accessor(tmp_path):
    broken = fx.broken_accessor(fx.clean(tmp_path / "a.glb"), tmp_path / "broken.glb")
    report = checks.run_checks(checks.Asset(glb=broken), only=["gltf.valid"])
    assert report["results"][0]["status"] == "fail"
    assert report["results"][0]["value"]["errors"] >= 1
