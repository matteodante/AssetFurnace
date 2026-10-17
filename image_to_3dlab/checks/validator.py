"""glTF validity, judged by Khronos' own glTF-Validator (Apache-2.0).

Install it with `python scripts/bootstrap_gltf_validator.py` or from Setup & Status. Errors
fail, warnings warn, infos and hints are ignored. Without the validator the check warns that
it did not run: never a pass, and never silent.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from image_to_3dlab.checks.registry import register, result

SEVERITY = {0: "error", 1: "warning", 2: "info", 3: "hint"}


def run_validator(binary: Path, glb: Path, timeout: int = 120) -> dict:
    """The validator's JSON report for `glb` (printed to stdout with `-o`)."""
    done = subprocess.run([str(binary), "-o", str(glb)], capture_output=True, text=True,
                          timeout=timeout, check=False, encoding="utf-8", errors="replace")
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"glTF-Validator gave no report: {(done.stderr or done.stdout)[:300]}") from exc


def grade(report: dict, keep: int = 10) -> dict:
    """Turn a validator report into a check result."""
    issues = report.get("issues", {})
    errors, warnings = int(issues.get("numErrors", 0)), int(issues.get("numWarnings", 0))
    status = "fail" if errors else "warn" if warnings else "pass"
    messages = [{"code": m.get("code"), "severity": SEVERITY.get(m.get("severity"), m.get("severity")),
                 "message": m.get("message"), "pointer": m.get("pointer")}
                for m in issues.get("messages", []) if m.get("severity") in (0, 1)][:keep]
    first = f": {messages[0]['code']} {messages[0]['message']}" if messages else ""
    return result("gltf.valid", status, {"errors": errors, "warnings": warnings},
                  {"errors": 0, "warnings": 0},
                  f"glTF-Validator found {errors} error(s) and {warnings} warning(s){first}",
                  {"validator": report.get("validatorVersion"), "messages": messages})


@register("gltf.valid", needs=("glb",))
def valid(asset, config):
    if asset.validator_path is None:
        return result("gltf.valid", "warn", None, {"errors": 0, "warnings": 0},
                      "not run: glTF-Validator is not installed (Setup & Status, or "
                      "python scripts/bootstrap_gltf_validator.py)")
    return grade(run_validator(asset.validator_path, asset.glb, int(config.get("timeout", 120))))
