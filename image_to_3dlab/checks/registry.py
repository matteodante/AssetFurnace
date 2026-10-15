"""The one contract every check follows, the registry that holds them, and the report.

A check is `check(asset, config) -> dict` and returns exactly::

    {"check": name, "status": "pass"|"warn"|"fail", "value": ..., "threshold": ...,
     "message": text}

plus an optional ``details`` object that readers ignore if they do not know it.

Three rules keep the report honest:

* A check that lacks its inputs (no reference, no rig, no validator) is **not run**. It is
  listed under ``skipped`` with the reason, so "pass" always means "measured and good",
  never "not measured".
* A check that crashes, or returns something off contract, is recorded as **fail**. A
  check that cannot measure has not passed.
* The report's status is the worst of its results.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = ("pass", "warn", "fail")
_RANK = {status: rank for rank, status in enumerate(STATUSES)}
PROFILES = Path(__file__).resolve().parent / "profiles"
REPORT_SUFFIX = ".checks.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckSpec:
    name: str
    func: Callable[[Any, dict], dict]
    needs: tuple[str, ...] = ("glb",)


REGISTRY: dict[str, CheckSpec] = {}


def register(name: str, *, needs: Iterable[str] = ("glb",)):
    """Decorator adding a check to the registry under `name` (dotted: family.check)."""

    def decorate(func):
        if name in REGISTRY:
            raise ValueError(f"check {name!r} is already registered")
        REGISTRY[name] = CheckSpec(name, func, tuple(needs))
        return func

    return decorate


def result(check: str, status: str, value: Any, threshold: Any, message: str,
           details: dict | None = None) -> dict:
    """Build one result in the contract's shape."""
    out = {"check": check, "status": status, "value": value, "threshold": threshold,
           "message": message}
    if details:
        out["details"] = details
    return validate_result(out)


def validate_result(obj: Any) -> dict:
    """Return `obj` if it follows the contract, else raise ValueError (or TypeError) saying why."""
    if not isinstance(obj, dict):
        raise TypeError(f"a check must return a dict, got {type(obj).__name__}")
    missing = [k for k in ("check", "status", "value", "threshold", "message") if k not in obj]
    if missing:
        raise ValueError(f"result is missing {missing}")
    if not isinstance(obj["check"], str) or not obj["check"]:
        raise ValueError("'check' must be a non-empty name")
    if obj["status"] not in STATUSES:
        raise ValueError(f"'status' must be one of {STATUSES}, got {obj['status']!r}")
    if not isinstance(obj["message"], str):
        raise TypeError("'message' must be text")
    if "details" in obj and not isinstance(obj["details"], dict):
        raise TypeError("'details' must be an object")
    return obj


def grade_max(value: float, warn: float | None = None, fail: float | None = None) -> str:
    """A limit: above `fail` fails, above `warn` warns."""
    if fail is not None and value > fail:
        return "fail"
    if warn is not None and value > warn:
        return "warn"
    return "pass"


def grade_band(value: float, low: float, high: float) -> str:
    """A band: inside passes, either side fails."""
    return "pass" if low <= value <= high else "fail"


def worst(statuses: Iterable[str]) -> str:
    return max(statuses, key=_RANK.__getitem__, default="pass")


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def profile_names() -> list[str]:
    return sorted(p.stem for p in PROFILES.glob("*.json") if p.stem != "default")


def load_profile(name: str = "prop", overrides: dict | None = None) -> dict:
    """Thresholds for one kind of asset: `default.json`, then `<name>.json`, then overrides.

    `overrides` maps check names to settings, the shape a batch row uses to tighten one
    budget without copying a whole profile.
    """
    if not re.fullmatch(r"[a-z0-9_]+", name or "") or not (PROFILES / f"{name}.json").is_file():
        raise ValueError(f"unknown profile {name!r}; choose one of {profile_names()}")
    profile = json.loads((PROFILES / "default.json").read_text(encoding="utf-8"))
    profile = _merge(profile, json.loads((PROFILES / f"{name}.json").read_text(encoding="utf-8")))
    profile["name"] = name
    profile["checks"] = _merge(profile.get("checks", {}), overrides or {})
    return profile


def _apply_warn_only(found: dict) -> dict:
    if found["status"] != "fail":
        return found
    softened = dict(found)
    softened["status"] = "warn"
    softened["message"] = f"{found['message']} (warn only until calibrated)"
    return softened


def run_checks(asset, profile: str = "prop", overrides: dict | None = None,
               only: Iterable[str] | None = None, now: datetime | None = None) -> dict:
    """Run every registered check that applies to `asset`; return the report.

    The report's status is the worst result, or "warn" when every check was skipped.
    """
    settings = load_profile(profile, overrides)
    wanted = set(only) if only is not None else None
    results, skipped = [], []
    for name, spec in REGISTRY.items():
        if wanted is not None and name not in wanted:
            continue
        config = settings["checks"].get(name, {})
        if config.get("enabled") is False:
            skipped.append({"check": name, "reason": "disabled in the profile"})
            continue
        try:
            reason = asset.missing(spec.needs)
            if reason:
                skipped.append({"check": name, "reason": reason})
                continue
            found = validate_result(spec.func(asset, config))
            # warn_only softens thresholds still being calibrated, never a check that broke.
            if config.get("warn_only"):
                found = _apply_warn_only(found)
        except Exception as exc:  # noqa: BLE001 - a check that cannot measure has not passed
            found = result(name, "fail", None, None, f"check could not run: {type(exc).__name__}: {exc}")
        results.append(found)
    return {
        "schema_version": SCHEMA_VERSION,
        "asset": str(asset.glb),
        "asset_sha256": asset.sha256,
        "asset_key": asset.key,
        "reference": asset.reference_record(),
        "profile": settings["name"],
        "ran_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),  # noqa: UP017 - 3.10
        # Nothing measured is not a pass: "pass" always means measured and good.
        "status": worst(r["status"] for r in results) if results else "warn",
        "results": results,
        "skipped": skipped,
    }


def report_path(glb: Path) -> Path:
    glb = Path(glb)
    return glb.with_name(glb.stem + REPORT_SUFFIX)


def write_report(report: dict, path: Path | None = None) -> Path:
    """Write the report beside its GLB (`<name>.checks.json`) unless told otherwise."""
    target = Path(path) if path else report_path(Path(report["asset"]))
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
